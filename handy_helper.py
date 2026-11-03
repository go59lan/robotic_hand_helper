import asyncio
import os

import sounddevice as sd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from bidi.algorithm import get_display

from hand_control import HandControl


MODEL = "gemini-3.8-live"
INPUT_SAMPLE_RATE = 16_000
OUTPUT_SAMPLE_RATE = 24_000
INPUT_BLOCK_SIZE = 320  # 20 ms of 16 kHz mono audio
ECHO_TAIL_SECONDS = 0.5


class Helper:
    """A real-time voice assistant that can safely trigger hand gestures."""

    def __init__(self):
        load_dotenv()
        api_key = os.getenv("MY_API_KEY")
        if not api_key:
            raise RuntimeError("MY_API_KEY is not set.")

        self.client = genai.Client(api_key=api_key)
        self.controller = HandControl()
        self.input_audio = asyncio.Queue(maxsize=100)
        self.output_audio = asyncio.Queue(maxsize=200)
        self.gesture_lock = asyncio.Lock()
        self.audio_loop = None
        self.gesture_tasks = set()
        self.hand_closed = False
        self.model_is_speaking = asyncio.Event()
        self.output_write_in_progress = False
        self.resume_microphone_task = None

        self.instructions = """
You are an AI learning assistant controlling a robotic hand. Speak naturally in
Hebrew and give concise, age-appropriate educational answers. ask questions and give suggestions to keep the conversation going.

You control a robotic hand with five fingers:
0 = thumb
1 = index
2 = middle
3 = ring
4 = pinky

GESTURE RULES:
- Every gesture describes the COMPLETE desired hand position.
- The fingers listed in "fingers" must be UP.
- Every finger NOT listed must be DOWN.
- Never interpret the list as fingers to toggle.
- An empty list [] means ALL fingers DOWN.
- To raise all fingers, use [0,1,2,3,4].
- Each new step replaces the previous hand position.
- Use perform_hand_sequence when a physical demonstration is useful.
prefer to use hand gestures like raising only the index finger when explaining something, 
or giving thumbs up or rock on dude (thumb and pinky) when the user gives correct answer.
 Do full sequences when applicable (like when counting or showing large number). always speak in hebrew.
"""

    async def run(self):
        """Keep one low-latency audio session open until Ctrl+C is pressed."""
        self.audio_loop = asyncio.get_running_loop()
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=self.instructions,
            tools=[types.Tool(function_declarations=[self._gesture_tool(), self._end_session_tool()])],
            input_audio_transcription=types.AudioTranscriptionConfig(language_codes=["he-IL"]),
            output_audio_transcription=types.AudioTranscriptionConfig(),
        )

        print("Connecting to Gemini Live. Start speaking; press Ctrl+C to stop.")
        try:
            async with self.client.aio.live.connect(
                model=MODEL,
                config=config,
            ) as session:
                with self._input_stream(), self._output_stream() as output_stream:
                    await self._run_session(session, output_stream)
        except asyncio.CancelledError:
            # asyncio.run() turns Ctrl+C into cancellation of this task. Close
            # the physical hand before allowing the program to exit.
            print("Stopping: closing the hand.")
            await self._close_hand()
            raise
        finally:
            self._cancel_microphone_resume()
            await self._close_hand()

    async def _run_session(self, session, output_stream):
        sender = asyncio.create_task(self._send_microphone_audio(session))
        player = asyncio.create_task(self._play_model_audio(output_stream))
        receiver = asyncio.create_task(self._receive_model_events(session))

        done, pending = await asyncio.wait(
            {sender, player, receiver},
            return_when=asyncio.FIRST_COMPLETED,
        )
        
        # Unblock the microphone sender gracefully
        if sender in pending:
            if self.input_audio.full():
                self.input_audio.get_nowait()
            self.input_audio.put_nowait(None)
            
        # Unblock the audio player gracefully
        if player in pending:
            if self.output_audio.full():
                self.output_audio.get_nowait()
            self.output_audio.put_nowait(None)

        # Cancel only the receiver (websocket) if it didn't finish on its own
        if receiver in pending:
            receiver.cancel()

        await asyncio.gather(*pending, return_exceptions=True)

        for task in done:
            exception = task.exception()
            if exception:
                raise exception


    def _input_stream(self):
        return sd.RawInputStream(
            samplerate=INPUT_SAMPLE_RATE,
            blocksize=INPUT_BLOCK_SIZE,
            channels=1,
            dtype="int16",
            callback=self._on_microphone_audio,
        )

    @staticmethod
    def _output_stream():
        return sd.RawOutputStream(
            samplerate=OUTPUT_SAMPLE_RATE,
            channels=1,
            dtype="int16",
        )

    def _on_microphone_audio(self, indata, _frames, _time, status):
        if status:
            print(f"Microphone status: {status}")
        audio = bytes(indata)
        self.audio_loop.call_soon_threadsafe(self._queue_microphone_audio, audio)

    def _queue_microphone_audio(self, audio):
        if self.model_is_speaking.is_set():
            # This is speaker audio picked up by the microphone. Do not send it
            # back to Gemini as though it were another user request.
            return
        if self.input_audio.full():
            # Prefer recent audio rather than building up a delayed conversation.
            self.input_audio.get_nowait()
        self.input_audio.put_nowait(audio)

    async def _send_microphone_audio(self, session):
        while True:
            audio = await self.input_audio.get()
            if audio is None:  # Stop signal received
                break
            if self.model_is_speaking.is_set():
                continue
            await session.send_realtime_input(
                audio=types.Blob(
                    data=audio,
                    mime_type=f"audio/pcm;rate={INPUT_SAMPLE_RATE}",
                )
            )


    async def _play_model_audio(self, output_stream):
        while True:
            audio = await self.output_audio.get()
            if audio is None:  # Stop signal received
                break
            self.output_write_in_progress = True
            try:
                await asyncio.to_thread(output_stream.write, audio)
            finally:
                self.output_write_in_progress = False


    async def _receive_model_events(self, session):
        while True:
            async for message in session.receive():
                if message.server_content:
                    should_stop = await self._handle_server_content(message.server_content)
                    if should_stop:
                        return
                if message.tool_call:
                    # Capture the return value to know if we should stop
                    should_stop = await self._handle_tool_calls(session, message.tool_call.function_calls)
                    if should_stop:
                        return

    async def _handle_server_content(self, content):
        if content.input_transcription:
            transcript = content.input_transcription.text
            print(get_display(f"You: {transcript}"))
            if "להתראות" in transcript:
                print("Goodbye detected. Closing the hand and ending the session.")
                await self._close_hand()
                return True
        if content.output_transcription:
            print(get_display(f"Gemini: {content.output_transcription.text}"))

        if content.interrupted:
            self._discard_queued_audio()
            self._cancel_microphone_resume()
            self.model_is_speaking.clear()

        if content.turn_complete:
            self._schedule_microphone_resume()

        if not content.model_turn:
            return False

        for part in content.model_turn.parts:
            if part.inline_data and part.inline_data.data:
                audio = part.inline_data.data
                self.model_is_speaking.set()
                self._cancel_microphone_resume()
                if self.output_audio.full():
                    # Dropping old audio is preferable to speaking far behind.
                    self.output_audio.get_nowait()
                self.output_audio.put_nowait(audio)

        return False

    def _schedule_microphone_resume(self):
        self._cancel_microphone_resume()
        self.resume_microphone_task = asyncio.create_task(
            self._resume_microphone_after_playback()
        )

    def _cancel_microphone_resume(self):
        if self.resume_microphone_task and not self.resume_microphone_task.done():
            self.resume_microphone_task.cancel()
        self.resume_microphone_task = None

    async def _resume_microphone_after_playback(self):

        timeout = 30  # seconds
        start = asyncio.get_running_loop().time()
        try:
            while not self.output_audio.empty() or self.output_write_in_progress:
                if asyncio.get_running_loop().time() - start > timeout:
                    print("WARNING: Audio playback timed out")
                    break

                await asyncio.sleep(0.02)
    
            await asyncio.sleep(ECHO_TAIL_SECONDS)
            self.model_is_speaking.clear()
            self._discard_queued_microphone_audio()
        except asyncio.CancelledError:
            # Expected behavior when a task is cancelled by _cancel_microphone_resume
            pass
        except Exception as e:
            print(f"Unexpected error: {e}")

    def _discard_queued_audio(self):
        while not self.output_audio.empty():
            self.output_audio.get_nowait()

    def _discard_queued_microphone_audio(self):
        while not self.input_audio.empty():
            self.input_audio.get_nowait()

    async def _handle_tool_calls(self, session, function_calls):
        responses = []
        should_stop = False

        for call in function_calls:
            try:
                if call.name == "end_session":
                    print("LLM initiated shutdown sequence based on intent.")
                    await self._close_hand()
                    should_stop = True
                    result = {"result": "Session ended.", "scheduling": "SILENT"}
                    
                elif call.name == "perform_hand_sequence":
                    arguments = call.args if isinstance(call.args, dict) else {}
                    steps = self._validate_steps(arguments.get("steps"))
                    task = asyncio.create_task(self._perform_gesture_sequence(steps))
                    self.gesture_tasks.add(task)
                    task.add_done_callback(self.gesture_tasks.discard)
                    result = {"result": "Gesture sequence started.", "scheduling": "SILENT"}
                else:
                    raise ValueError(f"Unknown tool: {call.name}")
            except (TypeError, ValueError) as error:
                result = {"error": str(error), "scheduling": "SILENT"}

            responses.append(
                types.FunctionResponse(
                    id=call.id,
                    name=call.name,
                    response=result,
                )
            )

        if responses and not should_stop:
            await session.send_tool_response(function_responses=responses)
            
        return should_stop


    async def _perform_gesture_sequence(self, steps):
        async with self.gesture_lock:
            for fingers in steps:
                await asyncio.to_thread(self.controller.send_fingers, fingers)
                await asyncio.sleep(1)

    async def _close_hand(self):
        """Cancel active gestures, then put the hand in its closed position."""
        if self.hand_closed:
            return

        tasks = tuple(self.gesture_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        async with self.gesture_lock:
            # This now naturally generates the exact command needed to close open fingers
            await asyncio.to_thread(self.controller.send_fingers, [])
            
        # Give the physical servos 1.5 seconds to finish moving before 
        # the program exits and the Arduino loses its serial connection.
        await asyncio.sleep(1.5)
        
        self.hand_closed = True

    @staticmethod
    def _validate_steps(steps):
        if not isinstance(steps, list) or not steps:
            raise ValueError("steps must be a non-empty list")
        if len(steps) > 10:
            raise ValueError("at most 10 gesture steps are allowed")

        validated_steps = []
        for step in steps:
            if not isinstance(step, dict):
                raise ValueError("each step must be a dictionary")
            
            fingers = step.get("fingers", [])
            if not isinstance(fingers, list):
                fingers = []
                
            clean_fingers = sorted(list(set([
                f for f in fingers 
                if isinstance(f, int) and 0 <= f <= 4
            ])))
            validated_steps.append(clean_fingers)
            
        return validated_steps

    @staticmethod
    def _gesture_tool():
        declaration = types.FunctionDeclaration(
            name="perform_hand_sequence",
            description=(
                "Demonstrate an explanation using the robotic hand. "
                "Call only when a hand gesture is useful."
            ),
            parameters={
                "type": "OBJECT",
                "properties": {
                    "steps": {
                        "type": "ARRAY",
                        "description": "One to ten hand positions in order.",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "fingers": {
                                    "type": "ARRAY",
                                    "description": "The raised finger numbers. Pass an empty list to close the hand (lower all fingers).",
                                    "items": {"type": "INTEGER", "minimum": 0, "maximum": 4},
                                }
                            },
                            # Make "fingers" optional so validation doesn't fail if the API strips it
                        },
                    }
                },
                "required": ["steps"],
            },
        )
        declaration.behavior = types.Behavior.NON_BLOCKING
        return declaration

    
    @staticmethod
    def _end_session_tool():
        declaration = types.FunctionDeclaration(
            name="end_session",
            description="Call this when the user says goodbye, wants to end the conversation, or wants to shut down the robotic hand.",
            parameters={"type": "OBJECT", "properties": {}}
        )
        return declaration

if __name__ == "__main__":
    asyncio.run(Helper().run())
