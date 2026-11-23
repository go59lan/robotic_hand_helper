import array
import asyncio
import math
import os
import re
from pathlib import Path

import sounddevice as sd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from bidi.algorithm import get_display

from hand_control import HandControl
from screen import Screen


MODEL = "gemini-3.8-live"
# The tutor's behaviour lives in prompts/<name>.txt; HANDY_PROMPT picks one,
# so a new subject is a new text file, not a code change.
PROMPTS_DIR = Path(__file__).parent / "prompts"
DEFAULT_PROMPT = "math_he"
INPUT_SAMPLE_RATE = 16_000
OUTPUT_SAMPLE_RATE = 24_000
INPUT_BLOCK_SIZE = 320  # 20 ms of 16 kHz mono audio
ECHO_TAIL_SECONDS = 0.5
# If Gemini stops sending audio but never sends turn_complete, reopen the mic
# after this much quiet so the child isn't locked out.
MISSING_TURN_END_SECONDS = 3
PLAYBACK_STALL_SECONDS = 5  # give up waiting if playback makes no progress
SPEECH_RMS = 600  # mic loudness (int16 RMS) that counts as the child talking
LEVEL_INTERVAL_SECONDS = 0.1  # how often the screen's mic meter updates
THINKING_SILENCE_SECONDS = 0.7  # quiet time after the child spoke = "thinking"
MAX_CAPTION_LENGTH = 24
# Transcripts sometimes contain markup like "<no speech>" or "{pause}". The
# last alternative hides a tag that is still arriving ("<no").
TRANSCRIPT_MARKUP = re.compile(r"<[^<>]*>|\{[^{}]*\}|[<{][^<>{}]*$")


def load_prompt(name):
    path = PROMPTS_DIR / f"{name}.txt"
    if not path.is_file():
        available = ", ".join(sorted(p.stem for p in PROMPTS_DIR.glob("*.txt")))
        raise RuntimeError(f"Prompt {name!r} not found in {PROMPTS_DIR} (available: {available})")
    return path.read_text(encoding="utf-8")


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
        # Unbounded: the model sends speech faster than real time, and dropping
        # chunks would skip words in the middle of an answer.
        self.output_audio = asyncio.Queue()
        self.gesture_lock = asyncio.Lock()
        self.audio_loop = None
        self.gesture_tasks = set()
        self.hand_shut_down = False
        self.model_is_speaking = asyncio.Event()
        self.output_write_in_progress = False
        self.last_playback_progress = 0.0
        self.resume_microphone_task = None

        self.screen = Screen()
        self.screen_state = None
        self.child_text = ""
        self.handy_text = ""
        self.new_child_line = True
        self.new_handy_line = True
        self.heard_child = False
        self.last_level_time = 0.0
        self.last_loud_time = 0.0

        self.instructions = load_prompt(os.getenv("HANDY_PROMPT", DEFAULT_PROMPT))

    async def run(self):
        """Keep one low-latency audio session open until Ctrl+C is pressed."""
        self.audio_loop = asyncio.get_running_loop()
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=self.instructions,
            tools=[types.Tool(function_declarations=[self._gesture_tool(), self._end_session_tool()])],
            input_audio_transcription=types.AudioTranscriptionConfig(language_codes=["he-IL"]),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Aoede" # Aoede, Puck, or Charon often have good energy
                    )
                )
            )
        )

        print("Connecting to Gemini Live. Start speaking; press Ctrl+C to stop.")
        try:
            await self.screen.start()
            self._set_screen_state("connecting")
            async with self.client.aio.live.connect(
                model=MODEL,
                config=config,
            ) as session:
                with self._input_stream(), self._output_stream() as output_stream:
                    self._set_screen_state("listening")
                    await self._run_session(session, output_stream)
        except asyncio.CancelledError:
            # asyncio.run() turns Ctrl+C into cancellation of this task. Close
            # the physical hand before allowing the program to exit.
            print("Stopping: closing the hand.")
            await self._shutdown_hand()
            raise
        finally:
            self._cancel_microphone_resume()
            self._set_screen_state("bye")
            await self._shutdown_hand()
            await self.screen.close()

    def _set_screen_state(self, state):
        """Show the child whose turn it is: connecting, listening, thinking, speaking or bye."""
        if state != self.screen_state:
            self.screen_state = state
            self.screen.send("state", state)

    async def _run_session(self, session, output_stream):
        sender = asyncio.create_task(self._send_microphone_audio(session))
        player = asyncio.create_task(self._play_model_audio(output_stream))
        receiver = asyncio.create_task(self._receive_model_events(session))

        done, pending = await asyncio.wait(
            {sender, player, receiver},
            return_when=asyncio.FIRST_COMPLETED,
        )
        
        # Stop the pending mic-resume task first so it can't drain the queues
        # while we shut down.
        self._cancel_microphone_resume()

        # Cancelling the sender is safe; it only waits on the queue or the
        # websocket, never on an audio device.
        if sender in pending:
            sender.cancel()

        # Unblock the audio player gracefully
        if player in pending:
            self.output_audio.put_nowait(None)

        # Cancel only the receiver (websocket) if it didn't finish on its own
        if receiver in pending:
            receiver.cancel()

        await asyncio.gather(*pending, return_exceptions=True)

        for task in done:

            if not task.cancelled():
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
        self._update_microphone_level(audio)
        if self.input_audio.full():
            # Prefer recent audio rather than building up a delayed conversation.
            self.input_audio.get_nowait()
        self.input_audio.put_nowait(audio)

    def _update_microphone_level(self, audio):
        now = self.audio_loop.time()
        samples = array.array("h", audio)
        rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples)) if samples else 0
        if rms > SPEECH_RMS:
            self.last_loud_time = now
            if self.screen_state == "thinking":
                self._set_screen_state("listening")
        elif (self.heard_child and self.screen_state == "listening"
              and now - self.last_loud_time > THINKING_SILENCE_SECONDS):
            # The child spoke and then went quiet: Gemini is working on a reply.
            self._set_screen_state("thinking")

        if now - self.last_level_time >= LEVEL_INTERVAL_SECONDS:
            self.last_level_time = now
            self.screen.send("level", round(min(1.0, rms / (SPEECH_RMS * 4)), 2))

    async def _send_microphone_audio(self, session):
        while True:
            audio = await self.input_audio.get()
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
                self.last_playback_progress = asyncio.get_running_loop().time()


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
        if content.input_transcription and content.input_transcription.text:
            transcript = content.input_transcription.text
            print(get_display(f"You: {transcript}"))
            self._show_child_words(transcript)
        if content.output_transcription and content.output_transcription.text:
            transcript = content.output_transcription.text
            print(get_display(f"Gemini: {transcript}"))
            self._show_handy_words(transcript)

        if content.interrupted:
            self._discard_queued_audio()
            self._cancel_microphone_resume()
            self.model_is_speaking.clear()
            self._set_screen_state("listening")

        if content.turn_complete:
            self.new_handy_line = True
            self._schedule_microphone_resume(ECHO_TAIL_SECONDS)

        if not content.model_turn:
            return False

        for part in content.model_turn.parts:
            if part.inline_data and part.inline_data.data:
                audio = part.inline_data.data
                self.model_is_speaking.set()
                self.heard_child = False
                self._set_screen_state("speaking")
                # turn_complete normally reopens the mic; this is the fallback
                # in case it never arrives. Each new chunk restarts the wait.
                self._schedule_microphone_resume(MISSING_TURN_END_SECONDS)
                self.output_audio.put_nowait(audio)

        return False

    def _show_child_words(self, text):
        # Transcripts arrive in small pieces; join them into one line per turn.
        if self.new_child_line:
            self.child_text = ""
            self.new_child_line = False
        self.child_text += text
        self.heard_child = True
        self.new_handy_line = True
        self.screen.send("you", self._clean_transcript(self.child_text))

    def _show_handy_words(self, text):
        if self.new_handy_line:
            self.handy_text = ""
            self.new_handy_line = False
        self.handy_text += text
        self.new_child_line = True
        self.screen.send("handy", self._clean_transcript(self.handy_text))

    @staticmethod
    def _clean_transcript(text):
        return " ".join(TRANSCRIPT_MARKUP.sub(" ", text).split())

    def _schedule_microphone_resume(self, quiet_seconds):
        self._cancel_microphone_resume()
        self.resume_microphone_task = asyncio.create_task(
            self._resume_microphone_after_playback(quiet_seconds)
        )

    def _cancel_microphone_resume(self):
        if self.resume_microphone_task and not self.resume_microphone_task.done():
            self.resume_microphone_task.cancel()
        self.resume_microphone_task = None

    async def _resume_microphone_after_playback(self, quiet_seconds):

        loop = asyncio.get_running_loop()
        # Long answers are fine; only give up if playback stops progressing.
        self.last_playback_progress = loop.time()
        try:
            while not self.output_audio.empty() or self.output_write_in_progress:
                if loop.time() - self.last_playback_progress > PLAYBACK_STALL_SECONDS:
                    print("WARNING: Audio playback stalled")
                    break

                await asyncio.sleep(0.02)
    
            await asyncio.sleep(quiet_seconds)
            if quiet_seconds == MISSING_TURN_END_SECONDS:
                print("No turn_complete from Gemini; reopening the mic.")
            self.model_is_speaking.clear()
            self._set_screen_state("listening")
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
                    await self._shutdown_hand()
                    should_stop = True
                    result = {"result": "Session ended.", "scheduling": "SILENT"}
                    
                elif call.name == "perform_hand_sequence":
                    arguments = call.args if isinstance(call.args, dict) else {}
                    steps = self._validate_steps(arguments.get("steps"))
                    # The newest gesture wins, so the hand doesn't fall
                    # behind the speech when the model gestures often.
                    for old_task in self.gesture_tasks:
                        old_task.cancel()
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
            # Each call starts with a clean caption; steps without one keep
            # the caption of the step before (e.g. a thumbs up after "4").
            caption = ""
            for fingers, step_caption in steps:
                if step_caption is not None:
                    caption = step_caption
                self.screen.send("hand", {"fingers": fingers, "caption": caption})
                await asyncio.to_thread(self.controller.send_fingers, fingers)
                await asyncio.sleep(1)

    async def _shutdown_hand(self):
        """Cancel active gestures, close the hand and release the serial port."""
        if self.hand_shut_down:
            return

        tasks = tuple(self.gesture_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        async with self.gesture_lock:
            self.screen.send("hand", {"fingers": [], "caption": ""})
            await asyncio.to_thread(self.controller.send_fingers, [])
            
        # Give the physical servos 1.5 seconds to finish moving before 
        # the program exits and the Arduino loses its serial connection.
        await asyncio.sleep(1.5)

        self.controller.close()
        self.hand_shut_down = True

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
            
            # A missing "fingers" means a closed hand (the API may strip an
            # empty list).
            fingers = step.get("fingers", [])
            if not isinstance(fingers, list):
                raise ValueError("fingers must be a list")

            clean_fingers = set()
            for finger in fingers:
                # JSON numbers may arrive as floats (1.0); bools are not fingers.
                if isinstance(finger, bool) or not isinstance(finger, (int, float)):
                    raise ValueError(f"invalid finger: {finger!r}")
                if finger != int(finger) or not 0 <= finger <= 4:
                    raise ValueError(f"finger must be a whole number 0-4, got {finger!r}")
                clean_fingers.add(int(finger))
            validated_steps.append((sorted(clean_fingers), Helper._clean_caption(step.get("caption"))))
            
        return validated_steps

    @staticmethod
    def _clean_caption(caption):
        """Return the caption as short text, or None when the step has none."""
        if caption is None or isinstance(caption, bool):
            return None
        if isinstance(caption, float) and caption == int(caption):
            caption = int(caption)  # the model may send the number 4 as 4.0
        caption = str(caption).strip()
        return caption[:MAX_CAPTION_LENGTH] or None

    @staticmethod
    def _gesture_tool():
        declaration = types.FunctionDeclaration(
            name="perform_hand_sequence",
            description=(
                "Move the robotic hand through one or more positions. "
                "Call it whenever a number is spoken, and to act out emotions. "
                "A new call cancels a sequence that is still running."
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
                                },
                                "caption": {
                                    "type": "STRING",
                                    "description": "Optional short text shown big on the screen for this step, e.g. \"4\" or \"2 + 2 = ?\". Use it whenever the step shows a number.",
                                },
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
            description="Call this when the user says goodbye, wants to end the conversation, or wants to shut down the robotic hand. Say goodbye out loud before calling it.",
            parameters={"type": "OBJECT", "properties": {}}
        )
        return declaration

if __name__ == "__main__":
    asyncio.run(Helper().run())
