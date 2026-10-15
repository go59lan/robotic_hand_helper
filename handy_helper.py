import serial
import speech_recognition as sr
from bidi.algorithm import get_display
from gtts import gTTS
import os
from playsound3 import playsound
import asyncio
import tempfile

from google_model import Model
from hand_control import HandControl


class Helper:

    def __init__(self):
        
        self.model = Model()
        self.controller = HandControl()
        self.mic = None
        self.speaker = None

    async def listen(self):

        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("talk:")
            audio = r.listen(source)
            text = r.recognize_google(audio, language="iw-IL")
            print(get_display(text))

            response = self.model.query(text)

            # Generate the MP3 before the gesture starts. gTTS is a network
            # call, so generating it inside speak() delayed audio playback.
            output_file = await self.prepare_speech(response["response"])
            try:
                await asyncio.gather(
                    self.speak(output_file),
                    self.gesture_sequence(response["steps"])
                )
            finally:
                if os.path.exists(output_file):
                    os.remove(output_file)

            await asyncio.sleep(2)
            self.controller.send_fingers([5])



    async def gesture_sequence(self, steps):

        for step in steps:
            # Serial reads are blocking, so do not pause audio playback while
            # waiting for the Arduino response.
            await asyncio.to_thread(self.controller.send_fingers, step["fingers"])
            await asyncio.sleep(1)

    async def prepare_speech(self, text):
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as file:
            output_file = file.name

        try:
            await asyncio.to_thread(self._create_speech_file, text, output_file)
            return output_file
        except Exception:
            os.remove(output_file)
            raise

    async def speak(self, output_file):
        await asyncio.to_thread(playsound, output_file)

    def _create_speech_file(self, text, output_file):
        tts = gTTS(text=text, lang='iw', slow=False)
        tts.save(output_file)


            



# r = sr.Recognizer()
# with sr.Microphone() as source:
#     print("start")
#     audio = r.listen(source)
#     text = r.recognize_google(audio, language="iw-IL")
#     print(get_display(text))

# print("end")

# Define the text and language
# text_to_speak = "שתיים ועוד שתיים זה ארבע"
# language = 'iw'

# # Create the gTTS object
# tts = gTTS(text=text_to_speak, lang='iw', slow=False)

# # Save the script into an MP3 file
# output_file = "welcome.mp3"
# tts.save(output_file)

# # Play the saved audio file
# playsound(output_file)


helper = Helper()

asyncio.run(helper.listen())
