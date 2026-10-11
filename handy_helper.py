import serial
import time
import speech_recognition as sr
from bidi.algorithm import get_display
from gtts import gTTS
import os
from playsound3 import playsound
import asyncio

from google_model import Model
from hand_control import HandControl


class Helper:

    def __init__(self):
        
        self.model = Model()
        print("model")
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
            
            await asyncio.gather(
                self.speak(response["response"]),
                self.gesture_sequence(response["steps"])
            )

            time.sleep(2)
            self.controller.send_fingers([5])



    async def gesture_sequence(self, steps):

        for step in steps:
            self.controller.send_fingers(step["fingers"])
            await asyncio.sleep(1)

    async def speak(self, text):
        await asyncio.to_thread(self._speak_blocking, text)


    def _speak_blocking(self, text):
        tts = gTTS(text=text, lang='iw', slow=False)

        output_file = "welcome.mp3"
        tts.save(output_file)

        playsound(output_file)
        os.remove(output_file)


            



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