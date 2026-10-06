import serial
import time
import speech_recognition as sr
from bidi.algorithm import get_display
from gtts import gTTS
import os
from playsound3 import playsound

from google_model import Model



class Helper:

    def __init__(self):

        # self.arduino = serial.Serial(
        #     '/dev/ttyACM0',
        #     115200,
        #     timeout=1
        # )

        self.model = Model()

        self.mic = None
        self.speaker = None

    def listen(self):

        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("talk:")
            audio = r.listen(source)
            text = r.recognize_google(audio, language="iw-IL")
            print(get_display(text))

            response = self.model.query(text)

            # Create the gTTS object
            tts = gTTS(text=response["response"], lang='iw', slow=False)

            # Save the script into an MP3 file
            output_file = "welcome.mp3"
            tts.save(output_file)

            # Play the saved audio file
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

helper.listen()