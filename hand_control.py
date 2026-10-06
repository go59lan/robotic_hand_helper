import board
import busio
from adafruit_servokit import ServoKit
import serial
import time



class HandControl:

    def __init__(self):
        self.arduino = serial.Serial(
            '/dev/ttyACM0',
            115200,
            timeout=1
        )
        time.sleep(2)
        response = self.arduino.readline().decode().strip()
        
        print("Arduino:", response)

    def send_fingers(self, fingers):

        command = ",".join(str(finger) for finger in fingers)

        print(command)
        self.arduino.write((command + "\n").encode())

        response = self.arduino.readline().decode().strip()

        print("Arduino:", response)


# helper = HandControl()
# helper.send_fingers([0,1,2])
# time.sleep(2)
# helper.send_fingers([3,4])
# time.sleep(2)
# helper.send_fingers([2,1,0])
# time.sleep(2)
# helper.send_fingers([4,3])
# time.sleep(2)
# helper.send_fingers([4,0])
# time.sleep(2)
# helper.send_fingers([4,0])