import board
import busio
from adafruit_servokit import ServoKit
import serial
import time



class HandControl:

    def __init__(self):

        self.open_flag = [False,False,False,False,False]

        self.arduino = serial.Serial(
            '/dev/ttyACM0',
            115200,
            timeout=1
        )
        time.sleep(2)
        response = self.arduino.readline().decode().strip()


    def send_fingers(self, fingers):
        
        toggle_list = self.fingers_to_toggle(fingers)
        
        # If no fingers need to move, skip sending anything to the Arduino
        if not toggle_list:
            return

        command = ",".join(str(finger) for finger in toggle_list)


        self.arduino.write((command + "\n").encode())

        response = self.arduino.readline().decode().strip()


    def fingers_to_toggle(self, fingers):
        
        toggle_list = []

        for finger in range(5):
            # If the current flag doesn't match the desired state, add it to the list
            if self.open_flag[finger] != (finger in fingers):
                toggle_list.append(finger)
                self.open_flag[finger] = not self.open_flag[finger]

        return toggle_list