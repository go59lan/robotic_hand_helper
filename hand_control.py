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
        
        print("Arduino:", response)

    def send_fingers(self, fingers):

        toggle_list = self.fingers_to_toggle(fingers)
        command = ",".join(str(finger) for finger in toggle_list)

        print(command)
        self.arduino.write((command + "\n").encode())

        response = self.arduino.readline().decode().strip()

        print("Arduino:", response)

    def fingers_to_toggle(self, fingers):

        if not fingers:

            for i in range(5):
                self.open_flag[finger] = False
            return [5]

        toggle_list = []

        for finger in range(5):
            if  self.open_flag[finger] != (finger in fingers):
                toggle_list += [finger]
                self.open_flag[finger] = not self.open_flag[finger]

        return toggle_list


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