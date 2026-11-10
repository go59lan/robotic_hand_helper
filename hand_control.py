import threading
import time

import serial


class HandControl:

    def __init__(self):

        # Serial access happens from worker threads (asyncio.to_thread), so
        # make sure only one command is on the wire at a time.
        self.lock = threading.Lock()

        self.arduino = serial.Serial(
            '/dev/ttyACM0',
            115200,
            timeout=1
        )
        time.sleep(2)
        response = self.arduino.readline().decode().strip()
        if response != "HAND_READY":
            print(f"WARNING: unexpected Arduino greeting: {response!r}")


    def send_fingers(self, fingers):

        # Send the complete hand position, e.g. [0, 2] -> "10100".
        # The Arduino sets every finger to match, so the two sides can never
        # drift out of sync the way they could with toggle commands.
        command = "".join("1" if finger in fingers else "0" for finger in range(5))

        with self.lock:
            self.arduino.write((command + "\n").encode())
            response = self.arduino.readline().decode().strip()

        if response != "OK":
            print(f"WARNING: Arduino did not confirm {command!r}: {response!r}")
