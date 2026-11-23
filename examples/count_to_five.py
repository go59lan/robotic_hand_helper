"""Move the robotic hand without any AI: count from 1 to 5, then close.

Shows that the hardware layer stands on its own. Run from the project root:

    python3 examples/count_to_five.py

Set HAND_SERIAL_PORT if the Arduino is not on /dev/ttyACM0.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hand_control import HandControl  # noqa: E402

COUNT = [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [0, 1, 2, 3, 4]]

hand = HandControl()
try:
    for fingers in COUNT:
        print("fingers up:", fingers)
        hand.send_fingers(fingers)
        time.sleep(1)
finally:
    hand.send_fingers([])  # always finish with a closed hand
    time.sleep(1)
    hand.close()
