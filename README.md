# Robotic Hand Learning Platform

A voice tutor for young children that talks in Hebrew (Gemini Live) and shows
every number on a five-finger robotic hand and on a screen. It is built as a
**platform**: separate layers with small, documented interfaces, so a new
project can replace or extend one layer without touching the others.

```
 Child  ──voice──►  Python app  ◄──WebSocket──►  Gemini Live (cloud)
                    │        │
         screen events      serial "01111\n"
                    ▼        ▼
            Browser screen   Arduino Uno ──I²C + power──► Robotic hand
                                                         (Nano, servo driver,
                                                          5 servos, own supply)
```

| Layer | Interface | Change it to… |
|---|---|---|
| Hardware | Serial protocol (below) | drive the hand from any language, or plug in another hand |
| Hand control | `HandControl.send_fingers(fingers)` | use the hand from any Python program, no AI needed |
| AI and tools | Gemini Live function declarations | add tools, or swap the model |
| Content | `prompts/<name>.txt` | teach a different subject or play a game |
| Screen | WebSocket events | build another UI, a dashboard or a logger |

## Hardware

- **Arduino Uno** (outside the hand): connected to the computer by USB, runs
  [`handy_helper.ino`](handy_helper.ino). Upload it with the Arduino IDE (needs
  the *Adafruit PWM Servo Driver* library).
- **The hand**: an Arduino Nano and a servo driver at I²C address `0x40`, five
  servos, and its own power supply plugged into mains.
- **Data connector** between the Uno and the hand: `A4` (SDA), `A5` (SCL),
  `5V`, `GND`.
- Servo channels: `0` thumb, `1` index, `2` middle, `3` ring, `4` pinky.

## Setup and run

```bash
pip install -r requirements.txt
echo "MY_API_KEY=<your Gemini API key>" > .env
python3 handy_helper.py
```

The child's screen opens at `http://localhost:8080`. Stop with Ctrl+C or by
saying goodbye; the hand always closes on exit.

Optional settings (environment or `.env`):

| Variable | Default | Meaning |
|---|---|---|
| `HANDY_PROMPT` | `math_he` | which file in `prompts/` defines the tutor |
| `HAND_SERIAL_PORT` | `/dev/ttyACM0` | the Uno's serial port |
| `HANDY_UI_PORT` | `8080` | port of the screen server |
| `HANDY_OPEN_BROWSER` | `1` | set `0` to not open a browser tab |

## Layer interfaces

### 1. Serial protocol (Python ↔ Uno)

115200 baud, one line per message.

| Direction | Message | Meaning |
|---|---|---|
| Uno → PC | `HAND_READY` | sent once after reset, with all fingers closed |
| PC → Uno | `01111\n` | complete hand position, one character per finger from thumb to pinky: `1` up, `0` down |
| Uno → PC | `OK` | position applied |
| Uno → PC | `ERR` | malformed line (wrong length or character); nothing moved |

Because every command is the whole hand, neither side keeps hidden state, and
one good command always puts the hand right.

### 2. Hand control (Python)

```python
from hand_control import HandControl

hand = HandControl()          # opens the port, waits for HAND_READY
hand.send_fingers([1, 2])     # index and middle up, the rest down
hand.send_fingers([])         # closed fist
hand.close()
```

[`examples/count_to_five.py`](examples/count_to_five.py) counts on the hand
with no AI at all.

### 3. AI tools (Gemini Live)

The model gets two tools, declared in `handy_helper.py`:

- `perform_hand_sequence(steps)`: 1–10 steps, each
  `{"fingers": [0-4…], "caption": "optional text for the screen"}`. Validated
  in Python before anything moves; one second per step; a new call cancels the
  running one.
- `end_session()`: called after the tutor says goodbye.

**Adding a tool:** write a `types.FunctionDeclaration` next to `_gesture_tool()`,
add it to the `tools=[…]` list in `run()`, and handle `call.name` in
`_handle_tool_calls()`. Describe when to use it in the prompt.

### 4. Content (prompts)

A prompt is a plain text file in [`prompts/`](prompts). It sets the language,
the teaching style and how to use the hand. Two are included:

- `math_he`: the arithmetic tutor (default)
- `rps_he`: a game of rock, paper, scissors

```bash
HANDY_PROMPT=rps_he python3 handy_helper.py
```

### 5. Screen events (WebSocket)

The page at `/` connects to `ws://localhost:8080/ws` and receives JSON
messages `{"type": …, "value": …}`. A page that connects late first receives
the latest value of each type.

| `type` | `value` |
|---|---|
| `state` | `connecting`, `listening`, `thinking`, `speaking` or `bye` |
| `level` | microphone level, `0.0`–`1.0` (about 10 per second) |
| `you` | the child's words so far in this turn |
| `handy` | the tutor's words so far in this turn |
| `hand` | `{"fingers": [...], "caption": "..."}` for each gesture step |

## Ideas for follow-up projects

- **Answer with your fingers**: a camera recognises how many fingers the child
  shows, and the tutor checks the answer.
- **New subjects without code**: a new prompt file, e.g. counting in English or
  a quiz.
- **Feedback from the hand**: position or touch sensors through the Nano, so
  the system knows where each finger really is.
- **Offline**: replace Gemini with a local speech model behind the same tools.
- **Gesture library**: tools like wave, point or slow counting, and servo speed.
- **Progress tracking**: log the screen events into a report for a teacher or
  parent.
