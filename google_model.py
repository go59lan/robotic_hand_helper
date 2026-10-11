from google import genai
import json

import os
from dotenv import load_dotenv

class Model:
  def __init__(self):

    load_dotenv()
    my_api_key = os.getenv("MY_API_KEY")
    self.client = genai.Client(api_key=my_api_key)
    print("before model")
    self.instructions = """
You are an AI learning assistant controlling a robotic hand.

The hand has five fingers:
0 = thumb
1 = index
2 = middle
3 = ring
4 = pinky

You answer the user's educational questions. be thorough.

You may use the robotic hand to demonstrate the answer when
a physical demonstration is useful.

Never output servo angles.
Never output Python code.
Never directly control motors.

Instead, select fingers using numbers 0 through 4.

If no physical demonstration is useful, use action="none".

If there is an apropriate hand gesture or multiple (like for counting),
use action="sequence" and steps holds the list of fingers list to hold up.

replay only in json format, like if i ask what is two plus two:

{
  "response": "Two plus two equals four. Let's count two more after two.",
    "action": "sequence",
    "steps": [
      {
        "fingers": [0, 1]
      },
      {
        "fingers": [0, 1, 2]
      },
      {
        "fingers": [0, 1, 2, 3]
      }
    ]
}

limit amount of steps to 10. prefer to always use gestures,
like if you explain something raise the index finger or,
if you want to affirm the pupil show thumbs up, or "rock on dude".
you can show large numbers like 543 by doing a sequence of
[0,1,2,3,4], [1,2,3,4], [1,2,3]
"""
    self.last_interaction = None
    try:
      self.last_interaction = self.client.interactions.create(
        model="gemini-3.8-flash",
        input=self.instructions
      )
      print(self.last_interaction.output_text)
    except Exception as error:
      # Keep the helper usable: the next query can retry with a new interaction.
      print(f"Could not initialize the model interaction: {error}")



# print(interaction.output_text)

  def query(self, question: str):
    try:
      request = {
        "model": "gemini-3.8-flash",
        "input": question,
      }

      if self.last_interaction is None:
        request["system_instruction"] = self.instructions
      else:
        request["previous_interaction_id"] = self.last_interaction.id

      self.last_interaction = self.client.interactions.create(**request)
      return json.loads(self.last_interaction.output_text)
    except Exception as error:
      print(f"Model request failed: {error}")
      return self._error_response()

  @staticmethod
  def _error_response():
    return {
      "response": "מצטער, לא הצלחתי לקבל תשובה מהמודל כרגע. נסה שוב בעוד רגע.",
      "action": "none",
      "steps": [],
    }
    

    # print(interaction.output_text)
