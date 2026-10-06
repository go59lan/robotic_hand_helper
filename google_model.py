from google import genai
import json

import os
from dotenv import load_dotenv

class Model:
  def __init__(self):

    my_api_key = os.getenv("MY_API_KEY")
    self.client = genai.Client(api_key=my_api_key)

    self.instructions = """
You are an AI learning assistant controlling a robotic hand.

The hand has five fingers:
0 = thumb
1 = index
2 = middle
3 = ring
4 = pinky

You answer the user's educational questions.

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

limit amount of steps to 10.
"""

    self.last_interaction = self.client.interactions.create(
      model="gemini-3.8-flash",
      input=self.instructions
)


# print(interaction.output_text)

  def query(self, question: str):

    self.last_interaction = self.client.interactions.create(
    model="gemini-3.8-flash",
    input=question,
    previous_interaction_id=self.last_interaction.id
    )


    print(self.last_interaction.output_text)
    return json.loads(self.last_interaction.output_text)
    


    # print(interaction.output_text)