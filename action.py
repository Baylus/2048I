"""Defines the actions that we can take on the board.


IMPORTANT:
    Because we have 3 different trainers, they have different ways of selecting outputs.
    Currently we have:
        NEAT - list: int[4], where the first 1 in the list corresponding with the priority order of actions
            determines which action to take
        DQNTrainer - random.sample(Action). As expected, just chooses one of the options

    This means we have to be careful to treat these methods of obtaining the actions
    to take.
"""

from enum import auto, IntEnum



# Only in its own file for circular dependency concerns.
class Action(IntEnum):
    UP = auto()
    RIGHT = auto()
    DOWN = auto()
    LEFT = auto()

NETWORK_OUTPUT_MAP = [
    Action.UP,
    Action.RIGHT,
    Action.DOWN,
    Action.LEFT,
]

def get_state_action(out_state: list):
    for i in range(len(NETWORK_OUTPUT_MAP)):
        if out_state[i]:
            return NETWORK_OUTPUT_MAP[i]
    return None