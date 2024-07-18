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

def get_state_action(out_state):
    for i in range(len(NETWORK_OUTPUT_MAP)):
        if out_state[i]:
            return NETWORK_OUTPUT_MAP[i]
    return None