from enum import auto, IntEnum

# Only in its own file for circular dependency concerns.
class Action(IntEnum):
    UP = auto()
    RIGHT = auto()
    DOWN = auto()
    LEFT = auto()
