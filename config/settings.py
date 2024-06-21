from enum import IntEnum, Enum

WINDOW_WIDTH = 700
WINDOW_HEIGHT = 700

BOARD_WIDTH = 600
BOARD_HEIGHT = 600

# Pixel spacings between the tiles
TILE_GAP = 10

TPS = 30 # Ticks per second
GENERATIONS = 50

BACKGROUND_COLOR = (187, 173, 160)
BOARD_COLOR = (187, 173, 160)
# Grabbed these from some github, so hopefully they are sortof accurate
TILE_COLOR = {
    0: (204, 192, 179),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
    # From here its just extra stuff, gonna just pick green because it will be easier to see than a darker color like purple. 
    # Plus green is way better.
    4096:  (20,255,64),
    8192: (20,255,64),
    16384: (20,255,64),
    # If we get past here, we are better than any other AI that has come before it, and peoples
    # theoretial best performances on a 4x4 grid. So, if this raises an exception, then the 
    # AI has learned to cheat.
}

FONT_COLOR = (119, 110, 101)
FONT_SIZE = 75
