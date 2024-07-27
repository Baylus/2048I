from enum import IntEnum, Enum

############ NEAT STUFF
CACHE_CHECKPOINTS = True
NEAT_CONFIG_PATH = "config/neat_config.txt"

CHECKPOINTS_PATH = "checkpoints/"
GAMESTATES_PATH = "game_states/"

GENERATIONS = 1000
# Number of iterations that one model will train before training the other one.
TRAINING_INTERVAL = 5
CACHE_CHECKPOINTS = True
CHECKPOINT_INTERVAL = 10
RESTORE_CHECKPOINTS = True # Restore checkpoint override
CHECKPOINT_PREFIX = "neat-checkpoint-"

##### Gamestates #####
SAVE_GAMESTATES = False
BATCH_REMOVE_GENS = 5 # Number of generations to batch delete when appropriate
# Max size of gamestates folder, in GB
MAX_SIZE_OF_GAMESTATES = 25
# The interval at which we should save gamestate generations, incase we want to save
# some old ones, but not all the unnecessary. 
# e.g. interval = 5, gens 5, 10, 15, 20, etc. will be saved
ARCHIVE_GEN_INTERVAL = 10
###

##### PARALLELIZE #####
PARALLEL_OVERRIDE = False
HIDE_OVERRIDE = False

MAX_WORKERS = 1
### END

### Replays and Statistics
NUM_BEST_GENS_FOR_STATS = 5 # Number of best generations to show for stats
REPLAY_TPS = 10
### End replays and statistics

# Epsilon-greedy strategy settings
ENABLE_EPSILON = True
EPSILON_START = 0.8  # Initial exploration rate
EPSILON_END = 0.0    # Final exploration rate
# Started at 0.995
EPSILON_DECAY = 0.96  # Decay rate per generation

IS_HUMAN = False

##############

TPS = 1000 # Ticks per second
# Might have to change this to max number of consecutive no-ops
MAX_UPDATES_PER_GAME = 10000

WINDOW_WIDTH = 700
WINDOW_HEIGHT = 700

BOARD_WIDTH = 600
BOARD_HEIGHT = 600

# Pixel spacings between the tiles
TILE_GAP = 10


class DQNSettings:
    DQN_OVERRIDE = False
    EPISODES = 50
    MAX_TURNS = 1000
    PENALTY_FOR_NOOP = -25

    # Learning stuff
    LEARNING_RATE = 0.001

    ENABLE_EPSILON = True
    EPSILON_START = 0.9  # Initial exploration rate
    EPSILON_MIN = 0.01    # Final exploration rate
    # Started at 0.995
    EPSILON_DECAY = 0.99  # Decay rate per generation

    REPLAY_BATCH_SIZE = 64

    # Within the CHECK
    CHECKPOINTS_PATH = CHECKPOINTS_PATH + "dqn/"
    CHECKPOINT_INTERVAL = 10

class RayllibSettings:
    EPISODES = 10
    RAY_CONFIG = {
        "env": "custom_2048",
        "num_workers": 4,  # Number of parallel workers
        "framework": "tf",
        "gamma": 0.99,
        "lr": DQNSettings.LEARNING_RATE,
        "train_batch_size": DQNSettings.REPLAY_BATCH_SIZE,
        "exploration_config": {
            "type": "EpsilonGreedy",
            "initial_epsilon": DQNSettings.EPSILON_START,
            "final_epsilon": DQNSettings.EPSILON_MIN,
            "epsilon_timesteps": int(DQNSettings.EPSILON_DECAY * 1e6),  # Number of timesteps over which to anneal epsilon
        },
        "model": {
            "fcnet_hiddens": [256, 256],  # Same as your DQN model structure
            "fcnet_activation": "relu",
        },
        "target_network_update_freq": 500,
        "replay_buffer_config": {
            "capacity": 2000, # Size of replay buffer
        },
    }

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


DEBUG = False