from argparse import ArgumentParser
from enum import IntEnum, auto
import json
import neat
import numpy as np
import os
import pathlib
import pygame
from queue import PriorityQueue
import random
import shutil
import sys

from action import Action
from game import Board, GameDone

from config.settings import *

parser = ArgumentParser()
parser.add_argument("-r", "--reset", dest="reset", action="store_true", default=False,
                    help="Reset training to not use previous checkpoints")
### Replays ###
# parser.add_argument("-p", "--replay", dest="replay", default=None,
#                     help="Replay a specific replay file")
# parser.add_argument("-b", "--best", dest="best", default=None, type=int,
#                     help="Number of best to show from the given/each generation")
# generations_help = """\
# Specify which generation to use. Used with best or replay to point to generation.
# Providing none, but specifying the argument gives all generations.
# Provide with only one number to get the last X generations of bests.
# Provide with 2 integers for a range of generations to process.
# If more than 2 generations are specified, then only those generations will be processed.
# """
# parser.add_argument("-g", "--generations", dest="gens", default=None, type=int, nargs='*',
#                     help=generations_help)
### Statistics ###
statistics_help = """\
Will we be printing out the statistics for the generations? Will show results from least to greatest generation number
Providing no integers will process all generations
Provide 1 int just to show the last X generations.
Provide 2 ints; number of generations to summarize and the other for interval of generations to display.
Prov
    e.g. -s 5 5. Will show statistics for 5 generations, at an interval of 5. with 100 generations, the \
\ \ generations statistics shown would be 80, 85, 90, 95, 100.
0 for the number of generations means all generations will be shown. 0 for interval is unacceptable.
"""
parser.add_argument("-s", "--stats", dest="stats", default=None, type=int, nargs='*',
                    help=statistics_help)
parser.add_argument("-q", "--quiet",
                    action="store_true", dest="quiet", default=False,
                    help="don't print status messages to stdout. Unused")
parser.add_argument("-dc", "--dont_clean", dest="clean", action="store_false", default=True,
                    help="Should we avoid cleaning up our previous gamestates?")

args = parser.parse_args()

# Check if args are valid
if args.stats:
    # We have stats, valid if we have 2 or no arguments
    l = len(args.stats)
    if l == 2:
        if args.stats[1] == 0:
            raise ValueError("You cannot give an interval of 0.")
        elif args.stats[0] < 0:
            raise ValueError("You cannot give a total generations less than 0.")
    elif l == 0:
        # This is okay. We are just processing all generations
        pass
    elif l == 1:
        # We are showing the last X generations
        if args.stats[0] < 0:
            raise ValueError("You provided a negative value for displaying the last X genertaions")
    else:
        raise ValueError(f"You didn't give the proper number of values for statistics. What you gave {args.stats}")


# replays = True if any([args.replay, args.best, args.gens != None]) else False


########## STARTUP CLEANUP

# DELETE GAME STATES #
# Only delete if we arent replaying, checking stats, and havent specified to not clean
# if not replays and args.clean and not SAVE_GAMESTATES:
if args.clean and not args.stats and not SAVE_GAMESTATES:
    folder = GAMESTATES_PATH
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    # Delete debug file to ensure we arent looking at old exceptions
    pathlib.Path.unlink("debug.txt", missing_ok=True)


##################


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


if WINDOW_HEIGHT < BOARD_HEIGHT or WINDOW_WIDTH < BOARD_WIDTH:
    raise ValueError("Board can't be bigger than the window")

if WINDOW_HEIGHT != WINDOW_WIDTH:
    print("WARNING: Just so you know your window ratio isnt 1:1")
if BOARD_HEIGHT != BOARD_WIDTH:
    raise ValueError("ERROR: Your board ratio isnt 1:1, which is definitely disgusting")

# Initialize Pygame
pygame.init()

# Set up the display
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("2048I")

board = Board()

# Globals modified and used all over
curr_pop = 0
curr_gen = 0
epsilon = 0

neat_config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            NEAT_CONFIG_PATH)

# Create the population
pop = neat.Population(neat_config)

def draw_text(surface, text, x, y, font_size=20, color=(255, 255, 255)):
    font = pygame.font.SysFont(None, font_size)
    text_surface = font.render(text, True, color)
    surface.blit(text_surface, (x, y))

def draw():
    # Fill background
    screen.fill(BACKGROUND_COLOR)

    board.draw(screen)

    draw_text(screen, "Generation: " + str(curr_gen), 100, 650, font_size=40, color=(255, 0, 0))
    draw_text(screen, "Population: " + str(curr_pop), 400, 650, font_size=40, color=(255, 0, 0))
    pygame.display.update()

def play_game(net=None) -> int:
    # Initial housekeeping
    global curr_pop

    curr_pop += 1
    board.reset()

    clock = pygame.time.Clock()
    game_result = {
        "fitness": 0,
        "game_states": [],
    }
    
    # Main game loop
    running = True
    updates = 0
    try:
        while running:
            clock.tick(TPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            if IS_HUMAN:
                keys = pygame.key.get_pressed()
                action = get_action(keys)
            else:
                if ENABLE_EPSILON and random.random() < epsilon:
                    # Choose a random action
                    # print("Picking random choice")
                    action = random.choice(list(Action))
                else:
                    # Choose the action suggested by the neural network
                    # print("We are actually choosing this time")
                    action = get_net_action(net, board.get_state())
            
            if action:
                board.act(action)

            draw()
            game_result["game_states"].append(list(board.get_state()))
            game_result["fitness"] = board.score
            if (board.is_done()):
                raise GameDone
            updates += 1
            if updates > MAX_UPDATES_PER_GAME:
                # game_result["notes"] = "Game stalemated"
                print("Something went really wrong here, the network wasnt outputting somehow.")
                running = False
    except GameDone:
        # Expected state
        # print("Finished game naturally")
        pass
    finally:
        file_name = f"{str(board.score)}_{str(curr_pop)}"
        file_name += ".json"
        with open(f"{GAMESTATES_PATH}/gen_{curr_gen}/{file_name}", 'w') as f:
            json.dump(game_result, f, cls=NpEncoder, indent=4)
    
    return int(game_result["fitness"])

NETWORK_OUTPUT_MAP = [
    Action.UP,
    Action.RIGHT,
    Action.DOWN,
    Action.LEFT,
]

def get_net_action(net, inputs) -> Action:
    # Now get the recommended outputs
    outputs = net.activate(inputs)
    for i in range(len(outputs)):
        if outputs[i]:
            return NETWORK_OUTPUT_MAP[i]
    return None

last_action = None

def get_action(inputs) -> Action:
    """Determine action to do

    Args:
        inputs (_type_): keys being pressed
    """
    global last_action

    action = None
    if inputs[pygame.K_w]: # W
        action = Action.UP
    elif inputs[pygame.K_s]: # S
        action = Action.DOWN
    elif inputs[pygame.K_a]: # A
        action = Action.LEFT
    elif inputs[pygame.K_d]: # D
        action = Action.RIGHT
    
    if action:
        print(f"This is our action {action}")
    # This makes it so that you can't just hold down a key and it will continue to spam the inputs
    # Only useful for user interactions, as the network will be able to keep up. Shocked it runs so fast tho
    if action and action != last_action:
        # These are different, and we found an action, so we can actually use it
        last_action = action
        return action

    last_action = action
    return None


# To fix it from doing n-1 checkpoint numbers
class OneIndexedCheckpointer(neat.Checkpointer):
    def __init__(self, generation_interval=1, time_interval_seconds=None, filename_prefix="neat-checkpoint-"):
        super().__init__(generation_interval, time_interval_seconds, filename_prefix)

    def save_checkpoint(self, config, population, species_set, generation):
        # Increment the generation number by 1 to make it 1-indexed
        super().save_checkpoint(config, population, species_set, generation + 1)


def eval_genomes(genomes, config_tarnished):
    global curr_gen
    global curr_pop
    global epsilon
    curr_pop = 0
    curr_gen += 1
    pathlib.Path(f"{GAMESTATES_PATH}/gen_{curr_gen}").mkdir(parents=True, exist_ok=True)

    epsilon = max(EPSILON_END, EPSILON_START * (EPSILON_DECAY ** curr_gen))

    if type(genomes) == dict:
        genomes = list(genomes.items())

    # Initializing everything to 0 and not None
    for _, genome in genomes:
        genome.fitness = 0

    for (genome_id_player, genome) in genomes:
        # Create separate neural networks for player and enemy
        player_net = neat.nn.FeedForwardNetwork.create(genome, config_tarnished)
        
        # Run the simulation
        player_fitness = play_game(player_net)
        
        # Assign fitness to each genome
        genome.fitness = player_fitness

        assert genome.fitness is not None

### Core processing functions ###
def main():
    # Add reporters, including a Checkpointer
    if CACHE_CHECKPOINTS:
        # Setup checkpoints
        curr_fitness_checkpoints = f"{CHECKPOINTS_PATH}"
        pathlib.Path(curr_fitness_checkpoints).mkdir(parents=True, exist_ok=True)
        # Find the run that we need to use
        runs = os.listdir(curr_fitness_checkpoints)
        i = 1
        for i in range(1, 10):
            if f"run_{i}" not in runs:
                break
        else:
            # If this happens then I have been running too many runs and I need to think of changing the fitnesss function
            raise Exception("Youve been trying this fitness function too many times. Fix the problem.")
        
        this_runs_checkpoints = f"{curr_fitness_checkpoints}/run_{i}"
        pathlib.Path(this_runs_checkpoints).mkdir(parents=True, exist_ok=True)

        pop.add_reporter(neat.StdOutReporter(True))

        checkpointer = OneIndexedCheckpointer(generation_interval=CHECKPOINT_INTERVAL, filename_prefix=f'{this_runs_checkpoints}/neat-checkpoint-')
        
        pop.add_reporter(checkpointer)
    
    try:
        winner = pop.run(lambda genomes, config: eval_genomes(genomes, config), n=GENERATIONS)
    except Exception as e:
        with open("debug.txt", "w") as f:
            f.write(str(e))
        raise
    pass

def process_statistics():
    # TODO: This
    # TODO: Adapt this
    # Figure out which generations that we need to process.
    existing_gens = os.listdir(GAMESTATES_PATH)
    gen_nums: list[int] = [int(name[4:]) for name in existing_gens]
    gen_nums.sort()
    
    gens_needed: list[int] = []
    if args.stats:
        # We have the stats arg specified. Find which ones.
        arg_len = len(args.stats)
        if arg_len == 0:
            gens_needed = gen_nums # None specified, so all generations
        elif arg_len == 1:
            # We need to get the last X generations
            gens_needed = gen_nums[-args.stats[0]:]
        elif arg_len == 2:
            # (X, Y) : Get X generations at interval Y
            gens, interval = args.stats
            # (gens - 1) because we are including the first end at the start.
            # + 1 because we need to have one more to include that first gen number
            max_gens_away = ((gens - 1) * interval) + 1
            NEG_OFFSET = 1
            gens_needed = gen_nums[-NEG_OFFSET:-(max_gens_away + NEG_OFFSET):-interval]
            if len(gens_needed) < gens:
                # We don't have enough generations. Put the last one on if we arent duplicating
                if gen_nums[0] not in gens_needed:
                    # Add the first one so we can get one more in to try to make up for how few we have
                    gens_needed.append(gen_nums[0])
        else:
            # These will need to intersect their input parameter lists and the 
            # list of available generations to get their answers
            gens_requested = args.stats
            
            gens_needed = list(set(gen_nums).intersection(gens_requested))
        
            if not gens_needed:
                raise ValueError(
                    "We could not find an intersection between the generations available " +
                    f"and the ones requested: {gen_nums} : {gens_requested}"
                )
    else:
        raise ValueError("We must not have specified generations... This should be addressed before here.")
    
    if not gens_needed:
        # Somehow we came up with no generations we could work
        raise ValueError("We didn't find any generations that we could work according to the input parameters")

    gens_needed.sort()

    for gen in gens_needed:
        display_stats_from_gen(gen)

# def process_replays():
#     """Process all replays that are requested
#     """
#     # TODO: Adapt this
#     # Figure out which generations that we need to process.
#     existing_gens = os.listdir(GAMESTATES_PATH)
#     gen_nums = [int(name[4:]) for name in existing_gens]
#     gen_nums.sort()
    
#     gens_needed = []
#     if args.gens and (arg_len := len(args.gens)) > 0:
#         # We have the gens arg specified. Find which ones.
#         if arg_len == 0:
#             gens_needed = gen_nums # None specified, so all generations
#         elif arg_len == 1:
#             # We need to get the last X generations
#             gens_needed = gen_nums[-args.gens[0]:]
#         else:
#             # These will need to intersect their input parameter lists and the list of available generations to get their answers
#             gens_requested = []
#             if arg_len == 2:
#                 # We need to get a range of generations
#                 gens_requested = list(range(args.gens[0], args.gens[1] + 1, 1))
#             elif arg_len > 2:
#                 # We just need to include the generations that are listed
#                 gens_requested = args.gens
            
#             gens_needed = list(set(gen_nums).intersection(gens_requested))
        
#             if not gens_needed:
#                 raise ValueError(f"We could not find an intersection between the generations available and the ones requested: {gen_nums} : {gens_requested}")
#     else:
#         gens_needed = gen_nums # None specified, so all generations
    
#     if not gens_needed:
#         # Somehow we came up with no generations we could work
#         raise ValueError("We didn't find any generations that we could work according to the input parameters")

#     # determine which trainers that we need to acquire bests from
#     ents = [Entities.TARNISHED, Entities.MARGIT]
#     if args.trainer:
#         # We are specifying one
#         if args.trainer == trainer_str(Entities.TARNISHED).lower():
#             # We are only training Tarnished
#             ents = [Entities.TARNISHED]
#         else:
#             ents = [Entities.MARGIT]
#     # Make the names readable
#     ents = [trainer_str(s) for s in ents]
#     gens_needed.sort()

#     # Replay best segments from trainer(s)
#     for trainer in ents:
#         # CONSIDER: Which is more important to go forwards or backwards in generations
#         # Going to go backwards and see what I like/dont
#         for gen in reversed(gens_needed):
#             replay_best_in_gen(gen, trainer, args.best or DEFAULT_NUM_BEST_GENS)

### End - Core processing functions ###
### Statistics ###
def display_stats_from_gen(gen_num):
    # Get most fit populations from gen
    gen_path = f"{GAMESTATES_PATH}/gen_{gen_num}/"
    pop_files = os.listdir(gen_path)
    pop_files.sort(key=lambda x: int(x[: x.find("_")])) # Sort by fitness value, best is last

    def get_file_details(file_name: str) -> tuple[int, int]:
        """Gets the fitness and population number from a given file.

        Args:
            file_name (str): _description_

        Returns:
            tuple[int, int]: _description_
        """
        index_split = file_name.find("_") # All files are organized by "{fitness}_{population num}"
        # Extension offset because all file names have ".json" on the end
        EXTENSION_OFFSET = -5
        fitness, pop_num = (file_name[ : index_split], file_name[index_split + 1 : EXTENSION_OFFSET])
        return int(fitness), int(pop_num)

    # find average, best, worst fitness and best tile and longest games. Pair the gen numbers with each stat
    # (val, pop_num)
    best_fitness: tuple[int, int] = get_file_details(pop_files[-1]) # Last file should have best fitness
    worst_fitness: tuple[int, int] = get_file_details(pop_files[0]) # Likewise, first is the worst

    sum_fitness = 0
    best_tile: tuple[int, int] = (0, 0)
    longest_game: tuple[int, int] = (0, 0)
    # Going in reverse will save a lot of overwriting, and could lead to skipping checks
    # altogether, but I am too lazy to implement that.
    for file in reversed(pop_files):
        with open(gen_path + file) as json_file:
            game_data = json.load(json_file)
        
        fitness, pop = get_file_details(file)
        sum_fitness += fitness
        last_frame = game_data["game_states"][-1]
        largest_tile = max(last_frame)
        if largest_tile > best_tile[0]:
            # We found a new larger tile
            best_tile = (largest_tile, pop)
        length = len(game_data["game_states"])
        if length > longest_game[0]:
            longest_game = (length, pop)
        
    average_fit = sum_fitness // len(pop_files)
    best_pops = pop_files[-NUM_BEST_GENS_FOR_STATS:]

    print(f"------------------------ Generation {gen_num} ------------------------")
    gen_stats = ""
    gen_stats += f"\tBest Fitness: {best_fitness[0]} (pop {best_fitness[1]})\n"
    gen_stats += f"\tWorst Fitness: {worst_fitness[0]} (pop {worst_fitness[1]})\n"
    gen_stats += f"\tAverage Fitness: {average_fit}\n"
    gen_stats += f"\tBest Tile: {best_tile[0]} (pop {best_tile[1]})\n"
    gen_stats += f"\tLongest Game: {longest_game[0]} (pop {longest_game[1]})"
    print(gen_stats)

    print("### Best Populations: ")
    # print(best_pops)
    for pop_file in best_pops:
        # Print populations statistics
        with open(gen_path + pop_file) as json_file:
            game_data = json.load(json_file)
        fitness, pop = get_file_details(pop_file)
        stats = "\t"
        stats += f"{pop}: "
        stats += f"Fitness: {fitness}. "
        last_frame = game_data["game_states"][-1]
        largest_tile = max(last_frame)
        stats += f"Best tile: {largest_tile}. "
        length = len(game_data["game_states"])
        stats += f"Game Length: {length}"
        print(stats)
    
    print("") # Just put a newline after this generation
### End - Statistics

### Replays ###
# def draw_replay(game_data):
#     """Specific draw function for replays
#     """
#     # TODO: Adapt this
#     WIN.blit(BG, (0,0))

#     tarnished.draw(WIN)
#     margit.draw(WIN)

#     trainer = curr_trainer or game_data["trainer"]
#     X = 160
#     curr_y_offset = 200
#     if trainer == trainer_str(Entities.TARNISHED):
#         draw_text(WIN, "Tarnished Fitness: " + str(game_data[f"{trainer_str(Entities.TARNISHED)}_fitness"]), X, curr_y_offset, font_size=40, color=(255, 0, 0))
#     else:
#         draw_text(WIN, "Margit Fitness: " + str(game_data[f"{trainer_str(Entities.MARGIT)}_fitness"]), X, curr_y_offset, font_size=40, color=(255, 0, 0))
    
#     # Generation meta stats for best replays
#     curr_y_offset += 25
#     if gen_best:
#         draw_text(WIN, "Best (Generation): " + str(gen_best), X, curr_y_offset, font_size=30, color=(255, 0, 0))
#         curr_y_offset += 25
#     if gen_average:
#         draw_text(WIN, "Avg. (Generation): " + str(gen_average), X, curr_y_offset, font_size=30, color=(255, 0, 0))
#         curr_y_offset += 25
#     curr_y_offset += 25


#     draw_text(WIN, "Generation: " + str(curr_gen or game_data["generation"]), X, curr_y_offset, font_size=30, color=(255, 0, 0))
#     curr_y_offset += 50
#     draw_text(WIN, "Population: " + str(curr_pop or game_data["population"]), X, curr_y_offset, font_size=30, color=(255, 0, 0))
#     curr_y_offset += 100

#     draw_text(WIN, "Fitness Details: ", X, curr_y_offset, font_size=40, color=(255, 0, 0))
#     for detail, val in game_data[f"{trainer_str(Entities.MARGIT)}_fitness_details"].items():
#         curr_y_offset += 25
#         draw_text(WIN, f"   {detail}: " + str(int(val)), X, curr_y_offset, font_size=30, color=(255, 0, 0))

#     pygame.display.update()


# def replay_game(game_data: dict):
#     global tarnished
#     global margit
#     # TODO: Adapt this
#     # Reset the npcs
#     tarnished = Tarnished()
#     margit = Margit()
#     tarnished.give_target(margit)
#     margit.give_target(tarnished)
    
#     time.sleep(0.2) # Give me time to stop pressing space before next game
#     # Main game loop
#     running = True
#     clock = pygame.time.Clock()
#     for frame in game_data["game_states"]:
#         clock.tick(REPLAY_TPS)
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False
#                 break
#         if not running:
#             break

#         keys = pygame.key.get_pressed()
#         if keys[pygame.K_SPACE]:
#             # Skip this game now.
#             break

#         tarn = frame["tarnished"]["state"]
#         marg = frame["margit"]["state"]

#         # Update Tarnished
#         tarnished.set_state(tarn)
#         margit.set_state(marg)
        
#         # Draw what has been updated
#         draw_replay(game_data)


# def replay_file(replay_file: str):
#     # TODO: Adapt this
#     global curr_trainer
#     curr_trainer = None # We are replaying without intention, so dont highlight anyone
#     # Get our game data
#     with open(replay_file) as json_file:
#         game_data = json.load(json_file)
    
#     replay_game(game_data)


# def replay_best_in_gen(gen: int, trainer: str, num_best = 0):
#     """Replay the best within a specific generation

#     We should separate out the games where the trainer is the active one being trained.
#     As it doesn't matter what the network did on the other party's training data, as it doesnt
#     affect the future generations.

#     Args:
#         gen (int): Which generation to display
#         num_best (int): How many of the top performers to show. If 0, show all of generation (not recommended)
#         trainer (str): Which trainer we are interested in.
#                        Very specific we need the same name as is written out to file names
#     """
#     # TODO: Adapt this
#     global curr_gen
#     global curr_pop
#     global curr_trainer
#     global gen_average
#     global gen_best

#     # Setup
#     curr_gen = gen
#     curr_trainer = trainer

#     gen_dir = f"{GAMESTATES_PATH}/gen_{gen}/"
#     runs = os.listdir(gen_dir)
#     gen_runs = [r for r in runs if trainer in r]

#     # Start collecting info on runs of generation
#     fitness_sum = 0
#     # (game's fitness for trainer, file name of game data)
#     runs_processed: list[tuple[int, str]] = []
#     for run_file in gen_runs:
#         # Get run data
#         with open(f"{gen_dir}{run_file}") as json_file:
#             game_data = json.load(json_file)
        
#         # Process data and keep a record of it for retrieval
#         this_fit = int(game_data[f"{trainer}_fitness"])
#         fitness_sum += this_fit
#         runs_processed.append((this_fit, run_file))
    
#     if not runs_processed:
#         raise ValueError(f"No runs to process for trainer {trainer}")
#     # Now we have the entire list of fitness scores for the generation
#     gen_average = fitness_sum / len(runs_processed) # Log this generations average for display

#     # Get ready to review runs
#     runs_processed.sort()
#     gen_best = runs_processed[-1][0] # Get the best fitness
#     # Pretty proud of this. If num best is 0, then we process whole list, because we splice whole list. ([0:])
#     # We put it in range because we are going to be popping elements off the back for efficiency, because we are starting with the best
#     # then working our way down.
#     for _ in range(len(runs_processed[-num_best:])):
#         # Get the next run to replay
#         fit, file = runs_processed.pop()
#         with open(f"{gen_dir}{file}") as json_file:
#             game_data = json.load(json_file)
#         curr_pop = game_data["population"]

#         # Now replay the game
#         replay_game(game_data)

### End - Replays ###


if __name__ == "__main__":
    if args.stats:
        process_statistics()
    # Adapt the replays for this to work
    # elif replays:
    #     process_replays()
    else:
        main()


pygame.quit()
sys.exit()