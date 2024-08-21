from argparse import ArgumentParser
import concurrent.futures
from contextlib import contextmanager
from enum import IntEnum, auto
import json
import logging
import neat
import numpy as np
import os
import pathlib
from queue import PriorityQueue
import random
import shutil
import signal
import sys
import traceback

# Setup import with env vars
os.environ['CUDA_VISIBLE_DEVICES '] = '0'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print("We got a GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("Sorry, no GPU for you...")

# Silences pygame welcome message
# https://stackoverflow.com/a/55769463
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from action import Action, get_state_action
from game import Board, GameDone, NoOpAction
from fitness import get_fitness

from algorithms.dqn import DQNTrainer
from config.settings import *
from files.manage_files import prune_gamestates, get_pop_and_gen
from utilities.gamestates import GameStates
from utilities.screen import SharedScreen

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
statistics_help = """
Will we be printing out the statistics for the generations? Will show results from least to greatest generation number
Providing no integers will process all generations
Provide 1 int just to show the last X generations.
Provide 2 ints; number of generations to summarize and the other for interval of generations to display.
Prov
    e.g. -s 5 5. Will show statistics for 5 generations, at an interval of 5. with 100 generations, the
    generations statistics shown would be 80, 85, 90, 95, 100.
0 for the number of generations means all generations will be shown. 0 for interval is unacceptable.
"""
parser.add_argument("-s", "--stats", dest="stats", default=None, type=int, nargs='*',
                    help=statistics_help)
parser.add_argument("-q", "--quiet",
                    action="store_true", dest="quiet", default=False,
                    help="don't print status messages to stdout. Unused")
parser.add_argument("-dc", "--dont_clean", dest="clean", action="store_false", default=True,
                    help="Should we avoid cleaning up our previous gamestates?")
parser.add_argument("-l", "--parallel", dest="parallel", action="store_true", default=False,
                    help="Should we run parallel instances of the game simulation?")
parser.add_argument("-i", "--hide", dest="hide", action="store_true", default=False,
                    help="Should we hide the game by not drawing the entities?")
parser.add_argument("-d", "--dqn", dest="dqn", action="store_true", default=False,
                    help="Should we train using DQN? Will take precedence over NEAT training.")

args = parser.parse_args()

args.parallel = PARALLEL_OVERRIDE or args.parallel
args.dqn = DQNSettings.DQN_OVERRIDE or args.dqn
# TODO: Once we have added screen visualization to DQN training, dont hide screen anymore
args.hide = HIDE_OVERRIDE or args.hide or args.parallel

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
def clean_gamestates(override = False):
    # Override enables forcing deletion of all things. Used in case things get really out of hand during automated training
    if override or (args.clean and not args.stats and args.reset and not SAVE_GAMESTATES):
        def _remove_directory(folder):
            print("Cleaning game states")
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))

        _remove_directory(GAMESTATES_PATH)
        # Make sure we are clearing our memory buffer if we are resetting.
        _remove_directory(DQNSettings.CHECKPOINTS_PATH + DQNSettings.MEMORY_SUBDIR)
    
    # Delete debug file to ensure we arent looking at old exceptions
    pathlib.Path.unlink(pathlib.Path("debug.txt"), missing_ok=True)
    pathlib.Path.unlink(pathlib.Path("debug.log"), missing_ok=True)

def setup_logger():
    logger = logging.getLogger('genome_logger')
    logger.setLevel(logging.DEBUG)
    
    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler('debug.log')
    
    # Set levels for handlers
    c_handler.setLevel(logging.WARNING)
    f_handler.setLevel(logging.INFO)
    
    # Create formatters and add them to handlers
    c_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)
    
    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    
    return logger


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
# pygame.init()
# We should be initializing pygame within screen here
shared_screen = SharedScreen()

if __name__ == "__main__":
    clean_gamestates()

# Logger enforces singleton, so should be safe to include as global
logger = setup_logger()

# Set up the display
if not args.hide:
    shared_screen.init_screen()

# Globals modified and used all over

def play_game(net, pop, gen, ep) -> int:
    try:
        # Initial housekeeping
        # logger.debug("Starting Board")
        board = Board()
        board.reset()
        logger.debug(f"Generation: {gen}, Population: {pop}")

        # logger.debug("before clock")
        clock = pygame.time.Clock()
        game_result = {
            "fitness": 0,
            "score": 0,
            "notes": "",
            "game_states": [],
        }
        logger.debug("About to run game")

        # Main game loop
        running = True
        updates = 0
        while running:
            clock.tick(TPS)
            if not args.hide:
                # We cannot get events if we are not displaying a window
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

            random_action: bool = False
            if IS_HUMAN:
                keys = pygame.key.get_pressed()
                action = get_action(keys)
            else:
                if ENABLE_EPSILON and random.random() < ep:
                    action = random.choice(list(Action))
                    random_action = True
                else:
                    # Choose the action suggested by the neural network
                    action = get_net_action(net, board.get_state())
            
            if action:
                try:
                    board.act(action)
                except NoOpAction:
                    # This means that the board didnt change as a result of the action taken.
                    # If this is a random action, we shouldn't penalize the model for nothing happening.
                    # So we will not record this as a frame as far as the game is concerned.
                    if random_action:
                        # Continue to next loop iteration, to avoid logging this frame and counting it against the model
                        continue
            
            if not args.hide:
                draw(board, pop)
            
            game_result["game_states"].append(list(board.get_state()))
            if (board.is_done()):
                raise GameDone
            updates += 1
            if updates > MAX_UPDATES_PER_GAME:
                # game_result["notes"] = "Game stalemated"
                logger.debug("Something went really wrong here, the network wasnt outputting somehow.")
                game_result["notes"] = "We ran out of time"
                running = False
    except GameDone:
        # Expected state
        pass
    except Exception:
        logger.debug("Something happened")
        with open("game_debug.txt", "w") as f:
        # with open("debug_game.txt", "w") as f:
            f.write(traceback.format_exc())
        raise
    finally:
        game_result["score"] = board.score
        fit = get_fitness(board.score, game_result["game_states"])
        game_result["fitness"] = fit
        file_name = f"{str(fit)}_{str(pop)}"
        file_name += ".json"
        logger.debug(f"We are trying to submit our gamestate for pop {pop} in gen {gen}")
        with open(f"{GAMESTATES_PATH}/gen_{gen}/{file_name}", 'w') as f:
            json.dump(game_result, f, cls=NpEncoder, indent=4)
    
    logger.debug(f"Returning result {int(game_result['fitness'])}")
    return int(game_result["fitness"])

def eval_genomes(genomes, config):
    gen = get_gen()
    pop_num = 0
    pathlib.Path(f"{GAMESTATES_PATH}/gen_{gen}").mkdir(parents=True, exist_ok=True)

    ep = epsilon(gen)
    if ENABLE_EPSILON:
        print(f"Our new epsilon for {gen} is {ep}")

    if type(genomes) == dict:
        genomes = list(genomes.items())

    # Initializing everything to 0 and not None
    for _, genome in genomes:
        genome.fitness = 0
    
    results: dict[int, int] = None
    if args.parallel:
        results = {}
        # Create a global flag for termination
        terminate_flag = False

        def handle_termination(signum, frame):
            global terminate_flag
            terminate_flag = True
            print("Termination signal received. Cleaning up...")

        # Register signal handlers
        signal.signal(signal.SIGINT, handle_termination)
        signal.signal(signal.SIGTERM, handle_termination)

        @contextmanager
        def terminating_executor(max_workers):
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                try:
                    yield executor
                finally:
                    if terminate_flag:
                        executor.shutdown(wait=True)
                        print("Executor shut down gracefully.")
                        sys.exit(0)
        
        # We need to execute the training in parallel
        # Create a process pool for parallel execution
        futures = {}
        with terminating_executor(max_workers=MAX_WORKERS) as executor:
            for (genome_id, genome) in genomes:
                pop_num += 1
                net = neat.nn.FeedForwardNetwork.create(genome, config)
                # Schedule the game simulation to run in parallel
                # We need to give it the pop_num, as parallel is gonna mess with it a lot
                future = executor.submit(play_game, net, int(pop_num), gen, ep)
                futures[future] = genome_id
                
            logger.debug("We made all the futures, now handle getting the results.")
            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    fit = future.result()
                    genome_id = futures[future]
                    results[genome_id] = fit
                except Exception as e:
                    logger.exception(f"Exception getting result for genome {genome_id}: {e}")
                    raise
                
                # It is inconceivable with the number of fitness modifications that tarnished has a fitness 0...
                # While it could be possible for us to reach 0 at some point, its extremely unlikely, as its only really
                # feasible at the start if the network continuously outputs one input. However, this is solved with us 
                # using epsilon training, as it gives us time in the beginning while it trains to avoid situations like 
                # this. So only raise an error if the epsilon training is active and it still hit 0, as that means we 
                # failed to assign it somehow.
                # IMPORTANT: This may change when we switch to Deep Q-Learning, as I do not know how that works, but we likely
                # wont be using epsilon training when we implement that.
                assert genome.fitness or ENABLE_EPSILON, "We failed to retrieve/assign the fitness, or we need to buy a lottery ticket"

    pop_num = 0
    for (genome_id, genome) in genomes:
        if results is not None:
            # We already did parallel execution, log results
            fitness = results[genome_id]
        else:
            # Create separate neural networks for player and enemy
            # NOTE: Not sure why its claiming that this code isn't reachable.
            # results is None if we do not execute parallel.
            logger.debug("We are executing this manually")
            player_net = neat.nn.FeedForwardNetwork.create(genome, config)
            
            # Run the simulation
            pop_num += 1
            fitness = play_game(player_net, pop_num, gen, ep)
        
        # Assign fitness to each genome
        genome.fitness = fitness

        # It is inconceivable with the number of fitness modifications that tarnished has a fitness 0...
        # See parallel for more info
        assert genome.fitness or ENABLE_EPSILON, "We failed to retrieve/assign the fitness, or we need to buy a lottery ticket"

    # See if we need to clean up our gamestates
    # We should only need to prune the same interval that we batch delete, since they should
    # All roughly be the same size.
    # Actually, we are going to do it one less, because we want to be able to catch up in case the
    # games are going longer due to fitter populations learning to survive.
    if (gen % (BATCH_REMOVE_GENS - 1)) == 0:
        prune_gamestates()


### Core processing functions ###
def main():
    pathlib.Path(f"{GAMESTATES_PATH}").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{CHECKPOINTS_PATH}").mkdir(parents=True, exist_ok=True)
    try:
        if args.dqn:
            dqns_dirs = DQNSettings.CHECKPOINTS_PATH + DQNSettings.MEMORY_SUBDIR
            pathlib.Path(f"{dqns_dirs}").mkdir(parents=True, exist_ok=True)
            # TODO: Add checkpoint resuming
            trainer = DQNTrainer(reset = args.reset, hide_screen = args.hide)
            import datetime as dt
            import time
            try:
                start = dt.datetime.now()
                pstart = time.process_time()
                trainer.train(DQNSettings.EPISODES)
            finally:
                end = dt.datetime.now()
                pend = time.process_time()
                duration = end - start
                print(f"Here is the standard time delta: {str(duration)}")
                print(f"Here is the processed time: {str(dt.timedelta(seconds=pend-pstart))}")
                with open("duration.txt", "w") as f:
                    f.write(str(duration))
            
        else:
            pop, start_gen_num = get_pop_and_gen(args)
            get_gen.current = start_gen_num

            _ = pop.run(eval_genomes, n=GENERATIONS - start_gen_num)
    except OSError as e:
        # This is likely because we ran out of memory.
        
        # # This would be ideal, but it would need testing to make sure it works, or it could cause serious issues since it will be left unattended, and I might not have time to test it before leaving, and I want to start a run before going.
        import errno
        if e.errno == errno.ENOSPC:
            # We did run out of memory.
            clean_gamestates(override=True)
        # clean_gamestates(override=True)
    except Exception:
        with open("debug.txt", "w") as f:
            f.write(traceback.format_exc())
        raise
    pass

def process_statistics():
    # TODO: Possibly move this to the files directory
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


def get_net_action(net, inputs) -> Action:
    # Now get the recommended outputs
    outputs = net.activate(inputs)
    return get_state_action(outputs)

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


def draw(board: Board, pop):
    # Fill background
    screen = shared_screen.get_screen()
    if not screen:
        return
    screen.fill(BACKGROUND_COLOR)

    board.draw(screen)

    shared_screen.draw_text("Generation: " + str(get_gen.current), 100, 650, font_size=40, color=(255, 0, 0))
    shared_screen.draw_text("Population: " + str(pop), 400, 650, font_size=40, color=(255, 0, 0))
    pygame.display.update()


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
            tuple[int, int]: (fitness, pop_num)
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
        file_name = gen_path + file
        game_data = GameStates.load_game_data(file_name)
        
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
        file_name = gen_path + pop_file
        game_data = GameStates.load_game_data(file_name)
        fitness, pop = get_file_details(pop_file)
        stats = "\t"
        stats += f"{pop}: "
        stats += f"Fitness: {fitness}. "
        last_frame = game_data["game_states"][-1]
        largest_tile = max(last_frame)
        stats += f"Best tile: {largest_tile}. "
        length = len(game_data["game_states"])
        stats += f"Game Length: {length}"
        lost_fit = int(game_data["score"]) - int(game_data["fitness"])
        stats += f"Fitness lost to failed moves: {lost_fit}"
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

def epsilon(gen = None):
    """Current epsilon value. Only sets if given a generation, to avoid repeated calculations

    Args:
        gen (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if not hasattr(epsilon, "current"):
        epsilon.current = 0  # it doesn't exist yet, so initialize it
    if gen:
        epsilon.current = max(EPSILON_END, EPSILON_START * (EPSILON_DECAY ** gen))
    return epsilon.current

def get_gen() -> int:
    """Really strange way of maintaining a global state for the current generation.
    Set using get_gen.current = X. Anytime retrieving will increment generation, so
    likely will need to use offset to get the right value.
    See this for details: https://stackoverflow.com/a/279597
    Returns:
        int: Current generation
    """
    if not hasattr(get_gen, "current"):
        get_gen.current = 0  # it doesn't exist yet, so initialize it
    get_gen.current += 1
    return get_gen.current

if __name__ == "__main__":
    if args.stats:
        process_statistics()
    # Adapt the replays for this to work
    # elif replays:
    #     process_replays()
    else:
        main()


pygame.quit()