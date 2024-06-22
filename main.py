from enum import IntEnum, auto
import json
import neat
import numpy as np
import os
import pathlib
import pygame
import random
import shutil
import sys

from action import Action
from game import Board, GameDone

from config.settings import *

########## STARTUP CLEANUP

# DELETE GAME STATES #

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

def main(net=None) -> int:
    # Initial housekeeping
    global curr_pop
    global curr_gen
    global epsilon

    curr_pop += 1
    board.reset()

    clock = pygame.time.Clock()
    game_result = {
        "game_states": [],
        "fitness": 0
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
                if random.random() < epsilon:
                    # Choose a random action
                    # print("Picking random choice")
                    action = random.choice(list(Action))
                else:
                    # Choose the action suggested by the neural network
                    print("We are actually choosing this time")
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
        print("Finished game naturally")
        pass
    finally:
        file_name = f"{str(board.score)}_{str(curr_pop)}"
        file_name += ".json"
        print(type(game_result["game_states"][0]))
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


# if __name__ == "__main__":
#     main()


# To fix it from doing n-1 checkpoint numbers
class OneIndexedCheckpointer(neat.Checkpointer):
    def __init__(self, generation_interval=1, time_interval_seconds=None, filename_prefix="neat-checkpoint-"):
        super().__init__(generation_interval, time_interval_seconds, filename_prefix)

    def save_checkpoint(self, config, population, species_set, generation):
        # Increment the generation number by 1 to make it 1-indexed
        super().save_checkpoint(config, population, species_set, generation + 1)


# Define the fitness function
def eval_genomes(genomes, config_tarnished):
    global curr_gen
    global curr_pop
    global epsilon
    curr_pop = 0
    curr_gen += 1
    pathlib.Path(f"{GAMESTATES_PATH}/gen_{curr_gen}").mkdir(parents=True, exist_ok=True)

    epsilon_start = 1.0  # Initial exploration rate
    epsilon_end = 0.1    # Final exploration rate
    epsilon_decay = 0.995  # Decay rate per generation
    epsilon = max(EPSILON_END, EPSILON_START * (EPSILON_DECAY ** curr_gen))

    print(type(genomes))
    if type(genomes) == dict:
        genomes = list(genomes.items())

    # Initializing everything to 0 and not None
    for _, genome in genomes:
        genome.fitness = 0

    for (genome_id_player, genome) in genomes:
        # Create separate neural networks for player and enemy
        player_net = neat.nn.FeedForwardNetwork.create(genome, config_tarnished)
        
        # Run the simulation
        player_fitness = main(player_net)
        
        # Assign fitness to each genome
        genome.fitness = player_fitness

        assert genome.fitness is not None

if __name__ == "__main__":
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
    


pygame.quit()
sys.exit()