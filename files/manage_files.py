"""Manage the files and their sizes to ensure we are not taking up the entirety of my drives.

We only need a select amount of generations of game states to get a good picture 
of the progress that the model is making in training.
"""
import os
from os.path import join, getsize, isfile, isdir, splitext
import pathlib
import neat

from config.settings import *

def p(s):
    """Debug Print
    """
    if DEBUG:
        print(s)

#### MANAGE GAME STATES ####

def GetFolderSize(path):
    # Not super important to be quick, as this will only happen once per generation.
    # https://stackoverflow.com/questions/2485719/very-quickly-getting-total-size-of-folder
    TotalSize = 0
    for item in os.walk(path):
        for file in item[2]:
            try:
                TotalSize = TotalSize + getsize(join(item[0], file))
            except:
                print("error with file:  " + join(item[0], file))
    return TotalSize


def delete_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                delete_folder(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    pathlib.Path.rmdir(folder)


def clean_gamestates():
    """Cleans out gamestates.

    Will delete the oldest $BATCH_REMOVE_GENS from gamestates, perserving every
    $ARCHIVE_GEN_INTERVAL.

    WARNING: However, if we are still unable to remove the desired amount of gamestates cleared,
    then we will start deleting the "archived" generations from oldest to youngest.

    Example:
        batch remove = 10, archive = 5
        for generations 1-20, the following will be what remains
            5, 10, 13, 14, 15, ...
        Because we removed 10 generations, but saved every 5th one.
    """
    p("Cleaning gamestates")
    # We know for a fact that we have to remove some generations.
    # Get all generations
    existing_gens = os.listdir(GAMESTATES_PATH)
    gen_nums = [int(name[4:]) for name in existing_gens]
    p(f"We have {len(gen_nums)} available gen nums before")
    gen_nums.sort()

    removed = 0
    # Delete $BATCH_REMOVE_GENS, while perserving archived generations
    for num in gen_nums[:]:
        # Check if we should keep this one archived
        if ARCHIVE_GEN_INTERVAL and (num % ARCHIVE_GEN_INTERVAL) == 0:
            # We should keep this one around.
            continue
        # delete this generation
        p(f"Deleting generation {num}")
        delete_folder(f"{GAMESTATES_PATH}/gen_{num}")
        # Housekeeping
        # Remove from original list incase we need to 
        # start removing protected generations
        gen_nums.remove(num)
        removed += 1
        if removed >= BATCH_REMOVE_GENS:
            # We removed enough.
            break

    # Now, we either finished removing all we needed, or we need to start deleting protected generations
    if removed >= BATCH_REMOVE_GENS:
        # We removed enough.
        return
    else:
        p("We couldn't remove enough generations, we are getting rid of protected generations now")
    
    # We did not remove enough. Start removing the protected generations
    for i in range(BATCH_REMOVE_GENS - removed):
        # Remove the oldest generations first
        delete_folder(f"{GAMESTATES_PATH}/gen_{gen_nums[i]}")

    pass

def prune_gamestates():
    """Will check if the gamestates folder is too full, then clean it up if it is

    Maybe try to be cheeky and return a guess for how many generations to wait before checking again.
    """
    size = GetFolderSize(GAMESTATES_PATH)
    if (size > MAX_SIZE_OF_GAMESTATES * (1024 * 1024 * 1024)):
        clean_gamestates()

#### END MANAGE GAME STATES ####

def get_dqn_checkpoint_file(dir: str, ignore_best: bool = False) -> tuple[str, int]:
    """Gets the best checkpoint filename to use for the dqn training

    Will choose the "best.weights.h5 or the most recent episode (so highest number), in that order.

    Args:
        dir (str): Directory of the DQN checkpoints
        ignore_best (bool): If we want to ignore the best file for any reason.

    Returns:
        str, int: Name of checkpoint file, or empty string if there is none and number of the episode
    """
    BEST_WEIGHTS = "best.weights.h5"
    files = os.listdir(dir)
    if not ignore_best and BEST_WEIGHTS in files:
        return BEST_WEIGHTS, 0
    weight_files = [f for f in files if f[-11:] == ".weights.h5"]
    valid_weight_nums = []
    for file in weight_files:
        try:
            valid_weight_nums.append(int(file[:-11]))
        except ValueError:
            # This means its one of our protected or archived weight files that we don't want to use
            pass
    
    # I think we should be already sorted because of the file sorting, but just in case sort it.
    valid_weight_nums.sort()
    if valid_weight_nums:
        newest = valid_weight_nums[-1]
    else:
        # There were no valid weight files
        return "", 0
    return str(newest) + ".weights.h5", newest

def get_newest_checkpoint_file(files: list[str], prefix: str) -> tuple[str, int]:
    """Gets the most recent checkpoint from the previous run the resume the training.

    Args:
        files (list[str]): _description_
        prefix (str): _description_

    Returns:
        tuple[str, int]: <file name, generation number>
    """
    def get_gen_num_from_name(file_name: str) -> int:
        if file_name[-1] == '-':
            raise ValueError(f"There is something really wrong. This checkpoint file is missing a gen number: {file_name}")
        max_gen_num_len = len(str(GENERATIONS))
        postfix = file_name[ -max_gen_num_len :]
        for i in range(len(postfix)):
            if postfix[i] == '-':
                # We found the dash, the rest is the gen number
                return int(postfix[i+1:])
        else:
            # We had no '-', so this whole thing must be the gen number
            return int(postfix)
    
    file_details = ["", 0]
    prefixed = [fn for fn in files if prefix in fn] # Files containing the prefix
    for name in prefixed:
        gen = get_gen_num_from_name(name)
        if gen > file_details[1]:
            file_details = (name, gen)

    return file_details



# To fix it from doing n-1 checkpoint numbers
class OneIndexedCheckpointer(neat.Checkpointer):
    def __init__(self, generation_interval=1, time_interval_seconds=None, filename_prefix="neat-checkpoint-"):
        super().__init__(generation_interval, time_interval_seconds, filename_prefix)

    def save_checkpoint(self, config, population, species_set, generation):
        # Increment the generation number by 1 to make it 1-indexed
        super().save_checkpoint(config, population, species_set, generation + 1)

def get_pop_and_gen(cmd_args) -> tuple[int, int]:
    """Get the population and checkpoint for our new run.

    Basically should just be everything in the main file before the actual run
    Args:
        cmd_args (_type_): _description_
    
    Returns:
        (int, int): (population to use, generation we are starting on)
    """
    # Create the population
    neat_config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                NEAT_CONFIG_PATH)
    pop = neat.Population(neat_config)
    start_gen_num = 0
    # Add reporters, including a Checkpointer
    if CACHE_CHECKPOINTS:
        # Setup checkpoints
        curr_fitness_checkpoints = f"{CHECKPOINTS_PATH}"
        pathlib.Path(curr_fitness_checkpoints).mkdir(parents=True, exist_ok=True)
        # Find the run that we need to use
        runs = os.listdir(curr_fitness_checkpoints)
        run_val = 1
        for i in range(1, 25):
            if f"run_{i}" not in runs:
                if not RESTORE_CHECKPOINTS or cmd_args.reset:
                    # We are not restoring from checkpoints, so we need to make a new directory, which would be the i'th run dir
                    run_val = i
                break
            # Store this int in case we need to restore to a previous checkpoint
            run_val = i
        else:
            raise Exception("Try clearing empty run directories or archiving some")
        
        this_runs_checkpoints = f"{curr_fitness_checkpoints}/run_{run_val}"
        print(f"We found our run folder is {run_val}")
        pathlib.Path(this_runs_checkpoints).mkdir(parents=True, exist_ok=True)
        start_gen_num = 0
        checkpointer = None
        if RESTORE_CHECKPOINTS and not cmd_args.reset:
            # We gotta find the right run to restore
            existing_checkpoint_files = os.listdir(this_runs_checkpoints)
            print(f"This is our existing checkpoints from {this_runs_checkpoints}:\n{existing_checkpoint_files}")
            if existing_checkpoint_files:
                # Since we have checkpoints, we need to actually initialize the population with them.
                checkpoint, start_gen_num = get_newest_checkpoint_file(existing_checkpoint_files, CHECKPOINT_PREFIX)
                pop = neat.Checkpointer.restore_checkpoint(f"{this_runs_checkpoints}/{checkpoint}")
                checkpointer = neat.Checkpointer(generation_interval=CHECKPOINT_INTERVAL, filename_prefix=f'{this_runs_checkpoints}/{CHECKPOINT_PREFIX}')
                print(f"We are using {checkpoint}")
        
        if not checkpointer:
            # If we are not resuming previous checkpoint, create it one indexed so we don't get weird numbers
            checkpointer = OneIndexedCheckpointer(generation_interval=CHECKPOINT_INTERVAL, filename_prefix=f'{this_runs_checkpoints}/neat-checkpoint-')
        
        # TODO: Add this to allow us to record to a output file too
        # https://stackoverflow.com/a/14906787
        pop.add_reporter(checkpointer)
    
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.StatisticsReporter())

    return pop, start_gen_num