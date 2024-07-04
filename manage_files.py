"""Manage the files and their sizes to ensure we are not taking up the entirety of my drives.

We only need a select amount of generations of game states to get a good picture 
of the progress that the model is making in training.
"""
import os
from os.path import join, getsize, isfile, isdir, splitext
import shutil

from config.settings import GAMESTATES_PATH, BATCH_REMOVE_GENS, MAX_SIZE_OF_GAMESTATES, ARCHIVE_GEN_INTERVAL


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

# print(float(GetFolderSize("C:\\")) /1024 /1024 /1024)


def delete_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    pass

def clean_gamestates():
    # We know for a fact that we have to remove some generations.
    # Get all generations
    existing_gens = os.listdir(GAMESTATES_PATH)
    gen_nums = [int(name[4:]) for name in existing_gens]
    gen_nums.sort()

    removed = 0
    for num in gen_nums[:]:
        # Check if we should keep this one archived
        if ARCHIVE_GEN_INTERVAL and (num % ARCHIVE_GEN_INTERVAL) == 0:
            # We should keep this one around.
            continue
        # delete this generation
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
    
    # We did not remove enough. 
    for i in range(BATCH_REMOVE_GENS - removed):
        # Remove the oldest generations first
        delete_folder(f"{GAMESTATES_PATH}/gen_{gen_nums[i]}")

    pass

def prune_gamestates():
    """Will check if the gamestates folder is too full, then clean it up if it is

    Maybe try to be cheeky and return a guess for how many generations to wait before checking again.
    """
    size = GetFolderSize(GAMESTATES_PATH)
    if (size > MAX_SIZE_OF_GAMESTATES):
        clean_gamestates()
    pass
