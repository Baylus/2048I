"""Manage the files and their sizes to ensure we are not taking up the entirety of my drives.

We only need a select amount of generations of game states to get a good picture 
of the progress that the model is making in training.

Some options for removing gamestates or reducing size:

I'd be happy to help you with that!

It's great that you're training an AI model for your game, but I understand the concern about disk space limitations. Storing game states for replaying can indeed consume a significant amount of storage.

Here are a few potential solutions to help you overcome this issue:

**1. Lossless Compression: Implement a lossless compression algorithm to reduce the size of the stored game states. This way, you can store more data in the same amount of disk space. Some popular lossless compression libraries are:
    * Zlib (Python: zlib module)
    * LZO (Python: lzo module)
    * Snappy (Python: snappy module)
    * LZ4 (Python: lz4 module)

You can compress the game state data before storing it, and decompress it when you need to replay the game.

**2. Delta Encoding: Instead of storing the entire game state, store the differences (deltas) between consecutive game states. This can significantly reduce the amount of data stored. You can use a combination of delta encoding and compression to further optimize storage.

For example, you can store the initial game state and then only store the changes made in each subsequent state. When replaying, you can reconstruct the game state by applying the deltas to the initial state.

**3. Game State Summarization: Develop a summarization technique to extract the most important information from the game state, reducing the amount of data stored. This could involve:
    * Extracting key game metrics (e.g., score, health, position)
    * Storing a simplified representation of the game state (e.g., a condensed grid or graph)
    * Using dimensionality reduction techniques (e.g., PCA, t-SNE) to reduce the feature space

**4. Random Sampling: Instead of storing all game states, randomly sample a subset of states from each generation. This can help reduce the overall storage requirements while still maintaining a representative sample of the game's evolution.

**5. Cloud-based Storage: Consider using cloud-based storage services like AWS S3, Google Cloud Storage, or Microsoft Azure Blob Storage. These services offer scalable, cost-effective storage solutions that can grow with your needs.

**6. Generational Storage: Store only the game states for a limited number of generations, and then discard the oldest generations as new ones are generated. This approach can help manage storage growth while still allowing you to maintain a history of the game's evolution.

**7. Custom Data Structures: Design custom data structures to store game states in a more compact form. For example, you could use bit-packing, run-length encoding (RLE), or other techniques to reduce storage requirements.

Which of these solutions resonates with you, or do you have any other ideas you'd like to explore? I'm here to help you implement a solution that fits your specific needs!
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
    # We know for a fact that we have to remove some generations.
    # Get all generations
    existing_gens = os.listdir(GAMESTATES_PATH)
    gen_nums = [int(name[4:]) for name in existing_gens]
    gen_nums.sort()

    removed = 0
    # Delete $BATCH_REMOVE_GENS, while perserving archived generations
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
    if (size > MAX_SIZE_OF_GAMESTATES):
        clean_gamestates()
    pass
