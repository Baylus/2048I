"""Class to create and train DQN networks


There will be several important areas that should be taken into consideration when
designing this. Mainly around the parallelizing this solution. The main aspects that
need to be considered are:
    The DQN agents will need to have a parallel

    
From the paper: http://arxiv.org/pdf/1507.04296
It is claiming that the "Parameter Server" is the only one that is parallel, and
the Learner and Actor should be singular instances

"""
from collections import deque
import concurrent.futures
from contextlib import contextmanager
import numpy as np
import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# import keras with tensorflow's warning message disabled
import keras
import shelve
import signal
import sys
import threading

from action import Action, NETWORK_OUTPUT_MAP
from config.settings import BATCH_REMOVE_GENS, DQNSettings as dqns
from files.manage_files import prune_gamestates
from game import Board, GameDone, NoOpAction
from utilities.singleton import Singleton
from utilities.gamestates import DQNStates

class MemoryBuffer():
    buffer: deque

    memory_file: shelve
    # Determine if there have been any updates since the last time we made any updates
    # to our deque in memory that we would need to save back to the file.
    # This prevents unnecessary disk writes on potentially very large files, since we are
    # going to have up to 2000 records in our replay buffer.
    unsaved_changes: bool

    def __init__(self, len=2000, mem_file_name = "memory.pickle", reset = False):
        print(f"What is this file name {mem_file_name}")
        self.memory_file = shelve.open(dqns.CHECKPOINTS_PATH + dqns.MEMORY_SUBDIR + mem_file_name)
        if "buffer" in self.memory_file and not reset:
            # We had a previous buffer stored in this memory file, and we arent trying to reset our memory.
            self.buffer = self.memory_file["buffer"]
        else:
            # We did not find a previous memory file. Create a new one.
            print("We did not find a previous replay buffer")
            self.buffer = deque(maxlen=len)
        
        self.unsaved_changes = False

    def get_samples(self, samples):
        """Gets a specified number of samples from this replay buffer

        Args:
            samples (int): Number of samples to get

        Returns:
            list[sample]: List of samples chosen
        """
        return random.sample(self.buffer, samples)

    def store(self, memory):
        self.buffer.append(memory)
        self.unsaved_changes = True

    def __len__(self):
        return len(self.buffer)
    
    def save(self):
        self.memory_file["buffer"] = self.buffer
        self.unsaved_changes = False
    
    def close(self):
        if self.unsaved_changes:
            self.save() # Make sure we save off before closing.
        self.memory_file.close()

class SharedMemory(MemoryBuffer, metaclass=Singleton):
    """ 
        Shared replay memory for when we need to have parallel processes where 
        we will access both local and shared memory buffers.
    """
    pass

class ReplayMemory(MemoryBuffer):
    """Replay memory state to store the current replay buffer for local actor processing
    """
    def __init__(self, mem_file_name = "replay_memory.pickle", *args, **kwargs):
        super().__init__(*args, mem_file_name=mem_file_name, **kwargs)


# Create a global flag for termination
terminate_flag = False

def handle_termination(signum, frame):
    global terminate_flag
    terminate_flag = True
    print("Termination signal received. Cleaning up...")

# Only setup signal handlers if we are going to be training in parallel
if dqns.ENABLE_PARALLEL_REPLAY_TRAINING:
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

@contextmanager
def lock_if_needed(lock, is_needed = dqns.ENABLE_PARALLEL_REPLAY_TRAINING):
    if is_needed:
        with lock:
            yield
    else:
        yield

class DQNTrainer():
    def __init__(self, checkpoint_file = "best.weights.h5", reset = False):
        self.model = None
        self.target_model = None
        self.board = Board()
        
        # Training models
        self.callbacks = [] # i.e. checkpointers
        # Define the neural network model
        def build_model(input_shape, action_size):
            model = keras.Sequential([
                keras.layers.Input(shape=input_shape),
                keras.layers.Dense(256, activation='relu'),
                keras.layers.Dense(256, activation='relu'),
                keras.layers.Dense(action_size, activation='linear')
            ])
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=dqns.LAERNING_RATE), loss='mse')
            return model

        # Initialize parameters
        state_size = (4,)  # 4x4 grid flattened
        action_size = 4  # up, down, left, right
        self.model = build_model(state_size, action_size)
        if checkpoint_file and not reset:
            if os.path.exists(check_path := dqns.CHECKPOINTS_PATH + checkpoint_file):
                # Load the previous weights.
                self.model.load_weights(check_path)
            else:
                print("We don't have our specified weights")
                # TODO: Consider looking for other weight files within the directory.

        self.target_model = build_model(state_size, action_size)
        self.target_model.set_weights(self.model.get_weights())
        filename = "best.weights.h5"
        # TODO: Consider saving more than just the weights
        self.callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=dqns.CHECKPOINTS_PATH + filename, 
                save_weights_only=True,
                save_best_only=True
            )        
        )

        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = dqns.EPSILON_START
        self.epsilon_min = dqns.EPSILON_MIN
        self.epsilon_decay = dqns.EPSILON_DECAY
        self.batch_size = dqns.REPLAY_BATCH_SIZE
        self.replay_buffer = ReplayMemory(len=dqns.REPLAY_BUFFER_SIZE, reset=reset)
        print(f"After making our replay buffer, it has {len(self.replay_buffer)} elements")


    def reset(self):
        # Make sure board is fresh
        self.board = Board()

    def train(self, episodes: int = dqns.EPISODES, max_time: int = dqns.MAX_TURNS):
        try:
            for episode in range(1, episodes + 1):
                print(f"Training episode {episode}")
                try:
                    # Reset the trainer
                    self.reset()
                    game_states = DQNStates(episode)
                    # TODO: Enable viewing somehow when display is not disabled.
                    for i in range(max_time):  # Arbitrary max time steps per episode
                        game_states.store(self.board)
                        action = self._choose_action()
                        # print(f"Our action results are type {type(action)}, and heres its value {action}")
                        curr_state = self.board.grid
                        next_state, reward, done = self._take_action(action)  # Execute action
                        next_state = np.reshape(next_state, (4, 4)) # TODO: Confirm that this is correct/needed
                        self.replay_buffer.store((curr_state, action, reward, next_state, done))
                        
                        if done:
                            self.target_model.set_weights(self.model.get_weights())
                            break
                        
                        if len(self.replay_buffer) > self.batch_size:
                            self._replay_train()
                            
                            if self.epsilon > self.epsilon_min:
                                self.epsilon *= self.epsilon_decay

                    if i == max_time:
                        game_states.add_notes("We stalled our game out too long.")
                finally: # After training one episode
                    game_states.log_game()
                    # House keeping
                    if episode % dqns.CHECKPOINT_INTERVAL == 0:
                        self.save_weights(episode)
                    # Save off our replay buffer
                    self.replay_buffer.save()

                    # See if we need to clean up our gamestates
                    # We should only need to prune the same interval that we batch delete, since they should
                    # All roughly be the same size.
                    # Actually, we are going to do it one less, because we want to be able to catch up in case the
                    # games are going longer due to fitter populations learning to survive.
                    if (episode % (BATCH_REMOVE_GENS - 1)) == 0:
                        prune_gamestates()
        finally: # After trying whole training loop
            # Make sure to close our replay buffer to ensure it works properly.
            self.replay_buffer.close()
            # Try to save off our weights, regardless of if its on the right interval.
            self.save_weights(episode)
        # End .train()

    def _replay_train(self):
        # print("Doing replay training now")
        model_lock = threading.Lock() # Lock to prevent multi-access to self.model in parallel training
        def train_one(replay) -> None:
            # state, action, reward, new state, done?
            s, a, r, ns, d = replay
            target = r
            if not d:
                target = r + self.gamma * np.amax(self.target_model.predict(ns,verbose=0)[0])
            # Only lock if we need to because we are running parallel execution.
            # This might not save much, but could reduce errors if we use it always
            with lock_if_needed(model_lock):
                target_f = self.model.predict(s, verbose=0)
                # a - 1: Because our action value begins at 1, we need to map it back to arrays
                target_f[0][a - 1] = target
                # This is where I would put the locking mechanism to prevent more than one process
                # from writing back to the model.
                self.model.fit(s, target_f, epochs=1, verbose=0, callbacks=self.callbacks)
        
        minibatch = self.replay_buffer.get_samples(self.batch_size)
        if dqns.ENABLE_PARALLEL_REPLAY_TRAINING:
            with terminating_executor(max_workers=dqns.REPLAY_TRAIN_WORKERS) as executor:
                try:
                    for replay in minibatch:
                        executor.submit(train_one, replay)
                except KeyboardInterrupt:
                    executor.shutdown(wait=False)
                    raise
                # We don't need to wait individually since they don't return anything.
                # Just wait till they are all finished then shutdown
                executor.shutdown(wait=True)
        else:
            # We are just normally running our training
            for replay in minibatch:
                train_one(replay)
                

    def save_weights(self, episode: int = 0):
        self.model.save_weights(dqns.CHECKPOINTS_PATH + f"{episode}.weights.h5")

    def _take_action(self, action: Action) -> tuple[list[int], int, bool]:
        """Executes action on current board, and retrieves the values relevant for that action.

        Args:
            action (Action): Action to take

        Returns:
            list[int], new_state: The new state after taking action
            int: Reward from having taken this action
            bool: Whether we are finished with episode after taking action
        """
        reward = 0
        done = False

        tmp = self.board.score # In case our game ends, we need to find out the score that we earned in move anyway
        try:
            reward = self.board.act(action)
        except GameDone:
            # Our game finished from our action
            done = True
            reward = self.board.score - tmp
        except NoOpAction:
            # Our action did not do anything. We need to punish the model for this.
            # NOTE: As opposed to NEAT, or other genetic learning algorithms, we
            #       actually do want to punish the reward, even when it is a random 
            #       move from the epsilon-greedy strategy, because we are training 
            #       the Q-values based on the moves chosen, so if we did not punish 
            #       it here, it would affect how the model interprets this action, 
            #       and it might get the values confused as opposed to when it 
            #       chooses an action that is not valid.
            reward = dqns.PENALTY_FOR_NOOP
        
        # We are always going to update the state anyway
        new_state = self.board.get_state()
        return (new_state, reward, done)

    # Function to choose action based on epsilon-greedy policy
    def _choose_action(self) -> Action:
        if np.random.rand() <= self.epsilon:
            return random.choice(list(Action))
        # print(f"Our current grid {self.board.grid}\n")
        q_values = self.model.predict(self.board.grid, verbose=0)
        return NETWORK_OUTPUT_MAP[np.argmax(q_values[0])]
