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
import numpy as np
import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# import keras with tensorflow's warning message disabled
import keras
import shelve

from action import Action, NETWORK_OUTPUT_MAP
from config.settings import BATCH_REMOVE_GENS, DQNSettings as dqns
from files.manage_files import prune_gamestates, get_dqn_checkpoint_file, clean_temporary_checkpoints
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
    # We cannot store native object inside the pickle object.
    # Having this here, because on linux we cannot store the object natively.
    can_store_as_object: bool = False

    def __init__(self, len=2000, mem_file_name = "memory.pickle", reset = False):
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



class DQNTrainer():
    def __init__(self, checkpoint_file = "", reset = False):
        if reset:
            clean_temporary_checkpoints()
        self.model = None
        self.target_model = None
        self.board = Board()

        reset = self.init_models(checkpoint_file, reset)
        # It is possible that we are now supposed to reset our checkpoints.
        if reset:
            clean_temporary_checkpoints()
        # Training models
        self.callbacks = [] # i.e. checkpointers
        filename = "best" + dqns.WEIGHTS_SUFFIX
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

    def init_models(self, checkpoint_file = "", reset = False) -> bool:
        """Does a bunch of fancy stuff to resume from a previous checkpoint. Model or weights.

        I didn't need to make this whole thing, but its easier to change whether I am saving models
        or weights if I do it like this. Also, I can do custom stuff like resuming a model file
        if it is only X episodes old, or settle for the weights file that is newer.

        Args:
            checkpoint_file (str, optional): _description_. Defaults to "".
            reset (bool, optional): _description_. Defaults to False.

        Returns:
            bool: Whether we need to reset or not.
        """
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

        def resume_weights():
            if not self.model:
                self.model = build_model(state_size, action_size)
            else:
                # We shouldnt be trying to build a new model if we are resuming. Something is wrong
                raise Exception("We are making another model after having done so. Something must be wrong.")
            self.resume_episode = 1 # Episode that we should start back on
            checkpoint_path = ""
            if checkpoint_file and os.path.exists(check_path := dqns.CHECKPOINTS_PATH + checkpoint_file):
                # Try to find specified checkpoint file
                checkpoint_path = check_path
            else:
                # Try to the load previous weights.
                cfile, self.resume_episode = get_dqn_checkpoint_file(dqns.CHECKPOINTS_PATH)
                checkpoint_path = dqns.CHECKPOINTS_PATH + cfile
            
            if checkpoint_path:
                print(f"We found a checkpoint file to resume: {checkpoint_path}")
                if not os.path.exists(checkpoint_path):
                    print("What the... the checkpoint doesn't exist")
                self.model.load_weights(checkpoint_path)
                print("We loaded checkpoint properly")
            else:
                print("We don't have our specified weights. We are now resetting")
                global reset
                reset = True
            
        # Initialize parameters
        state_size = (4,)  # 4x4 grid flattened
        action_size = 4  # up, down, left, right

        self.model = None
        self.resume_episode = 1
        if not reset:
            # Determine if we can resume from a checkpointed model or need to use weights
            # Determine if the file provided is valid
            if checkpoint_file and os.path.exists(check_path := dqns.CHECKPOINTS_PATH + checkpoint_file):
                # We have been told to resume this checkpoint. figure out whether model or weights
                print(f"We are resuming from this checkpoint file: {checkpoint_file}")
                if dqns.WEIGHTS_SUFFIX in checkpoint_file:
                    # Just toss it to the weights resumer. It should handle all the messy stuff.
                    print("its a weight checkpoint")
                    resume_weights()
                elif checkpoint_file[-3:] == dqns.MODEL_SUFFIX:
                    # We have checked already its not a weight file, so it has to be a model
                    print("its a model checkpoint")
                    self.model = keras.models.load_model(check_path)
                else:
                    print(f"We were given a faulty checkpoint file name that doesn't exist: {checkpoint_file}\nAttempting to resume normally")
            
            if not self.model:
                # We now have to choose between loading a model or weights.
                model_file, mep = get_dqn_checkpoint_file(checkpoint_suffix=dqns.MODEL_SUFFIX)
                _, wep = get_dqn_checkpoint_file(checkpoint_suffix=dqns.WEIGHTS_SUFFIX)
                if mep and mep + dqns.STALER_MODEL_RESUME < wep:
                    # Our model file is too stale to resume from it, resume from weights
                    print(f"Our model checkpoint file is too old to resume {mep} < {wep}")
                    # Again, I am just tossing it to the entire resume weights function. It's slower
                    # and does a bit of redundant calculations, but I know it workds and I am not gonna touch it
                    resume_weights()
                elif model_file:
                    # We have a fresh enough model file for it to be worth resuming from.
                    print("We are resuming from our model checkpoint")
                    self.model = keras.models.load_model(dqns.CHECKPOINTS_PATH + model_file)
                    self.resume_episode = mep
                elif wep:
                    # So, we have gotten here because our weights file is less than 10 and theres no model.
                    resume_weights()

        if not self.model:
            print("We couldn't find a checkpoint file to resume from.")
            # We have to start an entirely new model
            self.model = build_model(state_size, action_size)
            # Since we failed to find a checkpoint file, then we shouldn't resume our replay memory
            reset = True
        
        self.target_model = build_model(state_size, action_size)
        self.target_model.set_weights(self.model.get_weights())
        
        return reset

    def reset(self):
        # Make sure board is fresh
        self.board = Board()

    def train(self, episodes: int = dqns.EPISODES, max_time: int = dqns.MAX_TURNS):
        try:
            for episode in range(self.resume_episode, episodes + 1):
                print(f"Training episode {episode}")
                try:
                    # Reset the trainer
                    self.reset()
                    game_states = DQNStates(episode)
                    # TODO: Enable viewing somehow when display is not disabled.
                    i = 0
                    while True:
                        if max_time and i >= max_time:
                            break
                        i += 1
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
                            # print("Doing replay training now")
                            minibatch = self.replay_buffer.get_samples(self.batch_size)
                            # state, action, reward, new state, done?
                            for s, a, r, ns, d in minibatch:
                                target = r
                                if not d:
                                    target = r + self.gamma * np.amax(self.target_model.predict(ns, verbose=0)[0])
                                target_f = self.model.predict(s, verbose=0)
                                # a - 1: Because our action value begins at 1, we need to map it back to arrays
                                target_f[0][a - 1] = target
                                self.model.fit(s, target_f, epochs=1, verbose=0, callbacks=self.callbacks)
                            
                            if self.epsilon > self.epsilon_min:
                                self.epsilon *= self.epsilon_decay

                    if i >= max_time:
                        game_states.add_notes("We stalled our game out too long.")
                finally: # After training one episode
                    game_states.log_game()
                    # House keeping
                    if episode % dqns.CHECKPOINT_INTERVAL == 0:
                        self.save_weights(episode)
                        self.save_model(episode)
                    # Save off our replay buffer
                    self.replay_buffer.save()

                    # See if we need to clean up our gamestates
                    # We should only need to prune the same interval that we batch delete, since they should
                    # All roughly be the same size.
                    # Actually, we are going to do it one less, because we want to be able to catch up in case the
                    # games are going longer due to fitter populations learning to survive.
                    if (episode % (BATCH_REMOVE_GENS - 1)) == 0:
                        prune_gamestates(dqn=True)
        finally: # After trying whole training loop
            # Make sure to close our replay buffer to ensure it works properly.
            self.replay_buffer.close()
            # Try to save off our weights, regardless of if its on the right interval.
            self.save_weights(episode)
            self.save_model(episode)
        # End .train()

    def save_weights(self, episode: int = 0):
        self.model.save_weights(dqns.CHECKPOINTS_PATH + f"{episode}{dqns.WEIGHTS_SUFFIX}")
    def save_model(self, episode: int = 0):
        self.model.save(dqns.CHECKPOINTS_PATH + f"{episode}{dqns.MODEL_SUFFIX}")
    
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
