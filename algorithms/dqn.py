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

from action import Action, NETWORK_OUTPUT_MAP
from config.settings import BATCH_REMOVE_GENS, DQNSettings as dqns
from files.manage_files import prune_gamestates
from game import Board, GameDone, NoOpAction
from utilities.singleton import Singleton
from utilities.gamestates import DQNStates

class MemoryBuffer(deque):
    def __init__(self, len=2000):
        super().__init__(maxlen=len)

    def store(self, memory):
        self.append(memory)

class SharedMemory(MemoryBuffer, metaclass=Singleton):
    """ 
        Shared replay memory for when we need to have parallel processes where 
        we will access both local and shared memory buffers.
    """
    pass

class ReplayMemory(MemoryBuffer):
    """Replay memory state to store the current replay buffer for local actor processing
    """
    pass


class DQNTrainer():
    def __init__(self, checkpoint_file = "best.weights.h5"):
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
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=dqns.LEARNING_RATE), loss='mse')
            return model

        # Initialize parameters
        state_size = (4,)  # 4x4 grid flattened
        action_size = 4  # up, down, left, right
        self.model = build_model(state_size, action_size)
        if checkpoint_file:
            if os.path.exists(check_path := dqns.CHECKPOINTS_PATH + checkpoint_file):
                # Load the previous weights.
                self.model.load_weights(check_path)
            else:
                print("We don't have our specified weights")
                # TODO: Consider looking for other weight files within the directory.

        self.target_model = build_model(state_size, action_size)
        self.target_model.set_weights(self.model.get_weights())
        filename = "best.weights.h5"
        self.callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=dqns.CHECKPOINTS_PATH + filename, 
                save_weights_only=True,
                save_freq=dqns.CHECKPOINT_INTERVAL,
            )        
        )

        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = dqns.EPSILON_START
        self.epsilon_min = dqns.EPSILON_MIN
        self.epsilon_decay = dqns.EPSILON_DECAY
        self.batch_size = dqns.REPLAY_BATCH_SIZE
        self.replay_buffer = ReplayMemory(2000)

        self.turn_limit = dqns.MAX_TURNS

    def reset(self):
        # Make sure board is fresh
        self.board = Board()

    def train(self, episodes: int = 1):
        for episode in range(episodes):
            print(f"On episode {episode + 1}")
            try:
                self._train(episode)
            finally:
                # House keeping
                if episode % dqns.CHECKPOINT_INTERVAL == 0:
                    self.save_weights(episode)
                # See if we need to clean up our gamestates
                # We should only need to prune the same interval that we batch delete, since they should
                # All roughly be the same size.
                # Actually, we are going to do it one less, because we want to be able to catch up in case the
                # games are going longer due to fitter populations learning to survive.
                if (episode % (BATCH_REMOVE_GENS - 1)) == 0:
                    prune_gamestates()
        # End .train()

    def _train(self, episode: int):
        """Train one episode

        Args:
            episode (int): Episode number
        """
        print(f"On episode {episode + 1}")
        try:
            # Reset the trainer
            self.reset()
            game_states = DQNStates(episode + 1)
            # TODO: Enable viewing somehow when display is not disabled.
            for i in range(self.turn_limit):  # Arbitrary max time steps per episode
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
                    minibatch = random.sample(self.replay_buffer, self.batch_size)
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

            if i == self.turn_limit:
                game_states.add_notes("We stalled our game out too long.")
        finally:
            game_states.log_game()
        pass

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
