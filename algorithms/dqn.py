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
import keras
import numpy as np
import random
import tensorflow as tf

from action import Action
from config.settings import DQNSettings as dqs
from game import Board, GameDone, NoOpAction
from utilities.singleton import Singleton

class MemoryBuffer():
    def __init__(self, len=2000):
        self.buffer = deque(maxlen=len)

    def store(self, memory):
        self.buffer.append(memory)

class SharedMemory(MemoryBuffer, meta_class=Singleton):
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
    def __init__(self, checkpoint_file = ""):
        self.model = None
        self.target_model = None
        if checkpoint_file:
            # TODO: Implement checkpoint resuming
            raise NotImplementedError
        else:
            # Define the neural network model
            def build_model(input_shape, action_size):
                model = keras.Sequential([
                    keras.layers.Input(shape=input_shape),
                    keras.layers.Dense(256, activation='relu'),
                    keras.layers.Dense(256, activation='relu'),
                    keras.layers.Dense(action_size, activation='linear')
                ])
                model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')
                return model

            # Initialize parameters
            state_size = (16,)  # 4x4 grid flattened
            action_size = 4  # up, down, left, right
            self.model = build_model(state_size, action_size)
            self.target_model = build_model(state_size, action_size)
            self.target_model.set_weights(self.model.get_weights())

        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = dqs.EPSILON_START
        self.epsilon_min = dqs.EPSILON_MIN
        self.epsilon_decay = dqs.EPSILON_DECAY
        self.batch_size = dqs.REPLAY_BATCH_SIZE
        self.replay_buffer = ReplayMemory(2000)

        self.board = Board()

    def reset(self):
        self.epsilon = dqs.EPSILON_START
        # Make sure board is fresh
        self.board = Board()
        self.state = (16,)

    def train(self, max_time: int):
        # Reset the trainer
        self.reset()
        # TODO: Enable viewing somehow when display is not disabled.
        
        for _ in range(dqs.MAX_TURNS):  # Arbitrary max time steps per episode
            action = self._choose_action()
            next_state, reward, done = self._take_action(action)  # Execute action
            next_state = np.reshape(next_state, (4, 4)) # TODO: Confirm that this is correct/needed
            self.replay_buffer.store((self.state, action, reward, next_state, done))
            self.state = next_state
            
            if done:
                self.target_model.set_weights(self.model.get_weights())
                break
            
            if len(self.replay_buffer) > self.batch_size:
                minibatch = random.sample(self.replay_buffer, self.batch_size)
                # state, action, reward, new state, done?
                for s, a, r, ns, d in minibatch:
                    target = r
                    if not d:
                        target = r + self.gamma * np.amax(self.target_model.predict(ns)[0])
                    target_f = self.model.predict(s)
                    target_f[0][a] = target
                    self.model.fit(s, target_f, epochs=1, verbose=0)
                
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
        pass

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
            reward = self.board - tmp
        except NoOpAction:
            # Our action did not do anything. We need to punish the model for this.
            reward = dqs.PENALTY_FOR_NOOP
        
        # We are always going to update the state anyway
        new_state = self.board.get_state()
        return (new_state, reward, done)

    # Function to choose action based on epsilon-greedy policy
    def _choose_action(self):
        if np.random.rand() <= self.epsilon:
            return random.choice(list(Action))
        q_values = self.model.predict(self.board.get_state())
        return np.argmax(q_values[0])



# Training loop (simplified)
def train_dqn(episodes):

    for e in range(episodes):
        state = reset_game()  # Reset game to initial state
        state = np.reshape(state, [1, state_size[0]])
        
        for time in range(500):  # Arbitrary max time steps per episode
            action = choose_action(state, epsilon)
            next_state, reward, done = step_game(action)  # Execute action
            next_state = np.reshape(next_state, [1, state_size[0]])
            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state

            if done:
                target_model.set_weights(model.get_weights())
                break
            
            if len(replay_buffer) > batch_size:
                minibatch = random.sample(replay_buffer, batch_size)
                # state, action, reward, new state, done?
                for s, a, r, ns, d in minibatch:
                    target = r
                    if not d:
                        target = r + gamma * np.amax(target_model.predict(ns)[0])
                    target_f = model.predict(s)
                    target_f[0][a] = target
                    model.fit(s, target_f, epochs=1, verbose=0)
                
                if epsilon > epsilon_min:
                    epsilon *= epsilon_decay

# Start training
train_dqn(1000)