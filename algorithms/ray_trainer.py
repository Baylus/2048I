from collections import deque
import numpy as np
import os
import random
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.dqn import DQN
from gym import spaces
import gym

# Disable TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras

# Your custom classes and methods
from action import Action, NETWORK_OUTPUT_MAP
from config.settings import BATCH_REMOVE_GENS, RayllibSettings, DQNSettings as dqns
from files.manage_files import prune_gamestates
from game import Board, GameDone, NoOpAction
from utilities.singleton import Singleton
from utilities.gamestates import DQNStates

class Custom2048Env(gym.Env):
    def __init__(self, config):
        super(Custom2048Env, self).__init__()
        self.board = Board()
        self.action_space = RayllibSettings.RAY_CONFIG["action_space"]
        self.observation_space = spaces.Box(low=0, high=1, shape=(4, 4, 1), dtype=np.float32)

    def reset(self):
        self.board = Board()
        return self._get_state()

    def _get_state(self):
        state = np.array(self.board.grid).reshape(4, 4, 1)
        return state

    def step(self, action: int):
        reward = 0
        done = False

        tmp = self.board.score
        try:
            # IMPORTANT: Due to the way we are defining action space, output will be single
            # integer, from 0-3. So we must use this mapping here.
            reward = self.board.act(NETWORK_OUTPUT_MAP[action])
        except GameDone:
            done = True
            reward = self.board.score - tmp
        except NoOpAction:
            reward = dqns.PENALTY_FOR_NOOP

        new_state = self._get_state()
        return new_state, reward, done, {}

    def render(self, mode='human'):
        pass

def env_creator(config):
    return Custom2048Env(config)
    
class RayTrainer:
    def __init__(self):
        pass
    
    def train(self):
        register_env("custom_2048", env_creator)

        ray.init(ignore_reinit_error=True)

        results = tune.run(
            DQN,
            config=RayllibSettings.RAY_CONFIG,
            stop={"episodes_total": RayllibSettings.EPISODES},  # Adjust based on how many episodes you want to run
            checkpoint_at_end=True
        )

        ray.shutdown()

        return results
