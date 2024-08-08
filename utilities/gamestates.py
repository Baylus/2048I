"""Controller for game states which should make it easier to store 
"""
import json
import numpy as np


from fitness import get_fitness
from game import Board

from config.settings import DQNSettings, GAMESTATES_PATH


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class GameStates():
    game_state_dir = GAMESTATES_PATH

    def __init__(self, pop: int, file_name: str = ""):
        self.frames = []
        self.latest_board = None
        self.pop = pop
        self.notes = ""
        self.file_name = ""

    def store(self, board: Board):
        self.latest_board = board # Used for getting fitness later
        self.frames.append(board.get_state())
    
    def log_game(self):
        data = self._get_data()
        if self.file_name:
            path = self.file_name
        else:
            # Nope, we have to find it.
            path = self.game_state_dir + self._get_file_name()
        
        with open(path, 'w') as f:
            json.dump(data, f, cls=NpEncoder)

    def _get_data(self) -> dict:
        score = self.latest_board.score
        self.data = {
            "score": score,
            "fitness": get_fitness(score, self.frames),
            "notes": self.notes,
            "game_states": self.frames
        }
        return self.data
    
    def _get_file_name(self):
        raise NotImplementedError("This needs to be defined in sub classes")

    def add_notes(self, notes):
        self.notes = notes

    def load_game_data(file: str = "") -> dict:
        if not file.startswith(GAMESTATES_PATH):
            file = GAMESTATES_PATH + file

        with open(file, 'r') as f:
            data = json.load(file, f, cls=NpEncoder)
        
        return data


class DQNStates(GameStates):
    def __init__(self, episode: int):
        self.episode = episode
        super().__init__(pop = episode)

    def _get_file_name(self):
        name = ""
        if self.data:
            # Alright we can grab our fitness, default to score
            name += str(self.data.get("fitness", self.latest_board.score))
        else:
            name += str(self.latest_board.score)
        name += f"_{self.episode}.json"
        
        return name


class NEATStates(GameStates):
    def __init__(self, gen, pop):
        self.gen = gen
        super().__init__(pop)
        raise NotImplementedError("TODO: Finish the rest of this later.")