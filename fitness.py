import numpy as np


class FitnessSettings():
    NOOP_ACTION_PENALTY = 0.5

def boards_are_same(first, second) -> bool:
    if np.all(first == second):
        return True
    return False

def get_fitness(score, frames: list[list[int]]) -> int:
    s = FitnessSettings
    fitness = score

    # Penalize the model for each frame that it's action resulted in nothing changing
    # This also penalizes the model when no action is taken.
    for i in range(1, len(frames)):
        if boards_are_same(frames[i-1], frames[i]):
            # Model's action did nothing. Penalize it for this.
            fitness -= s.NOOP_ACTION_PENALTY
    
    return int(fitness)