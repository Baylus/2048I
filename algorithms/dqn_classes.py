import numpy as np
import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# import keras with tensorflow's warning message disabled
import shelve

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0

    def add(self, priority, data):
        tree_index = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_index, priority)
        
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0  # Wrap around

    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        self._propagate(tree_index, change)

    def _propagate(self, tree_index, change):
        parent = (tree_index - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def get_leaf(self, value):
        parent_index = 0
        while True:
            left_child = 2 * parent_index + 1
            right_child = left_child + 1
            
            if left_child >= len(self.tree):
                leaf_index = parent_index
                break
            else:
                if value <= self.tree[left_child]:
                    parent_index = left_child
                else:
                    value -= self.tree[left_child]
                    parent_index = right_child
                    
        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    def total_priority(self):
        return self.tree[0]

class ReplayBuffer:
    def close(self):
        print("WARNING: We tried closing our non-shelved replay buffer. If we meant to use a shelf, something went wrong.")
        return
    
    def save(self):
        print("WARNING: We tried saving our non-shelved replay buffer. If we meant to use a shelf, something went wrong.")
        return


class PrioritizedReplayMemory(ReplayBuffer):
    def __init__(self, capacity, alpha=0.6):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha  # Priority scaling factor
        self.epsilon = 0.01  # Small value to avoid zero priority

    def store(self, experience, priority=1.0):
        # Adjust priority to be proportional with alpha
        adjusted_priority = (priority + self.epsilon) ** self.alpha
        self.tree.add(adjusted_priority, experience)

    def sample(self, batch_size, beta=0.4):
        # Sampling based on priority
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total_priority() / batch_size
        
        for i in range(batch_size):
            value = random.uniform(i * segment, (i + 1) * segment)
            idx, priority, data = self.tree.get_leaf(value)
            sampling_prob = priority / self.tree.total_priority()
            importance_sampling_weight = (self.capacity * sampling_prob) ** (-beta)
            batch.append((data, importance_sampling_weight))
            idxs.append(idx)
            priorities.append(priority)

        return batch, idxs, priorities

    def update_priorities(self, idxs, priorities):
        for idx, priority in zip(idxs, priorities):
            self.tree.update(idx, (priority + self.epsilon) ** self.alpha)

    def __len__(self):
        return len(self.tree.data)


class ShelvedPrioritizedReplayMemory(ReplayBuffer):
    def __init__(self, capacity, reset=False, mem_file_name="replay_memory.shelve"):
        self.capacity = capacity
        self.memory_file = shelve.open(mem_file_name, writeback=True)
        if "buffer" not in self.memory_file or reset:
            # Initialize buffer if starting fresh
            self.memory_file["buffer"] = []
            self.memory_file["priorities"] = []
        self.unsaved_changes = False  # Flag to track if there are unsaved changes
        self.data_pointer = 0  # Pointer for circular buffer behavior

    def store(self, experience, priority):
        if len(self.memory_file["buffer"]) < self.capacity:
            # Add new experience if the buffer isn't full
            self.memory_file["buffer"].append(experience)
            self.memory_file["priorities"].append(priority)
        else:
            # Overwrite the oldest experience and priority in a circular manner
            self.memory_file["buffer"][self.data_pointer] = experience
            self.memory_file["priorities"][self.data_pointer] = priority
            self.data_pointer = (self.data_pointer + 1) % self.capacity
        
        self.unsaved_changes = True

    def sample(self, batch_size, beta=0.4):
        priorities = np.array(self.memory_file["priorities"])
        # Calculate sampling probabilities based on priority
        probabilities = priorities ** 0.6  # alpha = 0.6 for scaling
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.memory_file["buffer"]), batch_size, p=probabilities)
        samples = [self.memory_file["buffer"][i] for i in indices]
        
        # Importance sampling weights
        weights = (len(self.memory_file["buffer"]) * probabilities[indices]) ** (-beta)
        weights /= weights.max()  # Normalize for stability
        
        return samples, indices, weights

    def update_priorities(self, indices, new_priorities):
        for idx, priority in zip(indices, new_priorities):
            self.memory_file["priorities"][idx] = priority
        self.unsaved_changes = True

    def save(self):
        if self.unsaved_changes:
            self.memory_file.sync()  # Save any unsaved changes to disk
            self.unsaved_changes = False

    def close(self):
        self.save()
        self.memory_file.close()

    def __len__(self):
        return len(self.memory_file["buffer"])
