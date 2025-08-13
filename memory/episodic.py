import numpy as np
import random
import pickle

# A SumTree data structure for efficient prioritized sampling
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        if self.n_entries < self.capacity:
            # Buffer is not full, add to next available spot
            idx = self.data_pointer + self.capacity - 1
            self.data[self.data_pointer] = data
            self.update(idx, p)
            self.data_pointer += 1
            self.n_entries += 1
        else:
            # Buffer is full, find lowest priority leaf and replace it
            min_leaf_idx = np.argmin(self.tree[-self.capacity:])
            min_tree_idx = min_leaf_idx + self.capacity - 1

            self.data[min_leaf_idx] = data
            self.update(min_tree_idx, p)

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

class EpisodicMemory:
    """ Prioritized Experience Replay buffer """
    def __init__(self, capacity=100_000, alpha=0.6, beta0=0.4, beta_increment_per_sampling=0.001):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha  # [0,1], converts TD error to priority. a=0 means uniform sampling, a=1 means full prioritization.
        self.beta = beta0   # [0,1], importance-sampling correction. b=1 fully compensates for the non-uniform probabilities.
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.epsilon = 0.01  # Small value to ensure no transition has zero priority
        self.version = "1.0"

    def _get_priority(self, pred_err):
        return (np.abs(pred_err) + self.epsilon) ** self.alpha

    def add(self, s, a, r, s2, done, pred_err):
        priority = self._get_priority(pred_err)
        self.tree.add(priority, (s, a, r, s2, done, pred_err))

    def sample(self, batch_size):
        """ Returns (batch, importance_weights, tree_indices) """
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)

            if isinstance(data, np.ndarray) and data.size == 0:
                # Data not yet filled, sample again from the start
                s = random.uniform(0, segment)
                (idx, p, data) = self.tree.get(s)

            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max() # Normalize for stability

        return batch, is_weights, idxs

    def update_priorities(self, tree_indices, pred_errors):
        for i, err in zip(tree_indices, pred_errors):
            p = self._get_priority(err)
            self.tree.update(i, p)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
