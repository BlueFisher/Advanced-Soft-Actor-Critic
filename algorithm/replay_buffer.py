from collections import deque
import math
import random
import time

import numpy as np


class DataStorage:
    _size = 0
    _pointer = 0
    _buffer = None

    def __init__(self, capacity):
        # TODO: MongoDB
        self.capacity = capacity
        self.episode_index = list()

    def add(self, data: dict):
        """
        args: list
            The first dimension of each element is the length of an episode
        """
        tmp_len = list(data.values())[0].shape[0]

        if self._buffer is None:
            self._buffer = dict()
            for k, v in data.items():
                self._buffer[k] = np.empty([self.capacity] + list(v.shape[1:]), dtype=np.float32)

        pointers = (np.arange(tmp_len) + self._pointer) % self.capacity
        for k, v in data.items():
            self._buffer[k][pointers] = v

        self._size += tmp_len
        if self._size > self.capacity:
            self._size = self.capacity

        self._pointer = pointers[-1] + 1
        if self._pointer == self.capacity:
            self._pointer = 0

        return pointers

    def update(self, pointers, key, data):
        self._buffer[key][pointers] = data

    def get(self, pointers):
        return {k: v[pointers] for k, v in self._buffer.items()}

    def clear(self):
        self._size = 0
        self._pointer = 0
        self._buffer = None

    @property
    def size(self):
        return self._size

    @property
    def is_full(self):
        return self._size == self.capacity


class ReplayBuffer:
    _data_pointer = 0

    def __init__(self, batch_size=256, capacity=1e6, **kwargs):
        self.batch_size = int(batch_size)
        self.capacity = int(capacity)

        self._trans_storage = EpisodeStorage(self.capacity)

    def add(self, *transitions):
        self._trans_storage.add(transitions)

    def sample(self):
        if not self.is_lg_batch_size:
            return None

        pointers = np.random.choice(range(self._trans_storage.size), size=self.batch_size, replace=False)

        transitions = self._trans_storage.get(pointers)
        return pointers, transitions

    def update_transitions(self, pointers, index, data):
        assert len(pointers) == len(data)

        self._trans_storage.update(pointers, index, data)

    @property
    def is_full(self):
        return self._trans_storage.is_full

    @property
    def size(self):
        return self._trans_storage.size

    @property
    def is_lg_batch_size(self):
        return self._trans_storage.size > self.batch_size


class SumTree:
    def __init__(self, capacity):
        capacity = int(capacity)
        assert capacity & (capacity - 1) == 0

        self.capacity = capacity  # for all priority values
        self.depth = int(math.log2(capacity)) + 1

        self._tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity

    def add(self, data_idx, p):
        tree_idx = self.data_idx_to_leaf_idx(data_idx)
        self.update(tree_idx, p)  # update tree_frame

    def update(self, tree_idx, p):
        self._tree[tree_idx] = p

        for _ in range(self.depth - 1):
            parent_idx = (tree_idx - 1) // 2
            parent_idx = np.unique(parent_idx)
            node1 = self._tree[parent_idx * 2 + 1]
            node2 = self._tree[parent_idx * 2 + 2]
            self._tree[parent_idx] = node1 + node2

            tree_idx = parent_idx

    def sample(self, batch_size):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        pri_seg = self.total_p / batch_size       # priority segment
        pri_seg_low = np.arange(batch_size)
        pri_seg_high = pri_seg_low + 1
        v = np.random.uniform(pri_seg_low * pri_seg, pri_seg_high * pri_seg)
        leaf_idx = np.zeros(batch_size, dtype=np.int32)

        for _ in range(self.depth - 1):
            node1 = leaf_idx * 2 + 1
            node2 = leaf_idx * 2 + 2
            t = np.logical_or(v <= self._tree[node1], self._tree[node2] == 0)
            leaf_idx[t] = node1[t]
            leaf_idx[~t] = node2[~t]
            v[~t] -= self._tree[node1[~t]]

        return leaf_idx, self._tree[leaf_idx]

    def data_idx_to_leaf_idx(self, data_idx):
        return data_idx + self.capacity - 1

    def leaf_idx_to_data_idx(self, leaf_idx):
        return leaf_idx - self.capacity + 1

    def clear(self):
        self._tree[:] = 0

    def display(self):
        for i in range(self.depth):
            print(self._tree[2**i - 1:2**(i + 1) - 1])

    @property
    def total_p(self):
        return self._tree[0]  # the root

    @property
    def max(self):
        return self._tree[self.capacity - 1:].max()


class PrioritizedReplayBuffer:
    def __init__(self,
                 batch_size=256,
                 capacity=524288,
                 alpha=0.9,  # [0~1] convert the importance of TD error to priority
                 beta=0.4,  # importance-sampling, from initial value increasing to 1
                 beta_increment_per_sampling=0.001,
                 td_error_min=0.01,  # small amount to avoid zero priority
                 td_error_max=1.,  # clipped abs error
                 use_mongodb=False):
        self.batch_size = batch_size
        self.capacity = int(2**math.floor(math.log2(capacity)))
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.td_error_min = td_error_min
        self.td_error_max = td_error_max
        self._sum_tree = SumTree(self.capacity)
        self._trans_storage = DataStorage(self.capacity)

    def add(self, transitions: dict, ignore_size=0):
        if self._trans_storage.size == 0:
            max_p = self.td_error_max
        else:
            max_p = self._sum_tree.max

        data_pointers = self._trans_storage.add(transitions)
        probs = np.full(len(data_pointers), max_p, dtype=np.float32)
        # don't sample last ignore_size transitions
        probs[np.isin(data_pointers, np.arange(self.capacity - ignore_size, self.capacity))] = 0
        probs[-ignore_size:] = 0
        self._sum_tree.add(data_pointers, probs)

    def add_with_td_error(self, td_error, transitions: dict, ignore_size=0):
        td_error = np.asarray(td_error)
        td_error = td_error.flatten()

        data_pointers = self._trans_storage.add(transitions)
        clipped_errors = np.clip(td_error, self.td_error_min, self.td_error_max)
        probs = np.power(clipped_errors, self.alpha)
        # don't sample last ignore_size transitions
        probs[np.isin(data_pointers, np.arange(self.capacity - ignore_size, self.capacity))] = 0
        probs[-ignore_size:] = 0
        self._sum_tree.add(data_pointers, probs)

    def sample(self):
        if not self.is_lg_batch_size:
            return None

        leaf_pointers, p = self._sum_tree.sample(self.batch_size)

        trans_pointers = self._sum_tree.leaf_idx_to_data_idx(leaf_pointers)
        transitions = self._trans_storage.get(trans_pointers)

        is_weights = p * np.float32(self.size) / self._sum_tree.total_p
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1
        is_weights = np.power(is_weights, -self.beta).astype(np.float32)

        return leaf_pointers, transitions, np.expand_dims(is_weights, axis=1)

    def get_storage_data(self, leaf_pointers):
        return self._trans_storage.get(self._sum_tree.leaf_idx_to_data_idx(leaf_pointers))

    def update(self, leaf_pointers, td_error):
        td_error = np.asarray(td_error)
        td_error = td_error.flatten()

        clipped_errors = np.clip(td_error, self.td_error_min, self.td_error_max)
        if np.isnan(np.min(clipped_errors)):
            raise RuntimeError('td_error has nan')
        probs = np.power(clipped_errors, self.alpha)

        self._sum_tree.update(leaf_pointers, probs)

    def update_transitions(self, leaf_pointers, key, data):
        data_pointers = self._sum_tree.leaf_idx_to_data_idx(leaf_pointers)
        self._trans_storage.update(data_pointers, key, data)

    def clear(self):
        self._trans_storage.clear()
        self._sum_tree.clear()

    @property
    def is_full(self):
        return self._trans_storage.is_full

    @property
    def size(self):
        return self._trans_storage.size

    @property
    def is_lg_batch_size(self):
        return self._trans_storage.size > self.batch_size


if __name__ == "__main__":
    import time
    replay_buffer = PrioritizedReplayBuffer(256, 524288)
    n_step = 5

    while True:
        s = np.random.randint(n_step + 1, 500)
        replay_buffer.add({
            'state': np.random.randn(s, 1),
            'action': np.random.randn(s)
        }, ignore_size=n_step)
        # print(replay_buffer._sum_tree._tree[replay_buffer.capacity - 1:])
        sampled = replay_buffer.sample()
        if sampled is None:
            print('None')
        else:
            points, (a, b), ratio = sampled
            print(a, b)
            # print(replay_buffer._sum_tree.leaf_idx_to_data_idx(points))
            for i in range(1, n_step + 1):
                replay_buffer.get_storage_data(points + i)
            replay_buffer.update(points, np.random.random(len(points)).astype(np.float32))

        # input()
