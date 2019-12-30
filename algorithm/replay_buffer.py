from collections import deque
import math
import random
import time

import numpy as np


class EpisodeBuffer:
    def __init__(self, batch_step_size=16, batch_size=32, capacity=50):
        self.batch_step_size = batch_step_size
        self.batch_size = batch_size
        self.capacity = capacity
        self._buffer = deque(maxlen=capacity)

    def add(self, *episode):
        # n_states, n_actions, n_rewards, state_, done
        self._buffer.append(episode)

    @property
    def max_episode_len(self):
        lens = [t[0].shape[1] for t in self._buffer]
        return max(lens)

    def get_min_episode_len(self, buffer):
        lens = [t[0].shape[1] for t in buffer]
        return min(lens)

    def sample(self, fn_get_next_rnn_state):
        if len(self._buffer) < self.batch_size:
            return None

        buffer_sampled = random.sample(self._buffer, self.batch_size)

        batch_step_size = min(self.get_min_episode_len(buffer_sampled), self.batch_step_size)

        start_list = list()
        n_states_for_next_rnn_state_list = list()
        for episode in buffer_sampled:
            n_states, n_actions, state_ = episode[0], episode[1], episode[3]

            episode_len = n_states.shape[1]
            start = np.random.randint(0, episode_len - batch_step_size + 1)
            start_list.append(start)
            n_states_for_next_rnn_state_list.append(n_states[:, 0:start, :])

        rnn_state = fn_get_next_rnn_state(n_states_for_next_rnn_state_list)
        if rnn_state is None:
            return None

        results = None
        for i, episode in enumerate(buffer_sampled):
            n_states, n_actions, state_ = episode[0], episode[1], episode[3]
            start = start_list[i]

            m_states = np.concatenate([n_states, np.reshape(state_, (-1, 1, state_.shape[-1]))], axis=1)
            m_states = m_states[:, start:start + batch_step_size + 1, :]
            n_actions = n_actions[:, start:start + batch_step_size, :]

            if results is None:
                results = [m_states, n_actions]
            else:
                results[0] = np.concatenate([results[0], m_states], axis=0)
                results[1] = np.concatenate([results[1], n_actions], axis=0)

        # m_states, n_actions, rnn_state
        return results[0], results[1], rnn_state

    def sample_without_rnn_state(self):
        if len(self._buffer) < self.batch_size:
            return None

        buffer_sampled = random.sample(self._buffer, self.batch_size)

        batch_step_size = min(self.get_min_episode_len(buffer_sampled), self.batch_step_size)

        n_states_for_next_rnn_state_list = list()
        results = None

        for episode in buffer_sampled:
            n_states, n_actions, state_ = episode[0], episode[1], episode[3]

            episode_len = n_states.shape[1]
            start = np.random.randint(0, episode_len - batch_step_size + 1)

            n_states_for_next_rnn_state_list.append(n_states[:, 0:start, :])

            m_states = np.concatenate([n_states, np.reshape(state_, (-1, 1, state_.shape[-1]))], axis=1)
            m_states = m_states[:, start:start + batch_step_size + 1, :]
            n_actions = n_actions[:, start:start + batch_step_size, :]

            if results is None:
                results = [m_states, n_actions]
            else:
                results[0] = np.concatenate([results[0], m_states], axis=0)
                results[1] = np.concatenate([results[1], n_actions], axis=0)

        # n_states_list, m_states, n_actions
        return n_states_for_next_rnn_state_list, results


class DataStorage:
    _size = 0
    _pointer = 0
    _buffer = None

    def __init__(self, capacity):
        # TODO: MongoDB
        self.capacity = capacity

    def add(self, args):
        """
        args: list
            The first dimension of each element is batch size
        """
        tmp_len = len(args[0])

        if self._buffer is None:
            self._buffer = [np.empty([self.capacity] + list(a.shape[1:]), dtype=np.float32) for a in args]

        pointers = (np.arange(tmp_len) + self._pointer) % self.capacity
        for i, arg in enumerate(args):
            self._buffer[i][pointers] = arg

        self._size += tmp_len
        if self._size > self.capacity:
            self._size = self.capacity

        self._pointer = pointers[-1] + 1
        if self._pointer == self.capacity:
            self._pointer = 0

        return pointers

    def update(self, pointers, index, data):
        self._buffer[index][pointers] = data

    def get(self, pointers):
        return [data[pointers] for data in self._buffer]

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

        self._trans_storage = DataStorage(self.capacity)

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

        self._tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity

    def add(self, data_idx, p):
        tree_idx = self.data_idx_to_leaf_idx(data_idx)
        self.update(tree_idx, p)  # update tree_frame

    def update(self, tree_idx, p):
        self._tree[tree_idx] = p

        for i in range(self.depth - 2, -1, -1):
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
            t = v <= self._tree[node1]
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

    @property
    def total_p(self):
        return self._tree[0]  # the root

    @property
    def max(self):
        return self._tree[self.capacity - 1:].max()


class PrioritizedReplayBuffer:
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.9  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    td_err_upper = 1.  # clipped abs error

    def __init__(self,
                 batch_size=256,
                 capacity=524288,
                 alpha=0.9,
                 use_mongodb=False):
        self.batch_size = batch_size
        self.capacity = int(2**math.floor(math.log2(capacity)))
        self.alpha = alpha
        self._sum_tree = SumTree(self.capacity)
        self._trans_storage = DataStorage(self.capacity)

    def add(self, *transitions):
        if self._trans_storage.size == 0:
            max_p = self.td_err_upper
        else:
            max_p = self._sum_tree.max

        data_pointers = self._trans_storage.add(transitions)
        self._sum_tree.add(data_pointers, max_p)

    def add_with_td_error(self, td_error, *transitions):
        td_error = np.asarray(td_error)
        if len(td_error.shape) == 2:
            td_error = td_error.flatten()

        td_error = np.maximum(td_error, 0) + self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(td_error, self.td_err_upper)
        probs = np.power(clipped_errors, self.alpha)

        data_pointers = self._trans_storage.add(transitions)
        self._sum_tree.add(data_pointers, probs)

    def sample(self):
        if not self.is_lg_batch_size:
            return None

        leaf_pointers, p = self._sum_tree.sample(self.batch_size)

        trans_pointers = self._sum_tree.leaf_idx_to_data_idx(leaf_pointers)
        transitions = self._trans_storage.get(trans_pointers)

        is_weights = p / self._sum_tree.total_p
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1
        is_weights = np.power(is_weights / is_weights.min(), -self.beta)

        return leaf_pointers, transitions, is_weights

    def update(self, leaf_pointers, td_error):
        td_error = np.asarray(td_error)
        if len(td_error.shape) == 2:
            td_error = td_error.flatten()

        td_error += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(td_error, self.td_err_upper)
        probs = np.power(clipped_errors, self.alpha)

        self._sum_tree.update(leaf_pointers, probs)

    def update_transitions(self, leaf_pointers, index, data):
        assert len(leaf_pointers) == len(data)

        data_pointers = self._sum_tree.leaf_idx_to_data_idx(leaf_pointers)
        self._trans_storage.update(data_pointers, index, data)

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

    # replay_buffer = ReplayBuffer()
    # replay_buffer.add(np.random.randn(200, 3, 2), np.random.randn(200, 5))
    # replay_buffer.add(np.random.randn(200, 3, 2), np.random.randn(200, 5))

    # replay_buffer.update_transitions([0, 1], 1, np.random.randn(2, 5))

    # pointers, data = replay_buffer.sample()
    # print(len(pointers))

    # for d in data:
    #     print(d.shape)

    replay_buffer = PrioritizedReplayBuffer(50, 120)

    for i in range(10):
        replay_buffer.add(np.random.randn(100, 1), np.random.randn(100, 5))
        sampled = replay_buffer.sample()
        if sampled is None:
            print('None')
        else:
            points, (a, b), ratio = sampled
