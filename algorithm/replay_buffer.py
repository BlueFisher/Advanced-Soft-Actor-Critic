import random
import math
import time
import threading
from concurrent.futures import ThreadPoolExecutor


import numpy as np


class ReplayBuffer(object):
    _data_pointer = 0
    _size = 0

    def __init__(self, batch_size=256, capacity=1e6):
        self.batch_size = int(batch_size)
        self.capacity = int(capacity)

        self._buffer = np.empty(self.capacity, dtype=object)

    def add(self, *args):
        for arg in args:
            assert len(arg.shape) == 2
            assert len(arg) == len(args[0])

        for i in range(len(args[0])):
            self._buffer[self._data_pointer] = tuple(arg[i] for arg in args)
            self._data_pointer += 1

            if self._data_pointer >= self.capacity:  # replace when exceed the capacity
                self._data_pointer = 0

            if self._size < self.capacity:
                self._size += 1

    def sample(self):
        n_sample = self.batch_size if self.is_lg_batch_size else self._size
        t = np.random.choice(self._buffer[:self._size], size=n_sample, replace=False)
        return [np.array(e) for e in zip(*t)]

    @property
    def is_full(self):
        return self._size == self.capacity

    @property
    def size(self):
        return self._size

    @property
    def is_lg_batch_size(self):
        return self._size > self.batch_size


class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Story data with its priority in the tree.
    """
    _data_pointer = 0
    _size = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self._tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self._data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self._data_pointer + self.capacity - 1
        self._data[self._data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self._data_pointer += 1
        if self._data_pointer >= self.capacity:  # replace when exceed the capacity
            self._data_pointer = 0

        if self._size < self.capacity:
            self._size += 1

        # if self._size < self.capacity:
        #     tree_idx = self._size + self.capacity - 1
        #     self._data[self._size] = data  # update data_frame
        #     self._size += 1
        # else:
        #     leaves = self._tree[self.capacity - 1:]
        #     data_idx = np.where(leaves == np.min(leaves))[0][0]
        #     tree_idx = data_idx + self.capacity - 1
        #     self._data[data_idx] = data

        # self.update(tree_idx, p)  # update tree_frame

    def update(self, tree_idx, p):
        change = p - self._tree[tree_idx]
        self._tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self._tree[tree_idx] += change

    def get_leaf(self, v):
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
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self._tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self._tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self._tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self._tree[leaf_idx], self._data[data_idx]

    def clear(self):
        self._size = 0

    def get_leaves(self):
        return self._tree[self.capacity - 1:self._size + self.capacity - 1]

    @property
    def total_p(self):
        return self._tree[0]  # the root

    @property
    def max(self):
        if self._size == 0:
            return 0
        return np.max(self._tree[self.capacity - 1:self._size + self.capacity - 1])

    @property
    def min(self):
        return np.min(self._tree[self.capacity - 1:self._size + self.capacity - 1])

    @property
    def size(self):
        return self._size


class PrioritizedReplayBuffer(object):  # stored as ( s, a, r, s_, done) in SumTree
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.9  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    td_err_upper = 1.  # clipped abs error
    # pool = ThreadPoolExecutor(max_workers=2)
    # thread_batch = 256

    _lock = threading.Lock()

    def __init__(self,
                 batch_size=256,
                 capacity=1e6,
                 alpha=0.9):
        self.batch_size = batch_size
        self.capacity = 2**math.floor(math.log2(capacity))
        self.alpha = alpha
        self._tree = SumTree(self.capacity)

    def add(self, *args):
        self._lock.acquire()
        max_p = self._tree.max
        if max_p == 0:
            max_p = self.td_err_upper

        for i in range(len(args[0])):
            self._tree.add(max_p, tuple(arg[i] for arg in args))
        self._lock.release()

    def add_with_td_errors(self, td_errors, *args):
        assert len(td_errors) == len(args[0])

        td_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(td_errors, self.td_err_upper)
        ps = np.power(clipped_errors, self.alpha)

        self._lock.acquire()
        for i in range(len(args[0])):
            self._tree.add(ps[i], tuple(t[i] for t in args))
        self._lock.release()

    def sample(self):
        self._lock.acquire()
        start = time.time()

        n_sample = self.batch_size if self.is_lg_batch_size else self.size
        if n_sample == 0:
            return None

        points, transitions, is_weights = np.empty((n_sample,), dtype=np.int32), np.empty((n_sample,), dtype=object), np.empty((n_sample, 1))
        pri_seg = self._tree.total_p / n_sample       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = self._tree.min / self._tree.total_p     # for later calculate ISweight
        if min_prob == 0:
            min_prob = self.epsilon

        # def cal(s_i):
            # for i in range(s_i, min(s_i + self.thread_batch, n_sample)):

        for i in range(n_sample):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self._tree.get_leaf(v)
            prob = p / self._tree.total_p

            is_weights[i, 0] = np.power(prob / min_prob, -self.beta)
            points[i], transitions[i] = idx, data

        # for _ in self.pool.map(cal, range(0, n_sample, self.thread_batch)):
        #     pass

        print(time.time() - start)
        self._lock.release()

        return points, [np.array(e) for e in zip(*transitions)], is_weights

    def update(self, points, td_errors):
        td_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(td_errors, self.td_err_upper)
        ps = np.power(clipped_errors, self.alpha)

        self._lock.acquire()
        for ti, p in zip(points, ps):
            self._tree.update(ti, p)
        self._lock.release()

    def clear(self):
        self._tree.clear()

    def get_leaves(self):
        return self._tree.get_leaves()

    @property
    def size(self):
        return self._tree.size

    @property
    def is_lg_batch_size(self):
        return self.size > self.batch_size

    @property
    def is_full(self):
        return self.size == self.capacity


if __name__ == "__main__":
    import time

    # replay_buffer = ReplayBuffer(256, 1000000)
    # for i in range(1000000):
    #     start = time.time()
    #     replay_buffer.add(np.random.randn(9, 1), np.random.randn(9, 2))
    #     print('add', time.time() - start)
    #     print(replay_buffer.size)

    #     start = time.time()
    #     data = replay_buffer.sample()
    #     print('sample', time.time() - start)

    #     print('=' * 10)

    replay_buffer = PrioritizedReplayBuffer(1024, 1e6)
    replay_buffer.add(np.random.randn(2048, 1), np.random.randn(2048, 2))

    while True:
        start = time.time()
        points, (a, b), ratio = replay_buffer.sample()
        print(time.time() - start)

        print('=' * 10)
