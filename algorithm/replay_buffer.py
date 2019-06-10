import math
import multiprocessing
import random
import threading
import time
import os

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


def sample_process_worker(tree, sample_leaf_idxes, pipe):
    while True:
        v_i_list = pipe.recv()

        for v, i in v_i_list:
            parent_idx = 0

            while True:     # the while loop is faster than the method in the reference code
                cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
                cr_idx = cl_idx + 1
                if cl_idx >= len(tree):        # reach bottom, end search
                    leaf_idx = parent_idx
                    break
                else:       # downward search, always search for a higher priority node
                    if v <= tree[cl_idx]:
                        parent_idx = cl_idx
                    else:
                        v -= tree[cl_idx]
                        parent_idx = cr_idx

            sample_leaf_idxes[i] = leaf_idx

        pipe.send(True)


class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Story data with its priority in the tree.
    """
    _data_pointer = 0
    _size = 0
    _min = 0
    _max = 0

    def __init__(self, batch_size, capacity):
        self.batch_size = batch_size
        self.capacity = capacity  # for all priority values
        self._tree = multiprocessing.Array('d', [0] * (2 * capacity - 1), lock=False)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self._sample_leaf_idxes = multiprocessing.Array('i', [0] * batch_size, lock=False)
        self._data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

        pipe_tuples = [multiprocessing.Pipe() for i in range(4)]
        self._pipes = [p[0] for p in pipe_tuples]

        for p in pipe_tuples:
            multiprocessing.Process(target=sample_process_worker,
                                    args=(self._tree, self._sample_leaf_idxes, p[1])).start()

    def add(self, p, data):
        if self._size == 0:
            self._min = self._max = p
        else:
            if p < self._min:
                self._min = p
            if p > self._max:
                self._max = p

        tree_idx = self._data_pointer + self.capacity - 1
        self._data[self._data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self._data_pointer += 1
        if self._data_pointer >= self.capacity:  # replace when exceed the capacity
            self._data_pointer = 0

        if self._size < self.capacity:
            self._size += 1

    def update(self, tree_idx, p):
        if p < self._min:
            self._min = p
        if p > self._max:
            self._max = p

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

    def get_leaves_parallel(self, v_list):
        v_i_list = [(v, i) for i, v in enumerate(v_list)]
        seg = math.ceil(len(v_i_list) / len(self._pipes))

        for i, pipe in enumerate(self._pipes):
            pipe.send(v_i_list[seg * i:seg * (i + 1)])

        for pipe in self._pipes:
            done = pipe.recv()

        idx = np.array(self._sample_leaf_idxes, dtype=np.int32)
        p = np.array([self._tree[i] for i in idx])
        data = np.array([self._data[i - self.capacity + 1] for i in idx], dtype=object)

        return idx, p, data

    def clear(self):
        for i in range(len(self._tree)):
            self._tree[i] = 0
        self._size = 0

    @property
    def total_p(self):
        return self._tree[0]  # the root

    @property
    def max(self):
        return self._max

    @property
    def min(self):
        return self._min

    @property
    def size(self):
        return self._size


class PrioritizedReplayBuffer(object):  # stored as ( s, a, r, s_, done) in SumTree
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.9  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    td_err_upper = 1.  # clipped abs error

    _lock = threading.Lock()

    def __init__(self,
                 batch_size=256,
                 capacity=524288,
                 alpha=0.9):
        self.batch_size = batch_size
        self.capacity = int(2**math.floor(math.log2(capacity)))
        self.alpha = alpha
        self._sum_tree = SumTree(self.batch_size, self.capacity)

    def add(self, *args):
        self._lock.acquire()
        max_p = self._sum_tree.max
        if max_p == 0:
            max_p = self.td_err_upper

        for i in range(len(args[0])):
            self._sum_tree.add(max_p, tuple(arg[i] for arg in args))
        self._lock.release()

    def add_with_td_errors(self, td_errors, *args):
        assert len(td_errors) == len(args[0])

        td_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(td_errors, self.td_err_upper)
        probs = np.power(clipped_errors, self.alpha)

        self._lock.acquire()
        for i in range(len(args[0])):
            self._sum_tree.add(probs[i], tuple(t[i] for t in args))
        self._lock.release()

    def sample(self):
        if not self.is_lg_batch_size:
            return None

        self._lock.acquire()

        pri_seg = self._sum_tree.total_p / self.batch_size       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = self._sum_tree.min / self._sum_tree.total_p     # for later calculate ISweight
        if min_prob == 0:
            min_prob = self.epsilon

        v_list = list()
        for i in range(self.batch_size):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            v_list.append(v)

        points, probs, transitions = self._sum_tree.get_leaves_parallel(v_list)

        probs = probs / self._sum_tree.total_p
        is_weights = np.power(probs / min_prob, -self.beta)

        self._lock.release()

        return points, [np.array(e) for e in zip(*transitions)], is_weights[:, np.newaxis]

    def update(self, points, td_errors):
        td_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(td_errors, self.td_err_upper)
        ps = np.power(clipped_errors, self.alpha)

        self._lock.acquire()
        for ti, p in zip(points, ps):
            self._sum_tree.update(ti, p)
        self._lock.release()

    def clear(self):
        self._sum_tree.clear()

    @property
    def size(self):
        return self._sum_tree.size

    @property
    def is_lg_batch_size(self):
        return self.size > self.batch_size

    @property
    def is_full(self):
        return self.size == self.capacity


if __name__ == "__main__":
    import time

    replay_buffer = PrioritizedReplayBuffer(1024)

    while True:
        replay_buffer.add_with_td_errors(np.abs(np.random.randn(1029)), np.random.randn(1029, 1), np.random.randn(1029, 2))
        points, (a, b), ratio = replay_buffer.sample()
