import io
import logging
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from tqdm import tqdm

from .utils import ReadWriteLock


class DataStorage:
    _size = 0
    _id = 0
    _buffer = None

    def __init__(self, capacity):
        self.capacity = capacity
        self.max_id = 10 * capacity  # For multithreading storage

        self._data_key_is_image = {}

    def add(self, data: dict) -> np.ndarray:
        """
        args: list
            The first dimension of each element is the length of an episode
        """
        tmp_len = list(data.values())[0].shape[0]

        if self._buffer is None:
            self._buffer = dict()
            self._buffer['_id'] = np.zeros(self.capacity, dtype=np.int64)
            for k, v in data.items():
                # Store uint8 if data is image
                self._data_key_is_image[k] = ('camera' in k.lower()
                                              or 'visual' in k.lower()
                                              or 'image' in k.lower()
                                              or 'segmentation' in k.lower()) and v.shape[-1] == 3

                dtype = np.uint8 if self._data_key_is_image[k] else v.dtype

                self._buffer[k] = np.zeros([self.capacity] + list(v.shape[1:]), dtype=dtype)

        ids = (np.arange(tmp_len) + self._id) % self.max_id
        pointers = ids % self.capacity

        self._buffer['_id'][pointers] = ids
        for k, v in data.items():
            # Store uint8 [0, 255] if data is image
            if self._data_key_is_image[k]:
                v = v * 255
            self._buffer[k][pointers] = v

        self._size = min(self._size + tmp_len, self.capacity)

        self._id = ids[-1] + 1
        if self._id == self.max_id:
            self._id = 0

        return pointers

    def get_curr_id(self) -> int:
        return self._id % self.capacity

    def update(self, ids, key, data):
        self._buffer[key][ids % self.capacity] = data

    def get(self, ids) -> Dict[str, np.ndarray]:
        """
        Get data from buffer without verifying whether ids in buffer
        """
        data = {}
        for k, v in self._buffer.items():
            if k == '_id':
                continue

            data[k] = v[ids % self.capacity]
            # It is faster to convert image uint8 to float32 in GPU
            # if self._data_key_is_image[k]:
            #     data[k] = data[k].astype(np.float32) / 255.

        return data

    def get_ids(self, ids) -> np.ndarray:
        """
        Get true data ids
        """
        return self._buffer['_id'][ids % self.capacity]

    def save(self, path: Path) -> None:
        np.savez(path, **self._buffer, p_size=self._size, p_id=self._id)

    def load(self, path: Path) -> None:
        if not path.exists():
            return

        saved = np.load(path)
        self._size = saved['p_size']
        self._id = saved['p_id']

        if self._buffer is None:
            self._buffer = dict()
        for k, v in saved.items():
            if k not in ('p_size', 'p_id'):
                self._buffer[k] = v

    def copy(self, src) -> None:
        src: DataStorage = src

        if self._size == 0:
            self._buffer = dict()
            for k in src._buffer:
                self._buffer[k] = src._buffer[k].copy()
        else:
            for k in self._buffer:
                np.copyto(self._buffer[k], src._buffer[k])

        self._size = src._size
        self._id = src._id

    def clear(self) -> None:
        self._size = 0
        self._id = 0
        self._buffer = None

    @property
    def size(self) -> int:
        return self._size

    @property
    def is_full(self) -> bool:
        return self._size == self.capacity


class SumTree:
    def __init__(self, capacity):
        capacity = int(capacity)
        assert capacity & (capacity - 1) == 0

        self.capacity = capacity  # for all priority values
        self.depth = int(math.log2(capacity)) + 1

        self._tree = np.zeros(2 * capacity - 1, dtype=np.float32)
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

        [--------------Parent nodes-------------][-------leaves to recode priority-------]
                    size: capacity - 1                       size: capacity
        """

    def add(self, data_idx, p) -> None:
        self.update(data_idx, p)  # update tree_frame

    def update(self, data_idx, p) -> None:
        tree_idx = self.data_idx_to_leaf_idx(data_idx)
        self._tree[tree_idx] = p

        for _ in range(self.depth - 1):
            parent_idx = (tree_idx - 1) // 2
            parent_idx = np.unique(parent_idx)
            node1 = self._tree[parent_idx * 2 + 1]
            node2 = self._tree[parent_idx * 2 + 2]
            self._tree[parent_idx] = node1 + node2

            tree_idx = parent_idx

    def sample(self, batch_size) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            leaf_idx: The index of leaves of the whole sum tree
            p: The value of the leaf_idx
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

    def data_idx_to_leaf_idx(self, data_idx) -> np.ndarray:
        return data_idx + self.capacity - 1

    def leaf_idx_to_data_idx(self, leaf_idx) -> np.ndarray:
        return leaf_idx - self.capacity + 1

    def clear(self):
        self._tree[:] = 0

    def display(self):
        for i in range(self.depth):
            print(self._tree[2**i - 1:2**(i + 1) - 1])

    def save(self, path: Path):
        np.save(path, self._tree)

    def load(self, path: Path):
        if not path.exists():
            return

        self._tree = np.load(path)

    def copy(self, src):
        src: SumTree = src
        np.copyto(self._tree, src._tree)

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
                 alpha=0.9,  # [0~1] Convert the importance of TD error to priority
                 beta=0.4,  # Importance-sampling, from initial value increasing to 1
                 beta_increment_per_sampling=0.001,
                 td_error_min=0.01,  # Small amount to avoid zero priority
                 td_error_max=1.):  # Clipped abs error
        self.batch_size = batch_size
        self.capacity = int(2**math.floor(math.log2(capacity)))
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.td_error_min = td_error_min
        self.td_error_max = td_error_max
        self._sum_tree = SumTree(self.capacity)
        self._trans_storage = DataStorage(self.capacity)

        self._logger = logging.getLogger('replay_buffer')

        self._lock = ReadWriteLock(None, 1, 1, True, self._logger)

        tqdm_out = TqdmToLogger(self._logger, level=logging.INFO)
        self._fill_bar = tqdm(total=self.batch_size, desc='Filling the buffer...', file=tqdm_out)

    def add(self, transitions: Dict[str, np.ndarray], ignore_size=0) -> None:
        with self._lock.write():
            if self._trans_storage.size == 0:
                max_p = self.td_error_max
            else:
                max_p = self._sum_tree.max

            data_pointers = self._trans_storage.add(transitions)
            probs = np.full(len(data_pointers), max_p, dtype=np.float32)

            if ignore_size > 0:
                # Don't sample last ignore_size transitions
                probs[np.isin(data_pointers, np.arange(self.capacity - ignore_size, self.capacity))] = 0
                probs[-ignore_size:] = 0
            self._sum_tree.add(data_pointers, probs)

            if not self._fill_bar.disable:
                self._fill_bar.n = self._trans_storage.size
                self._fill_bar.refresh()

                if self._trans_storage.size >= self.batch_size:
                    self._fill_bar.close()

    def add_with_td_error(self, td_error: np.ndarray,
                          transitions: Dict[str, np.ndarray],
                          ignore_size=0) -> None:
        td_error = np.asarray(td_error)
        td_error = td_error.flatten()

        with self._lock.write():
            data_pointers = self._trans_storage.add(transitions)
            clipped_errors = np.clip(td_error, self.td_error_min, self.td_error_max)
            if np.isnan(np.min(clipped_errors)):
                self._logger.error('td_error has nan')
                raise Exception('td_error has nan')

            probs = np.power(clipped_errors, self.alpha)

            if ignore_size > 0:
                # Don't sample last ignore_size transitions
                probs[np.isin(data_pointers, np.arange(self.capacity - ignore_size, self.capacity))] = 0
                probs[-ignore_size:] = 0
            self._sum_tree.add(data_pointers, probs)

    def sample(self) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
        """
        Returns:
            data index: [batch, ]
            transitions: dict
            priority weights: [batch, 1]
        """
        with self._lock.read():
            if self._trans_storage.size < self.batch_size:
                return None

            leaf_pointers, p = self._sum_tree.sample(self.batch_size)

            data_pointers = self._sum_tree.leaf_idx_to_data_idx(leaf_pointers)
            transitions = self._trans_storage.get(data_pointers)
            data_ids = self._trans_storage.get_ids(data_pointers)

            is_weights = p / self._sum_tree.total_p
            self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1
            is_weights = np.power(is_weights / np.min(is_weights), -self.beta).astype(np.float32)

            return data_ids, transitions, np.expand_dims(is_weights, axis=1)

    def get_curr_id(self) -> int:
        return self._trans_storage.get_curr_id()

    def get_storage_data(self, data_ids) -> Dict[str, np.ndarray]:
        """
        Get data without verifying whether data_ids exist
        """
        with self._lock.read():
            return self._trans_storage.get(data_ids)

    def get_storage_data_ids(self, data_ids) -> np.ndarray:
        with self._lock.read():
            return self._trans_storage.get_ids(data_ids)

    def update(self, data_ids, td_error) -> None:
        with self._lock.write():
            td_error = np.asarray(td_error)
            td_error = td_error.flatten()

            clipped_errors = np.clip(td_error, self.td_error_min, self.td_error_max)
            if np.isnan(np.min(clipped_errors)):
                self._logger.error('td_error has nan')
                raise Exception('td_error has nan')

            probs = np.power(clipped_errors, self.alpha)

            self._sum_tree.update(data_ids % self.capacity, probs)

    def update_transitions(self, data_ids, key, data) -> None:
        with self._lock.write():
            self._trans_storage.update(data_ids, key, data)

    def save(self, save_dir: Path, ckpt: int) -> None:
        # for p in p.glob('*-rb_tree.npy'):
        #     p.unlink()
        # for p in p.glob('*-rb_storage.npz'):
        #     p.unlink()
        self._sum_tree.save(save_dir.joinpath(f'{ckpt}-rb_tree.npy'))
        self._trans_storage.save(save_dir.joinpath(f'{ckpt}-rb_storage.npz'))

    def load(self, save_dir: Path, ckpt: int) -> None:
        self._sum_tree.load(save_dir.joinpath(f'{ckpt}-rb_tree.npy'))
        self._trans_storage.load(save_dir.joinpath(f'{ckpt}-rb_storage.npz'))

    def clear(self) -> None:
        self._trans_storage.clear()
        self._sum_tree.clear()

    def copy(self, src) -> None:
        with self._lock.write(), src._lock.write():
            self._trans_storage.copy(src._trans_storage)
            self._sum_tree.copy(src._sum_tree)

    @property
    def is_full(self) -> bool:
        with self._lock.read():
            return self._trans_storage.is_full

    @property
    def size(self) -> int:
        with self._lock.read():
            return self._trans_storage.size

    @property
    def is_lg_batch_size(self) -> bool:
        with self._lock.read():
            return self._trans_storage.size > self.batch_size


if __name__ == "__main__":
    import time
    replay_buffer = PrioritizedReplayBuffer(16, 128)

    while True:
        batch = 6

        td_error = np.abs(np.random.randn(batch))

        action = np.zeros((batch,))
        action[-1] = 100

        replay_buffer.add_with_td_error(td_error, {
            'state': np.zeros((batch, 1)),
            'action': action,
            'test': np.arange(6)
        }, ignore_size=2)

        sampled = replay_buffer.sample()
        if sampled is None:
            print('None')
        else:
            # print(replay_buffer._sum_tree.total_p)
            # print(replay_buffer.size)
            points, trans, ratio = sampled
            print(points)
            # # print(replay_buffer._sum_tree.leaf_idx_to_data_idx(points))
            # for i in range(1, n_step + 1):
            #     replay_buffer.get_storage_data(points + i)
            # replay_buffer.update(points, np.random.random(len(points)).astype(np.float32))

        # input()


class TqdmToLogger(io.StringIO):
    """
        Output stream for TQDM which will output to logger module instead of
        the StdOut.
    """
    logger = None
    level = None
    buf = ''

    def __init__(self, logger, level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO

    def write(self, buf):
        self.buf = buf.strip('\r\n\t ')

    def flush(self):
        self.logger.log(self.level, self.buf)
