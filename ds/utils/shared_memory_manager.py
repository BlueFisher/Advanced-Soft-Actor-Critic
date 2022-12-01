import logging
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from queue import Empty
from typing import Any, List, Optional, Tuple, Union

import numpy as np

from algorithm.utils import elapsed_counter, elapsed_timer, traverse_lists


class SharedMemoryManager:
    def __init__(self,
                 queue_size: int = 1,
                 logger: Optional[logging.Logger] = None,
                 counter_get_shm_index_empty_log: Optional[str] = None,
                 timer_get_shm_index_log: Optional[str] = None,
                 timer_get_data_log: Optional[str] = None,
                 counter_get_free_shm_index_empty_log: Optional[str] = None,
                 timer_get_free_shm_index_log: Optional[str] = None,
                 timer_put_data_log: Optional[str] = None,
                 log_repeat: int = 1):

        self.queue_size = queue_size
        self.counter_get_shm_index_empty_log = counter_get_shm_index_empty_log
        self.timer_get_shm_index_log = timer_get_shm_index_log
        self.timer_get_data_log = timer_get_data_log
        self.counter_get_free_shm_index_empty_log = counter_get_free_shm_index_empty_log
        self.timer_get_free_shm_index_log = timer_get_free_shm_index_log
        self.timer_put_data_log = timer_put_data_log
        self.log_repeat = log_repeat

        self.shm_index_queue = mp.Queue(queue_size)
        self.free_shm_index_queue = mp.Queue(queue_size)
        for i in range(queue_size):
            self.free_shm_index_queue.put(i)

        self.init_logger(logger)

    def init_logger(self, logger: Optional[logging.Logger] = None):
        self._counter_get_shm_index_empty = elapsed_counter(logger, self.counter_get_shm_index_empty_log, self.log_repeat)
        self._timer_get_shm_index = elapsed_timer(logger, self.timer_get_shm_index_log, self.log_repeat)
        self._timer_get_data = elapsed_timer(logger, self.timer_get_data_log, self.log_repeat)

        self._counter_get_free_shm_index_empty = elapsed_counter(logger, self.counter_get_free_shm_index_empty_log, self.log_repeat)
        self._timer_get_free_shm_index = elapsed_timer(logger, self.timer_get_free_shm_index_log, self.log_repeat)
        self._timer_put_data = elapsed_timer(logger, self.timer_put_data_log, self.log_repeat)

    def init_from_shapes(self, data_shapes, data_dtypes):
        self.buffer = traverse_lists((data_shapes, data_dtypes),
                                     lambda shape, dtype: np.zeros(shape, dtype=dtype))

        self.shms = traverse_lists(self.buffer,
                                   lambda b: [SharedMemory(create=True, size=b.nbytes) for _ in range(self.queue_size)])

    def init_from_data_buffer(self, data_buffer):
        self.buffer = data_buffer

        self.shms = traverse_lists(self.buffer,
                                   lambda b: [SharedMemory(create=True, size=b.nbytes) for _ in range(self.queue_size)])

    def get(self, timeout=None):
        with self._timer_get_shm_index, self._counter_get_shm_index_empty:
            try:
                shm_idx = self.shm_index_queue.get(timeout=timeout)
            except Empty:
                self._counter_get_shm_index_empty.add()
                self._timer_get_shm_index.ignore()
                return None, None

        # Copy shm to buffer
        def _tra(b: np.ndarray, shms: List[SharedMemory]):
            shm_np = np.ndarray(b.shape, dtype=b.dtype, buffer=shms[shm_idx].buf)
            np.copyto(b, shm_np)

        with self._timer_get_data:
            traverse_lists((self.buffer, self.shms), _tra)

        self.free_shm_index_queue.put(shm_idx)

        return self.buffer, shm_idx

    def put(self, data, pop_last=True, timeout=None):
        if pop_last:
            if self.shm_index_queue.full():
                shm_idx = self.shm_index_queue.get()
            else:
                shm_idx = self.free_shm_index_queue.get()
        else:
            with self._timer_get_free_shm_index, self._counter_get_free_shm_index_empty:
                try:
                    shm_idx = self.free_shm_index_queue.get(timeout=timeout)
                except Empty:
                    self._counter_get_free_shm_index_empty.add()
                    self._timer_get_free_shm_index.ignore()
                    return None

        # Copy data to shm
        def _tra(d: np.ndarray, shms: List[SharedMemory]):
            if len(d.shape) == 0:
                return
            shm_np = np.ndarray(d.shape, dtype=d.dtype, buffer=shms[shm_idx].buf)
            np.copyto(shm_np, d)

        with self._timer_put_data:
            traverse_lists((data, self.shms), _tra)

        self.shm_index_queue.put(shm_idx)

        return shm_idx
