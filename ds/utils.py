import functools
import logging
import multiprocessing as mp
import threading
from multiprocessing.shared_memory import SharedMemory
from queue import Empty
from typing import List, Optional, Tuple

import grpc
import numpy as np

from algorithm.utils import elapsed_counter, elapsed_timer

from .constants import *


def traverse_lists(data: Tuple, process):
    if not isinstance(data, tuple):
        data = (data, )

    buffer = []
    for d in zip(*data):
        if isinstance(d[0], list):
            buffer.append(traverse_lists(d, process))
        elif d[0] is None:
            buffer.append(None)
        else:
            buffer.append(process(*d))

    return buffer


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

    def init_from_shapes(self, data_shapes, dtype):
        self.buffer = traverse_lists(data_shapes,
                                     lambda shape: np.empty(shape, dtype=dtype))

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


def rpc_error_inspector(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        retry = RPC_ERR_RETRY
        while retry > 0:
            try:
                return func(self, *args, **kwargs)
            except grpc.RpcError:
                self._logger.error(f'Connection lost in {func.__name__}')
            except Exception as e:
                self._logger.error(f'Error in {func.__name__}')
                self._logger.error(e)

            retry -= 1
            if retry > 0:
                self._logger.warning(f'Retrying {func.__name__}...')
            else:
                self._logger.error(f'{func.__name__} failed')
    return wrapper


class PeerSet(object):
    def __init__(self, logger):
        self._peers_lock = threading.RLock()
        self._peers = dict()
        self._logger = logger

    def connect(self, peer):
        with self._peers_lock:
            if peer not in self._peers:
                self._peers[peer] = dict()
                self._peers[peer]['__conn'] = 1
            else:
                self._peers[peer]['__conn'] += 1

        self._logger.info(f'{peer} connected')

    def disconnect(self, peer):
        with self._peers_lock:
            if peer not in self._peers:
                raise RuntimeError(f'Tried to disconnect peer {peer} but it was never connected')
            self._peers[peer]['__conn'] -= 1
            if self._peers[peer]['__conn'] == 0:
                del self._peers[peer]

        self._logger.info(f'{peer} disconnected')

    def add_info(self, peer, info):
        with self._peers_lock:
            if peer not in self._peers:
                raise RuntimeError(f'Tried to add peer {peer} info but it was never connected')
            for k, v in info.items():
                self._peers[peer][k] = v

    def get_info(self, peer):
        with self._peers_lock:
            if peer in self._peers:
                return self._peers[peer]

    def peers(self):
        with self._peers_lock:
            return self._peers

    def __len__(self):
        with self._peers_lock:
            return len(self._peers)
