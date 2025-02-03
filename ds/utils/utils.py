import functools
import threading
import time
from typing import List, Tuple

import grpc
import numpy as np

from ..constants import *


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
                self._logger.warning(f'Retrying {func.__name__}, {retry}...')
                time.sleep(RPC_ERR_RETRY_INTERVAL)
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


def get_episode_shapes_dtypes(max_episode_length: int,
                              obs_shapes: List[Tuple],
                              action_size: int,
                              seq_hidden_state_shape=None):
    """
    ep_indexes
    ep_obses_list
    ep_actions
    ep_rewards
    ep_dones
    ep_mu_probs
    ep_pre_seq_hidden_states
    """
    episode_shapes = [
        (1, max_episode_length),
        [(1, max_episode_length, *o) for o in obs_shapes],
        (1, max_episode_length, action_size),
        (1, max_episode_length),
        (1, max_episode_length),
        (1, max_episode_length, action_size),
        (1, max_episode_length, *seq_hidden_state_shape)
    ]
    episode_dtypes = [
        np.int32,
        [np.float32 for _ in obs_shapes],
        np.float32,
        np.float32,
        bool,
        np.float32,
        np.float32
    ]

    return episode_shapes, episode_dtypes
