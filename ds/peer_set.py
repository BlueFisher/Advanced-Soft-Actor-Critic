import logging
import threading

class PeerSet(object):
    def __init__(self, logger):
        self._peers_lock = threading.RLock()
        self._peers = {}
        self._logger = logger

    def connect(self, peer):
        with self._peers_lock:
            if peer not in self._peers:
                self._peers[peer] = 1
            else:
                self._peers[peer] += 1

        self._logger.info(f'{peer} connected')

    def disconnect(self, peer):
        with self._peers_lock:
            if peer not in self._peers:
                raise RuntimeError(f'Tried to disconnect peer {peer} but it was never connected')
            self._peers[peer] -= 1
            if self._peers[peer] == 0:
                del self._peers[peer]

        self._logger.info(f'{peer} disconnected')

    def peers(self):
        with self._peers_lock:
            return list(self._peers.keys())
