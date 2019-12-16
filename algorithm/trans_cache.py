import numpy as np


class TransCache:
    _buffer = None

    def __init__(self, batch_size):
        self.batch_size = batch_size

    def add(self, *trans):
        if self._buffer is None:
            self._buffer = list(trans)
        else:
            for i, tran in enumerate(trans):
                self._buffer[i] = np.concatenate([self._buffer[i], tran], axis=0)

    def get_batch_trans(self):
        if self.size >= self.batch_size:
            trans = [t[:self.batch_size] for t in self._buffer]
            self._buffer = [t[self.batch_size:] for t in self._buffer]
            return trans
        else:
            return None

    def clear(self):
        self._buffer = None

    @property
    def size(self):
        if self._buffer is None:
            return 0
        else:
            return len(self._buffer[0])
