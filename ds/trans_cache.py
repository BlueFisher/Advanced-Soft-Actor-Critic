import numpy as np

class TransCache:
    _buffer = None

    def add(self, *trans):
        if self._buffer is None:
            self._buffer = list(trans)
        else:
            for i, tran in enumerate(trans):
                self._buffer[i] = np.concatenate([self._buffer[i], tran], axis=0)

    def get_trans_list_and_clear(self):
        trans = [t.tolist() for t in self._buffer]
        self.clear()
        return trans

    def get_trans_and_clear(self):
        trans = self._buffer
        self.clear()
        return trans

    def clear(self):
        self._buffer = None

    @property
    def size(self):
        if self._buffer is None:
            return 0
        else:
            return len(self._buffer[0])
