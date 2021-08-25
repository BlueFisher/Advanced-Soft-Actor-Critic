import logging
import time
from typing import Optional


class elapsed_timer:
    def __init__(self, logger: Optional[logging.Logger] = None,
                 custom_log: Optional[str] = None,
                 repeat: int = 1):
        self._logger = logger
        self._custom_log = custom_log
        self._repeat = repeat

        self._step = 1
        self._sum_time = 0
        self._last_avg_time = -1

        self._ignore = False

    def __enter__(self):
        self._start = time.time()

    def __exit__(self, exc_type, exc_value, traceback):
        if self._ignore:
            self._ignore = False
            return

        self._sum_time += time.time() - self._start

        if self._step == self._repeat:
            avg_time = self._sum_time / self._repeat
            if self._logger is not None and self._custom_log is not None \
                    and abs(avg_time - self._last_avg_time) > 0.01:
                self._logger.info(f'{self._custom_log}: {avg_time:.2f}s')
            self._last_avg_time = avg_time
            self._step = 0
            self._sum_time = 0

        self._step += 1

    def ignore(self):
        self._ignore = True


class elapsed_counter:
    def __init__(self, logger: Optional[logging.Logger] = None, custom_log: Optional[str] = None, repeat=1):
        self._logger = logger
        self._custom_log = custom_log
        self._repeat = repeat

        self._step = 1
        self._counter = 0

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        if self._step == self._repeat:
            if self._logger is not None and self._custom_log is not None \
                    and self._counter > 0:
                self._logger.info(f'{self._custom_log}: {self._counter} in {self._repeat}')
            self._step = 0
            self._counter = 0

        self._step += 1

    def add(self, n=1):
        self._counter += n
