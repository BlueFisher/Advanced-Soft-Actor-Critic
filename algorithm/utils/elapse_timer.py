import logging
import time
from typing import Optional


class UnifiedElapsedTimer:
    def __init__(self,
                 logger: Optional[logging.Logger] = None):
        self.logger = logger
        self._elapsed_timer_dict = {}

    def __call__(self, log: str, repeat: int = 1, force_report: bool = True):
        if log not in self._elapsed_timer_dict:
            self._elapsed_timer_dict[log] = elapsed_timer(logger=self.logger,
                                                          custom_log=log,
                                                          repeat=repeat,
                                                          force_report=force_report)

        return self._elapsed_timer_dict[log]


class elapsed_timer:
    def __init__(self,
                 logger: Optional[logging.Logger] = None,
                 custom_log: Optional[str] = None,
                 repeat: int = 1,
                 force_report: bool = False):
        self._logger = logger
        self._custom_log = custom_log
        self._repeat = repeat
        self._force_report = force_report

        self._step = 1
        self._last_report_avg_time = -1
        self._last_avg_time = -1

        self._ignore = False

    def __enter__(self):
        self._start = time.time()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._ignore:
            self._ignore = False
            return

        t = time.time() - self._start

        if self._step == 1:
            self._last_avg_time = t

        average_time = self._last_avg_time + (t - self._last_avg_time) / self._step

        if self._step % self._repeat == 0:
            if self._custom_log is not None and \
                    (self._force_report or abs(average_time - self._last_report_avg_time) > 0.01):
                if self._logger is not None:
                    self._logger.info(f'{self._custom_log}: {average_time:.2f}s')
                else:
                    print(f'{self._custom_log}: {average_time:.2f}s')
            self._last_report_avg_time = average_time

        self._last_avg_time = average_time
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
