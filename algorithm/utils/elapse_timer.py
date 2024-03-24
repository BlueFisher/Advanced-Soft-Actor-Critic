import logging
import time
from typing import Dict, Optional

CLEAR_EACH_LOG = True  # Whether clear last avg time when step % repeat == 0
REPEAT_OVERRIDE = None  # Override repeat argument


class UnifiedElapsedTimer:
    def __init__(self,
                 logger: Optional[logging.Logger] = None):
        self.logger = logging.getLogger(logger.name + '.profiler')
        self._elapsed_timer_dict: Dict[str, ElapsedTimer] = {}

    def __call__(self, log: str, repeat: int = 1, force_report: bool = True):
        if log not in self._elapsed_timer_dict:
            self._elapsed_timer_dict[log] = ElapsedTimer(log,
                                                         logger=self.logger,
                                                         repeat=repeat,
                                                         force_report=force_report)

        return self._elapsed_timer_dict[log]


def unified_elapsed_timer(log: str, repeat: int = 1, force_report: bool = True,
                          profiler: str = '_profiler'):
    def profile(func):
        def wrapper(self, *args, **kwargs):
            with getattr(self, profiler)(log, repeat, force_report) as p:
                results = func(self, p, *args, **kwargs)
            return results
        return wrapper
    return profile


class ElapsedTimer:
    def __init__(self,
                 log: str,
                 logger: Optional[logging.Logger] = None,
                 repeat: int = 1,
                 force_report: bool = True):
        self._log = log
        self._logger = logger
        self._repeat = repeat
        if REPEAT_OVERRIDE is not None:
            self._repeat = REPEAT_OVERRIDE
        # Whether force report when step % repeat == 0.
        # Otherwise, report if current avg time is highly different from last reported avg time
        self._force_report = force_report

        self._step = 1
        self._last_report_avg_time = -1
        self._last_avg_time = 0

        self._logger_effective = logger.getEffectiveLevel() == logging.DEBUG

        self._ignore = False

    def __enter__(self):
        self._start = time.time()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self._logger_effective:
            return

        if self._ignore:
            self._ignore = False
            return

        t = time.time() - self._start

        average_time = self._last_avg_time + (t - self._last_avg_time) / self._step

        if self._step % self._repeat == 0:
            if self._force_report or abs(average_time - self._last_report_avg_time) > 0.01:
                if self._logger is not None:
                    self._logger.debug(f'{self._log}: {average_time:.2f}s')
                else:
                    print(f'{self._log}: {average_time:.2f}s')
            self._last_report_avg_time = average_time

        if CLEAR_EACH_LOG and self._step % self._repeat == 0:
            self._last_avg_time = 0
            self._step = 1
        else:
            self._last_avg_time = average_time
            self._step += 1

    def ignore(self):
        self._ignore = True


class ElapsedCounter:
    def __init__(self,
                 log: str,
                 logger: Optional[logging.Logger] = None,
                 repeat=1):
        self._log = log
        self._logger = logger
        self._repeat = repeat

        self._step = 1
        self._counter = 0

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        if self._step == self._repeat:
            if self._logger is not None and self._counter > 0:
                self._logger.debug(f'{self._log}: {self._counter} in {self._repeat}')
            self._step = 0
            self._counter = 0

        self._step += 1

    def add(self, n=1):
        self._counter += n
