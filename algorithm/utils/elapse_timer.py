import logging
import time
from typing import Dict, Optional

CLEAR_EACH_LOG = True  # Whether clear last avg time when step % repeat == 0
REPEAT_OVERRIDE = None  # Override repeat argument
MAX_REPEAT_TO_DISABLE_COUNTER = 20


class UnifiedElapsedTimer:
    def __init__(self,
                 logger: Optional[logging.Logger] = None,
                 logger_level: int = logging.DEBUG):
        self._logger = logging.getLogger(logger.name + '.profiler')
        self._logger_level = logger_level
        self._elapsed_timer_dict: Dict[str, ElapsedTimer] = {}

    def __call__(self, log: str, repeat: int = 1, force_report: bool = True):
        if log not in self._elapsed_timer_dict:
            self._elapsed_timer_dict[log] = ElapsedTimer(log,
                                                         logger=self._logger,
                                                         logger_level=self._logger_level,
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
                 log: Optional[str],
                 logger: Optional[logging.Logger] = None,
                 logger_level: int = logging.DEBUG,
                 repeat: int = 1,
                 force_report: bool = True):
        self._log = log
        self._logger = logger
        self._logger_level = logger_level
        self._repeat = repeat
        if REPEAT_OVERRIDE is not None:
            self._repeat = REPEAT_OVERRIDE
        # Whether force report when step % repeat == 0.
        # Otherwise, report if current avg time is highly different from last reported avg time
        self._force_report = force_report

        self._step = 1
        self._last_report_avg_time = -1
        self._last_avg_time = 0

        self._logger_effective = log is not None \
            and logger is not None \
            and logger.getEffectiveLevel() <= self._logger_level

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
            if self._force_report or abs(average_time - self._last_report_avg_time) > 0.1:
                log = f'{self._log}: {average_time:.2f}s'
                if self._logger_level == logging.DEBUG:
                    self._logger.debug(log)
                elif self._logger_level == logging.INFO:
                    self._logger.info(log)
                elif self._logger_level == logging.WARNING:
                    self._logger.warning(log)
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
    """
    This is a counter typically used to record occurrences of UNEXPECTED situations
    """

    def __init__(self,
                 log: str,
                 logger: Optional[logging.Logger] = None,
                 logger_level: int = logging.DEBUG,
                 repeat=1):
        self._log = log
        self._logger = logger
        self._logger_level = logger_level
        self._repeat = repeat

        self._step = 1
        self._counter = 0
        # Record counter that self._counter==self._repeat
        # Meaning the unexpected situation occurred too many times
        self._counter_meets_repeat_times = 0

        self._logger_effective = log is not None \
            and logger is not None \
            and logger.getEffectiveLevel() <= self._logger_level

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        if not self._logger_effective:
            return

        if self._step == self._repeat:
            if self._counter > 0:
                log = f'{self._log}: {self._counter} in {self._repeat}'
                if self._counter < self._repeat / 2:
                    if self._logger_level == logging.DEBUG:
                        self._logger.debug(log)
                    elif self._logger_level == logging.INFO:
                        self._logger.info(log)
                    elif self._logger_level == logging.WARNING:
                        self._logger.warning(log)
                else:
                    self._logger.warning(log)

            if self._counter == self._repeat:
                self._counter_meets_repeat_times += 1
            if self._counter_meets_repeat_times == MAX_REPEAT_TO_DISABLE_COUNTER:
                self._logger.error(f'{self._log} occurred too many times, ElapsedCounter exited')
                self._logger_effective = False

            self._step = 0
            self._counter = 0

        self._step += 1

    def add(self, n=1):
        if not self._logger_effective:
            return

        self._counter += n
