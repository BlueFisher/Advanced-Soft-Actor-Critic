import gc
import logging
import time
import unittest
import weakref

from algorithm.utils.lock import LockLogState, ReadWriteLock, _format_lock_log


class _Payload:
    pass


class TestLockLogging(unittest.TestCase):
    def test_logging_releases_caller_locals_without_cyclic_gc(self):
        logger = logging.getLogger('test.lock_lifetime')
        logger.setLevel(logging.INFO)  # Debug messages are disabled in normal training.
        lock = ReadWriteLock(logger=logger)

        def use_lock(custom_log):
            payload = _Payload()
            reference = weakref.ref(payload)
            with lock.read(custom_log):
                pass
            with lock.write(custom_log):
                pass
            return reference

        was_enabled = gc.isenabled()
        gc.disable()
        try:
            for custom_log in ('sample', None):
                with self.subTest(custom_log=custom_log):
                    reference = use_lock(custom_log)
                    self.assertIsNone(reference(), 'Lock logging retained the caller frame')
        finally:
            if was_enabled:
                gc.enable()
            gc.collect()

    def test_custom_log_and_state_labels(self):
        for state, prefix in ((LockLogState.REQUEST, 'Request.'),
                              (LockLogState.TIMEOUT, 'Timeout '),
                              (LockLogState.ACQUIRED, 'Acquired in '),
                              (LockLogState.RELEASED, 'Released in ')):
            with self.subTest(state=state):
                message = _format_lock_log(state, time.time(), 'sample')
                self.assertTrue(message.startswith(prefix))
                self.assertIn('sample, id:', message)

    def test_implicit_log_identifies_context_manager_caller(self):
        logger = logging.getLogger('test.lock_location')
        lock = ReadWriteLock(logger=logger)
        with self.assertLogs(logger, level='DEBUG') as captured:
            with lock.read():
                pass
        self.assertTrue(all(__file__ in message for message in captured.output))
        self.assertTrue(all('test_implicit_log_identifies_context_manager_caller'
                            in message for message in captured.output))


if __name__ == '__main__':
    unittest.main()
