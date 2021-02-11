import threading
import time

import tensorflow as tf

from .utils import np_to_tensor


class CPU2GPUBuffer:
    def __init__(self, get_cpu_data, input_signature, can_return_None=False):
        self._get_cpu_data = get_cpu_data
        self._can_return_None = can_return_None

        self._cpu_data_buffer = None
        self._gpu_data_buffer = None
        self._None_buffer = False

        self._cpu2gpu = np_to_tensor(tf.function(self._cpu2gpu, input_signature=input_signature))

        self._lock = threading.Condition()
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    def _run(self):
        while True:
            with self._lock:
                self._lock.wait_for(lambda: self._gpu_data_buffer is None
                                    and not (self._can_return_None and self._None_buffer))
            
            data = self._get_cpu_data()
            if data is None:
                if self._can_return_None:
                    self._None_buffer = True
                    with self._lock:
                        self._lock.notify()
                else:
                    continue
            else:
                self._cpu_data_buffer, cpu_data_for_gpu = data
                if not isinstance(cpu_data_for_gpu, list) \
                        or not isinstance(cpu_data_for_gpu, tuple):
                    cpu_data_for_gpu = (cpu_data_for_gpu)
                self._gpu_data_buffer = self._cpu2gpu(*cpu_data_for_gpu)
                self._None_buffer = False
                with self._lock:
                    self._lock.notify()

    @tf.function
    def _cpu2gpu(self, *data):
        return data

    def get_data(self):
        with self._lock:
            self._lock.wait_for(lambda: self._gpu_data_buffer is not None
                                or (self._can_return_None and self._None_buffer))

        result = None
        if self._gpu_data_buffer is not None:
            result = (self._cpu_data_buffer, self._gpu_data_buffer)

        self._cpu_data_buffer = None
        self._gpu_data_buffer = None
        self._None_buffer = False
        with self._lock:
            self._lock.notify()

        return result
