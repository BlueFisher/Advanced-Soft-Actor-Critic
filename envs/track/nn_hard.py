import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import algorithm.nn_models as m


class ModelRep(m.ModelBaseSimpleRep):
    def __init__(self, obs_shapes):
        super().__init__(obs_shapes)

    def call(self, obs_list):
        return tf.concat(obs_list, axis=-1)


class ModelQ(m.ModelQ):
    def __init__(self, state_size, d_action_size, c_action_size):
        super().__init__(state_size, d_action_size, c_action_size,
                         c_state_n=128, c_state_depth=1,
                         c_action_n=128, c_action_depth=1,
                         c_dense_n=128, c_dense_depth=3)


class ModelPolicy(m.ModelPolicy):
    def __init__(self, state_size, d_action_size, c_action_size):
        super().__init__(state_size, d_action_size, c_action_size,
                         c_dense_n=128, c_dense_depth=2,
                         mean_n=128, mean_depth=1,
                         logstd_n=128, logstd_depth=1)
