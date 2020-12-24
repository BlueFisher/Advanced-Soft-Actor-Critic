import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import algorithm.nn_models as m

ModelRep = m.ModelSimpleRep


class ModelQ(m.ModelQ):
    def __init__(self, state_dim, d_action_dim, c_action_dim, name=None):
        super().__init__(state_dim, d_action_dim, c_action_dim,
                         c_dense_n=256, c_dense_depth=3)


class ModelPolicy(m.ModelPolicy):
    def __init__(self, state_dim, d_action_dim, c_action_dim, name=None):
        super().__init__(state_dim, d_action_dim, c_action_dim,
                         c_dense_n=256, c_dense_depth=3,
                         mean_n=256, mean_depth=1,
                         logstd_n=256, logstd_depth=1)
