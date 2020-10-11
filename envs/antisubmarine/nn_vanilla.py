import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import algorithm.nn_models as m

ModelRep = m.ModelSimpleRep


class ModelQ(m.ModelContinuesQ):
    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim,
                         state_n=128, state_depth=2,
                         action_n=128, action_depth=2,
                         dense_n=128, dense_depth=3)


class ModelPolicy(m.ModelContinuesPolicy):
    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim,
                         dense_n=128, dense_depth=2,
                         mean_n=128, mean_depth=2,
                         logstd_n=128, logstd_depth=2)
