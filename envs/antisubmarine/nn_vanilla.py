import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import algorithm.nn_models as m


class ModelRep(m.ModelBaseSimpleRep):
    def call(self, obs_list):
        return obs_list[0][..., :-2]


class ModelQ(m.ModelContinuesQ):
    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim,
                         state_n=256, state_depth=2,
                         action_n=256, action_depth=2,
                         dense_n=256, dense_depth=2)


class ModelPolicy(m.ModelContinuesPolicy):
    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim,
                         dense_n=256, dense_depth=2,
                         mean_n=256, mean_depth=2,
                         logstd_n=256, logstd_depth=2)
