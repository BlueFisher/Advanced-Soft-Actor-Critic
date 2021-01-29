import tensorflow as tf
import tensorflow_probability as tfp

import algorithm.nn_models as m


class ModelRep(m.ModelBaseSimpleRep):
    def call(self, obs_list):
        obs = obs_list[0][..., :-2]
        
        return obs


class ModelQ(m.ModelQ):
    def __init__(self, state_dim, d_action_dim, c_action_dim, name=None):
        super().__init__(state_dim, d_action_dim, c_action_dim,
                         dense_n=64, dense_depth=2)


class ModelPolicy(m.ModelPolicy):
    def __init__(self, state_dim, d_action_dim, c_action_dim, name=None):
        super().__init__(state_dim, d_action_dim, c_action_dim,
                         dense_n=64, dense_depth=2)
