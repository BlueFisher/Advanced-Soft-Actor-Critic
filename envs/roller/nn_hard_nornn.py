import tensorflow as tf
import tensorflow_probability as tfp

import algorithm.nn_models as m


class ModelRep(m.ModelBaseSimpleRep):
    def call(self, obs_list):
        obs = obs_list[0][..., :-2]
        
        return obs


class ModelQ(m.ModelQ):
    def __init__(self, state_size, d_action_size, c_action_size):
        super().__init__(state_size, d_action_size, c_action_size,
                         dense_n=64, dense_depth=2)


class ModelPolicy(m.ModelPolicy):
    def __init__(self, state_size, d_action_size, c_action_size):
        super().__init__(state_size, d_action_size, c_action_size,
                         dense_n=64, dense_depth=2)
