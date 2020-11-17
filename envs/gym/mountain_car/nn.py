import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import algorithm.nn_models as m

ModelRep = m.ModelSimpleRep


class ModelForward(m.ModelForward):
    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim,
                         dense_n=state_dim + action_dim, dense_depth=1)


class ModelRND(m.ModelBaseRND):
    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim)

        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation=tf.nn.relu),
            tf.keras.layers.Dense(32),
        ])

    def call(self, state, action):
        return self.dense(tf.concat([state, action], axis=-1))


class ModelQ(m.ModelQ):
    def __init__(self, state_dim, d_action_dim, c_action_dim):
        super().__init__(state_dim, d_action_dim, c_action_dim,
                         dense_n=64, dense_depth=2)


class ModelPolicy(m.ModelPolicy):
    def __init__(self, state_dim, d_action_dim, c_action_dim):
        super().__init__(state_dim, d_action_dim, c_action_dim,
                         dense_n=64, dense_depth=2)
