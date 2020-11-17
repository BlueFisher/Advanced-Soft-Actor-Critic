import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import algorithm.nn_models as m

ModelRep = m.ModelSimpleRep


class ModelForward(m.ModelForward):
    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim,
                         dense_n=256, dense_depth=3)


class ModelRND(m.ModelBaseRND):
    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim)

        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation=tf.nn.relu),
            tf.keras.layers.Dense(256, activation=tf.nn.relu),
            tf.keras.layers.Dense(256),
        ])

    def call(self, state, action):
        return self.dense(tf.concat([state, action], axis=-1))


class ModelQ(m.ModelQ):
    def __init__(self, state_dim, d_action_dim, c_action_dim):
        super().__init__(state_dim, d_action_dim, c_action_dim,
                         dense_n=256, dense_depth=3)


class ModelPolicy(m.ModelPolicy):
    def __init__(self, state_dim, d_action_dim, c_action_dim):
        super().__init__(state_dim, d_action_dim, c_action_dim,
                         dense_n=256, dense_depth=3,
                         mean_n=256, mean_depth=1,
                         logstd_n=256, logstd_depth=1)
