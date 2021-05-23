import tensorflow as tf
import tensorflow_probability as tfp

import algorithm.nn_models as m


class ModelRep(m.ModelBaseSimpleRep):
    def __init__(self, obs_shapes):
        super().__init__(obs_shapes)

        self.conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=16, kernel_size=8, strides=4, activation=tf.nn.relu),
            tf.keras.layers.Conv2D(filters=32, kernel_size=4, strides=2, activation=tf.nn.relu),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(64, activation=tf.nn.relu)
        ])

        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation=tf.nn.relu),
        ])

    def call(self, obs_list):
        vis_obs, vec_obs = obs_list

        vis_obs = m.through_conv(vis_obs, self.conv)

        state = self.dense(tf.concat([vis_obs, vec_obs], -1))

        return state


class ModelQ(m.ModelQ):
    def __init__(self, state_size, d_action_size, c_action_size):
        super().__init__(state_size, d_action_size, c_action_size,
                         dense_n=128, dense_depth=2)


class ModelPolicy(m.ModelPolicy):
    def __init__(self, state_size, d_action_size, c_action_size):
        super().__init__(state_size, d_action_size, c_action_size,
                         dense_n=128, dense_depth=2)
