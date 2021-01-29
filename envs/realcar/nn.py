import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import algorithm.nn_models as m


class ModelRep(m.ModelBaseSimpleRep):
    def __init__(self, obs_dims):
        super().__init__(obs_dims)

        self.conv_bbox = tf.keras.Sequential([
            # tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, activation=tf.nn.relu),
            # tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, activation=tf.nn.relu),
            tf.keras.layers.Conv2D(filters=16, kernel_size=8, strides=4, activation=tf.nn.relu),
            tf.keras.layers.Conv2D(filters=32, kernel_size=4, strides=2, activation=tf.nn.relu),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(64, activation=tf.nn.relu)
        ])

        self.conv_ray = tf.keras.Sequential([
            # tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, activation=tf.nn.relu),
            # tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, activation=tf.nn.relu),
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
        bbox_vis_obs, ray_vis_obs, vec_obs = obs_list

        bbox_vis_obs = m.through_conv(bbox_vis_obs, self.conv_bbox)
        ray_vis_obs = m.through_conv(ray_vis_obs, self.conv_ray)

        state = self.dense(tf.concat([bbox_vis_obs, ray_vis_obs, vec_obs], -1))

        return state


class ModelQ(m.ModelQ):
    def __init__(self, state_dim, d_action_dim, c_action_dim, name=None):
        super().__init__(state_dim, d_action_dim, c_action_dim,
                         c_state_n=128, c_state_depth=1,
                         c_action_n=128, c_action_depth=1,
                         c_dense_n=128, c_dense_depth=3)


class ModelPolicy(m.ModelPolicy):
    def __init__(self, state_dim, d_action_dim, c_action_dim, name=None):
        super().__init__(state_dim, d_action_dim, c_action_dim,
                         c_dense_n=128, c_dense_depth=2,
                         mean_n=128, mean_depth=1,
                         logstd_n=128, logstd_depth=1)
