import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import algorithm.nn_models as m


class ModelForward(m.ModelForward):
    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim,
                         dense_n=256, dense_depth=3)


class ModelTransition(m.ModelBaseTransition):
    def __init__(self, state_dim, d_action_dim, c_action_dim, use_extra_data):
        super().__init__(state_dim, d_action_dim, c_action_dim, use_extra_data)

        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation=tf.nn.tanh),
            tf.keras.layers.Dense(256, activation=tf.nn.relu),
            tf.keras.layers.Dense(256, activation=tf.nn.relu),
            tf.keras.layers.Dense(state_dim + state_dim)
        ])

        self.next_state_tfpd = tfp.layers.DistributionLambda(
            make_distribution_fn=lambda t: tfp.distributions.Normal(t[0], t[1]))

    def call(self, state, action):
        next_state = self.dense(tf.concat([state, action], -1))
        mean, logstd = tf.split(next_state, num_or_size_splits=2, axis=-1)
        next_state_dist = self.next_state_tfpd([mean, tf.exp(logstd)])

        return next_state_dist


class ModelReward(m.ModelBaseReward):
    def __init__(self, state_dim, use_extra_data):
        super().__init__(state_dim, use_extra_data)

        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation=tf.nn.relu),
            tf.keras.layers.Dense(256, activation=tf.nn.relu),
            tf.keras.layers.Dense(256, activation=tf.nn.relu),
            tf.keras.layers.Dense(1)
        ])

    def call(self, state):
        reward = self.dense(state)

        return reward


class ModelObservation(m.ModelBaseObservation):
    def __init__(self, state_dim, obs_dims, use_extra_data):
        super().__init__(state_dim, obs_dims, use_extra_data)

        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation=tf.nn.relu),
            tf.keras.layers.Dense(256, activation=tf.nn.relu),
            tf.keras.layers.Dense(256, activation=tf.nn.relu),
            tf.keras.layers.Dense(obs_dims[0][0] if use_extra_data else obs_dims[0][0] - 3)
        ])

    def call(self, state):
        obs = self.dense(state)

        return obs

    def get_loss(self, state, obs_list):
        approx_obs = self(state)

        obs = obs_list[0]
        if not self.use_extra_data:
            obs = tf.concat([obs[..., :3], obs[..., 6:]], axis=-1)

        return tf.reduce_mean(tf.square(approx_obs - obs))


class ModelRep(m.ModelBaseLSTMRep):
    def __init__(self, obs_dims, d_action_dim, c_action_dim):
        super().__init__(obs_dims, d_action_dim, c_action_dim, rnn_units=64)

        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation=tf.nn.tanh)
        ])

    def call(self, obs_list, pre_action, rnn_state):
        obs = obs_list[0]
        obs = tf.concat([obs[..., :3], obs[..., 6:]], axis=-1)

        rnn_state = tf.split(rnn_state, num_or_size_splits=2, axis=-1)
        outputs, *next_lstm_rnn_state = self.lstm(tf.concat([obs, pre_action], axis=-1),
                                                  initial_state=rnn_state)

        state = self.dense(tf.concat([obs, outputs], axis=-1))

        return state, tf.concat(next_lstm_rnn_state, axis=-1)  # tf.concat(next_rnn_state, axis=-1)w


class ModelQ(m.ModelQ):
    def __init__(self, state_dim, d_action_dim, c_action_dim, name=None):
        super().__init__(state_dim, d_action_dim, c_action_dim,
                         dense_n=256, dense_depth=3)


class ModelPolicy(m.ModelPolicy):
    def __init__(self, state_dim, d_action_dim, c_action_dim, name=None):
        super().__init__(state_dim, d_action_dim, c_action_dim,
                         dense_n=256, dense_depth=3,
                         mean_n=256, mean_depth=1,
                         logstd_n=256, logstd_depth=1)
