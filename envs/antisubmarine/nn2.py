import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import algorithm.nn_models as m


class ModelTransition(m.ModelBaseTransition):
    def __init__(self, state_dim, d_action_dim, c_action_dim, use_extra_data):
        super().__init__(state_dim, d_action_dim, c_action_dim, use_extra_data)

        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(state_dim + state_dim)
        ])

        self.next_state_tfpd = tfp.layers.DistributionLambda(
            make_distribution_fn=lambda t: tfp.distributions.Normal(t[0], t[1]))

    def init(self):
        if self.use_extra_data:
            self(tf.keras.Input(shape=(self.state_dim + 2,)),
                 tf.keras.Input(shape=(self.action_dim,)))
        else:
            self(tf.keras.Input(shape=(self.state_dim,)),
                 tf.keras.Input(shape=(self.action_dim,)))

    def call(self, state, action):
        next_state = self.dense(tf.concat([state, action], -1))
        mean, logstd = tf.split(next_state, num_or_size_splits=2, axis=-1)
        next_state_dist = self.next_state_tfpd([mean, tf.clip_by_value(tf.exp(logstd), 0.1, 1.0)])

        return next_state_dist

    def extra_obs(self, obs_list):
        return obs_list[0][..., -2:]


class ModelReward(m.ModelBaseReward):
    def __init__(self, state_dim, use_extra_data):
        super().__init__(state_dim, use_extra_data)

        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(1)
        ])

    def call(self, state):
        reward = self.dense(state)

        return reward


class ModelObservation(m.ModelBaseObservation):
    def __init__(self, state_dim, obs_dims, use_extra_data):
        super().__init__(state_dim, obs_dims, use_extra_data)

        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(obs_dims[0][0] if self.use_extra_data else obs_dims[0][0] - 2)
        ])

    def call(self, state):
        obs = self.dense(state)

        return obs

    def get_loss(self, state, obs_list):
        approx_obs = self(state)

        obs = obs_list[0]
        if not self.use_extra_data:
            obs = obs[..., :-2]

        return tf.reduce_mean(tf.square(approx_obs - obs))


class ModelRep(m.ModelBaseGRURep):
    def __init__(self, obs_dims, d_action_dim, c_action_dim):
        super().__init__(obs_dims, d_action_dim, c_action_dim, rnn_units=64)

        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation=tf.nn.relu)
        ])

    def call(self, obs_list, pre_action, rnn_state):
        # rnn_obs = tf.concat([obs_list[0][..., :5], pre_action], axis=-1)
        # buoy_obs = obs_list[0][..., 5:]

        obs = obs_list[0][..., :-2]
        obs = tf.concat([obs, pre_action], axis=-1)

        outputs, next_rnn_state = self.gru(obs, initial_state=rnn_state)

        # state = tf.concat([outputs, buoy_obs], axis=-1)
        state = self.dense(outputs)

        return state, next_rnn_state


class ModelQ(m.ModelQ):
    def __init__(self, state_dim, d_action_dim, c_action_dim, name=None):
        super().__init__(state_dim, d_action_dim, c_action_dim, name,
                         d_dense_n=128, d_dense_depth=3,
                         c_state_n=128, c_state_depth=1,
                         c_action_n=128, c_action_depth=1,
                         c_dense_n=128, c_dense_depth=3)


class ModelPolicy(m.ModelPolicy):
    def __init__(self, state_dim, d_action_dim, c_action_dim, name=None):
        super().__init__(state_dim, d_action_dim, c_action_dim, name,
                         d_dense_n=128, d_dense_depth=3,
                         c_dense_n=128, c_dense_depth=3,
                         mean_n=128, mean_depth=1,
                         logstd_n=128, logstd_depth=1)
