import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import algorithm.nn_models as m


class ModelTransition(m.ModelBaseTransition):
    def __init__(self, state_dim, d_action_dim, c_action_dim, use_extra_data):
        super().__init__(state_dim, d_action_dim, c_action_dim, use_extra_data)

        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation=tf.nn.tanh),
            tf.keras.layers.Dense(64, activation=tf.nn.relu),
            tf.keras.layers.Dense(state_dim + state_dim)
        ])

        self.next_state_tfpd = tfp.layers.DistributionLambda(make_distribution_fn=lambda t: tfp.distributions.Normal(t[0], t[1]))

    def init(self):
        if self.use_extra_data:
            self(tf.keras.Input(shape=(self.state_dim + 2,)), tf.keras.Input(shape=(self.action_dim,)))
        else:
            self(tf.keras.Input(shape=(self.state_dim,)), tf.keras.Input(shape=(self.action_dim,)))

    def call(self, state, action):
        next_state = self.dense(tf.concat([state, action], -1))
        mean, logstd = tf.split(next_state, num_or_size_splits=2, axis=-1)
        next_state_dist = self.next_state_tfpd([mean, tf.clip_by_value(tf.exp(logstd), 0.1, 1.)])

        return next_state_dist

    def extra_obs(self, obs_list):
        return obs_list[2][..., -2:]


class ModelReward(m.ModelBaseReward):
    def __init__(self, state_dim, use_extra_data):
        super().__init__(state_dim, use_extra_data)

        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation=tf.nn.relu),
            tf.keras.layers.Dense(64, activation=tf.nn.relu),
            tf.keras.layers.Dense(1)
        ])

    def call(self, state):
        reward = self.dense(state)

        return reward


class ModelObservation(m.ModelBaseObservation):
    def __init__(self, state_dim, obs_dims, use_extra_data):
        super().__init__(state_dim, obs_dims, use_extra_data)

        assert obs_dims[0] == (28, )
        assert obs_dims[1] == (28, )
        assert obs_dims[2] == (6, )

        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation=tf.nn.relu),
            tf.keras.layers.Dense(50, activation=tf.nn.relu),
            tf.keras.layers.Dense(28 + 28 + 4)
        ])

    def call(self, state):
        obs = self.dense(state)

        return obs[..., :28], obs[..., 28:28 + 28], obs[..., 28 + 28:]

    def get_loss(self, state, obs_list):
        approx_obs = self.dense(state)
        obs = tf.concat(obs_list, axis=-1)[..., :-2]

        mse = tf.losses.MeanSquaredError()

        return mse(approx_obs, obs)


class ModelRep(m.ModelBaseGRURep):
    def __init__(self, obs_dims, action_dim):
        super().__init__(obs_dims, action_dim,
                         rnn_units=32)

        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(64),
        ])

    def call(self, obs_list, pre_action, rnn_state):
        obs = tf.concat(obs_list, axis=-1)

        outputs, next_rnn_state = self.gru(obs_list[2][..., :-2], initial_state=rnn_state)
        state = self.dense(tf.concat([obs_list[0], obs_list[1], outputs], axis=-1))

        return state, next_rnn_state


class ModelQ(m.ModelQ):
    def __init__(self, state_dim, d_action_dim, c_action_dim, name=None):
        super().__init__(state_dim, d_action_dim, c_action_dim,
                         dense_n=64, dense_depth=2)


class ModelPolicy(m.ModelPolicy):
    def __init__(self, state_dim, d_action_dim, c_action_dim, name=None):
        super().__init__(state_dim, d_action_dim, c_action_dim,
                         dense_n=64, dense_depth=2)
