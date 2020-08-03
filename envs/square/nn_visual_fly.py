import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import algorithm.nn_models as m


class ModelTransition(m.ModelBaseTransition):
    def __init__(self, state_dim, action_dim, use_extra_data):
        super().__init__(state_dim, action_dim, use_extra_data)

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
        next_state_dist = self.next_state_tfpd([mean, tf.exp(logstd)])

        return next_state_dist

    def extra_obs(self, obs_list):
        return obs_list[1][..., -2:]


class ModelReward(m.ModelBaseReward):
    def __init__(self, state_dim, use_extra_data):
        super().__init__(state_dim, use_extra_data)

        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation=tf.nn.relu),
            tf.keras.layers.Dense(1)
        ])

    def call(self, state):
        reward = self.dense(state)

        return reward


class ModelObservation(m.ModelBaseObservation):
    def __init__(self, state_dim, obs_dims, use_extra_data):
        super().__init__(state_dim, obs_dims, use_extra_data)

        assert obs_dims[0] == (30, 30, 3)
        assert obs_dims[1] == (6, )  # Only first 4

        self.vis_seq = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(4 * 4 * 16, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(4, 4, 16)),
            tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=3, strides=1, activation=tf.nn.relu),
            tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=3, strides=2, activation=tf.nn.relu),
            tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=3, strides=2, activation=tf.nn.relu),
            tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=4, strides=1),
        ])
        self.vec_seq = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(4)
        ])

    def call(self, state):
        batch = tf.shape(state)[0]
        t_state = tf.reshape(state, [-1, state.shape[-1]])
        vis_obs = self.vis_seq(t_state)
        vis_obs = tf.reshape(vis_obs, [batch, -1, *vis_obs.shape[1:]])

        vec_obs = self.vec_seq(state)

        return vis_obs, vec_obs

    def get_loss(self, state, obs_list):
        vis_obs, vec_obs = obs_list
        vec_obs = vec_obs[..., :-2]
        approx_vis_obs, approx_vec_obs = self(state)

        mse = tf.keras.losses.MeanSquaredError()

        return 0.5 * mse(approx_vis_obs, vis_obs) + 0.5 * mse(approx_vec_obs, vec_obs)


class ModelRep(m.ModelBaseGRURep):
    def __init__(self, obs_dims, action_dim):
        super().__init__(obs_dims, action_dim, rnn_units=64)

        assert obs_dims[0] == (30, 30, 3)
        assert obs_dims[1] == (6, )  # Only first 4

        self.conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=2, activation=tf.nn.relu),
            tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=2, activation=tf.nn.relu),
            tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=2, activation=tf.nn.relu),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(64, activation=tf.nn.relu)
        ])

        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation=tf.nn.tanh)
        ])

    def call(self, obs_list, pre_action, rnn_state):
        vis_obs, vec_obs = obs_list
        vec_obs = vec_obs[..., :-2]

        obs = tf.concat([vec_obs, pre_action], axis=-1)
        outputs, next_rnn_state = self.gru(obs, initial_state=rnn_state)

        batch = tf.shape(vis_obs)[0]
        vis_obs = tf.reshape(vis_obs, [-1, *vis_obs.shape[2:]])
        vis_obs = self.conv(vis_obs)
        vis_obs = tf.reshape(vis_obs, [batch, -1, vis_obs.shape[-1]])
        state = self.dense(tf.concat([vis_obs, outputs], axis=-1))

        return state, next_rnn_state


class ModelQ(m.ModelContinuesQ):
    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim,
                         dense_n=64, dense_depth=2)


class ModelPolicy(m.ModelContinuesPolicy):
    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim,
                         dense_n=64, dense_depth=2)
