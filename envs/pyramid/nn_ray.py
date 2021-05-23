import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import algorithm.nn_models as m


class ModelTransition(m.ModelBaseTransition):
    def __init__(self, state_size, d_action_size, c_action_size, use_extra_data):
        super().__init__(state_size, d_action_size, c_action_size, use_extra_data)

        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation=tf.nn.tanh),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(state_size + state_size)
        ])

        self.next_state_tfpd = tfp.layers.DistributionLambda(
            make_distribution_fn=lambda t: tfp.distributions.Normal(t[0], t[1]))

    def init(self):
        if self.use_extra_data:
            self(tf.keras.Input(shape=(self.state_size + 2,)),
                 tf.keras.Input(shape=(self.action_size,)))
        else:
            super().init()

    def call(self, state, action):
        next_state = self.dense(tf.concat([state, action], -1))
        mean, logstd = tf.split(next_state, num_or_size_splits=2, axis=-1)
        next_state_dist = self.next_state_tfpd([mean, tf.clip_by_value(tf.exp(logstd), 0.1, 1.0)])

        return next_state_dist

    def extra_obs(self, obs_list):
        return obs_list[1][..., -2:]


class ModelReward(m.ModelBaseReward):
    def __init__(self, state_size, use_extra_data):
        super().__init__(state_size, use_extra_data)

        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(1)
        ])

    def call(self, state):
        reward = self.dense(state)

        return reward


class ModelObservation(m.ModelBaseObservation):
    def __init__(self, state_size, obs_shapes, use_extra_data):
        super().__init__(state_size, obs_shapes, use_extra_data)

        assert obs_shapes[0] == (44, )
        assert obs_shapes[1] == (6, )

        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation=tf.nn.relu),
            tf.keras.layers.Dense(50, activation=tf.nn.relu),
            tf.keras.layers.Dense(48)  # 44 + 6 - 2
        ])

    def call(self, state):
        obs = self.dense(state)

        return obs[..., :44], obs[..., 44:]

    def get_loss(self, state, obs_list):
        approx_ray_obs, approx_vec_obs = self(state)

        ray_obs, vec_obs = obs_list
        vec_obs = vec_obs[..., :-2]

        mse = tf.losses.MeanSquaredError()

        return 0.5 * mse(approx_ray_obs, ray_obs) + 0.5 * mse(approx_vec_obs, vec_obs)


class ModelRep(m.ModelBaseGRURep):
    def __init__(self, obs_shapes, d_action_size, c_action_size):
        super().__init__(obs_shapes, d_action_size, c_action_size,
                         rnn_units=8)

        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation=tf.nn.relu)
        ])

    def call(self, obs_list, pre_action, rnn_state):
        obs = tf.concat([obs_list[1][..., :-2], pre_action], axis=-1)
        outputs, next_rnn_state = self.gru(obs, initial_state=rnn_state)
        state = self.dense(tf.concat([obs_list[0], outputs], axis=-1))

        return state, next_rnn_state


class ModelQ(m.ModelQ):
    def __init__(self, state_size, d_action_size, c_action_size):
        super().__init__(state_size, d_action_size, c_action_size,
                         c_state_n=128, c_state_depth=1,
                         c_action_n=128, c_action_depth=1,
                         c_dense_n=128, c_dense_depth=3)


class ModelPolicy(m.ModelPolicy):
    def __init__(self, state_size, d_action_size, c_action_size):
        super().__init__(state_size, d_action_size, c_action_size,
                         c_dense_n=128, c_dense_depth=2,
                         mean_n=128, mean_depth=1,
                         logstd_n=128, logstd_depth=1)
