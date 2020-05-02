import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from algorithm.common_models import ModelTransition, ModelRNNRep


class ModelTransition(ModelTransition):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation=tf.nn.tanh),
            tf.keras.layers.Dense(64, activation=tf.nn.relu),
            tf.keras.layers.Dense(state_dim + state_dim)
        ])

        self.next_state_tfpd = tfp.layers.DistributionLambda(make_distribution_fn=lambda t: tfp.distributions.Normal(t[0], t[1]))

        self(tf.keras.Input(shape=(state_dim + 2,)), tf.keras.Input(shape=(action_dim,)))

    def call(self, state, action):
        next_state = self.seq(tf.concat([state, action], -1))
        mean, logstd = tf.split(next_state, num_or_size_splits=2, axis=-1)
        next_state_dist = self.next_state_tfpd([mean, tf.clip_by_value(tf.exp(logstd), 0.1, 1.)])

        return next_state_dist

    def extra_obs(self, obs_list):
        return obs_list[2][..., -2:]


class ModelReward(tf.keras.Model):
    def __init__(self, state_dim):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation=tf.nn.relu),
            tf.keras.layers.Dense(64, activation=tf.nn.relu),
            tf.keras.layers.Dense(1)
        ])

        self(tf.keras.Input(shape=(state_dim,)))

    def call(self, state):
        reward = self.seq(state)

        return reward


class ModelObservation(tf.keras.Model):
    def __init__(self, state_dim, obs_dims):
        super().__init__()
        assert obs_dims[0] == (28, )
        assert obs_dims[1] == (28, )
        assert obs_dims[2] == (6, )

        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation=tf.nn.relu),
            tf.keras.layers.Dense(50, activation=tf.nn.relu),
            tf.keras.layers.Dense(28 + 28 + 4)
        ])

        self(tf.keras.Input(shape=(state_dim,)))

    def call(self, state):
        obs = self.seq(state)

        return [obs[..., :28], obs[..., 28:28 + 28], obs[..., 28 + 28:]]

    def get_loss(self, state, obs_list):
        approx_obs = self.seq(state)
        obs = tf.concat(obs_list, axis=-1)[..., :-2]

        mse = tf.losses.MeanSquaredError()

        return mse(approx_obs, obs)


class ModelRep(ModelRNNRep):
    def __init__(self, obs_dims):
        super().__init__(obs_dims)
        self.rnn_units = 32
        self.layer_rnn = tf.keras.layers.RNN(tf.keras.layers.GRUCell(self.rnn_units),
                                             return_sequences=True,
                                             return_state=True)
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(64),
        ])
        self.get_call_result_tensors()

    def call(self, obs_list, initial_state):
        obs = tf.concat(obs_list, axis=-1)
        # obs = self.seq1(obs)
        outputs, next_rnn_state = self.layer_rnn(obs_list[2][..., :-2], initial_state=initial_state)
        state = self.seq(tf.concat([obs_list[0], obs_list[1], outputs], axis=-1))

        return state, next_rnn_state, outputs


class ModelQ(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation=tf.nn.relu),
            tf.keras.layers.Dense(64, activation=tf.nn.relu),
            tf.keras.layers.Dense(1)
        ])

        self(tf.keras.Input(shape=(state_dim,)), tf.keras.Input(shape=(action_dim,)))

    def call(self, state, action):
        l = tf.concat([state, action], -1)

        q = self.seq(l)
        return q


class ModelPolicy(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation=tf.nn.relu),
            tf.keras.layers.Dense(64, activation=tf.nn.relu),
            tf.keras.layers.Dense(action_dim + action_dim)
        ])

        self.tfpd = tfp.layers.DistributionLambda(make_distribution_fn=lambda t: tfp.distributions.Normal(t[0], t[1]))

        self(tf.keras.Input(shape=(state_dim,)))

    def call(self, state):
        l = self.seq(state)
        mean, logstd = tf.split(l, num_or_size_splits=2, axis=-1)

        return self.tfpd([mean, tf.clip_by_value(tf.exp(logstd), 0.1, 1.0)])
