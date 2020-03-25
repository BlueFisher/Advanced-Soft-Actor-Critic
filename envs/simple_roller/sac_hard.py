import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from algorithm.common_models import ModelRNNRep


class ModelTransition(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(ModelTransition, self).__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation=tf.nn.relu),
            tf.keras.layers.Dense(64, activation=tf.nn.relu),
            tf.keras.layers.Dense(state_dim + state_dim)
        ])

        self.next_state_tfpd = tfp.layers.DistributionLambda(make_distribution_fn=lambda t: tfp.distributions.Normal(t[0], t[1]))

        self(tf.keras.Input(shape=(state_dim,)), tf.keras.Input(shape=(action_dim,)))

    def call(self, state, action):
        next_state = self.seq(tf.concat([state, action], -1))
        mean, logstd = tf.split(next_state, num_or_size_splits=2, axis=-1)
        next_state_dist = self.next_state_tfpd([mean, tf.exp(logstd)])

        return next_state_dist


class ModelReward(tf.keras.Model):
    def __init__(self, state_dim):
        super(ModelReward, self).__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation=tf.nn.relu),
            tf.keras.layers.Dense(1)
        ])

        self(tf.keras.Input(shape=(state_dim,)))

    def call(self, state):
        reward = self.seq(state)

        return reward


class ModelObservation(tf.keras.Model):
    def __init__(self, state_dim, obs_dims):
        super(ModelObservation, self).__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation=tf.nn.relu),
            tf.keras.layers.Dense(obs_dims[0][0])
        ])

        self(tf.keras.Input(shape=(state_dim,)))

    def call(self, state):
        obs = self.seq(state)

        return [obs]

    def get_loss(self, state, obs_list):
        approx_obs = self.seq(state)

        return tf.reduce_mean(tf.square(approx_obs - obs_list[0]))


class ModelRep(ModelRNNRep):
    def __init__(self, obs_dims):
        super(ModelRep, self).__init__(obs_dims)
        self.rnn_units = 32
        self.layer_rnn = tf.keras.layers.GRU(self.rnn_units, return_sequences=True, return_state=True)
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(32)
        ])

        self.get_call_result_tensors()

    def call(self, obs_list, initial_state):
        obs = obs_list[0]
        outputs, next_rnn_state = self.layer_rnn(obs, initial_state=initial_state)

        # state = tf.concat([obs, outputs], -1)
        state = self.seq(outputs)

        return state, next_rnn_state, outputs


class ModelQ(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(ModelQ, self).__init__()
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
        super(ModelPolicy, self).__init__()
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

        return self.tfpd([tf.tanh(mean), tf.clip_by_value(tf.exp(logstd), 0.1, 1.0)])
