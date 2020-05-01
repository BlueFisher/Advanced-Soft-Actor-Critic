import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from algorithm.common_models import ModelSimpleRep, ModelRNNRep


class ModelTransition(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation=tf.nn.tanh),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
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
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(1)
        ])

        self(tf.keras.Input(shape=(state_dim,)))

    def call(self, state):
        reward = self.seq(state)

        return reward


class ModelObservation(tf.keras.Model):
    def __init__(self, state_dim, obs_dims):
        super().__init__()
        assert obs_dims[0] == (30, 30, 3)
        self.conv = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation=tf.nn.relu),
            tf.keras.layers.Dense(64, activation=tf.nn.relu),
            tf.keras.layers.Dense(2 * 2 * 32, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(2, 2, 32)),
            tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=4, strides=2, activation=tf.nn.relu),
            tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=8, strides=4, activation=tf.nn.relu),
            tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=3, strides=1),
        ])

        self(tf.keras.Input(shape=(state_dim,)))

    def call(self, state):
        batch = tf.shape(state)[0]
        state = tf.reshape(state, [-1, state.shape[-1]])
        approx_obs = self.conv(state)
        approx_obs = tf.reshape(approx_obs, [batch, -1, *approx_obs.shape[1:]])

        return [tf.clip_by_value(approx_obs, 0, 1.)]

    def get_loss(self, state, obs_list):
        batch = tf.shape(state)[0]
        state = tf.reshape(state, [-1, state.shape[-1]])
        approx_obs = self.conv(state)
        approx_obs = tf.reshape(approx_obs, [batch, -1, *approx_obs.shape[1:]])

        return tf.reduce_mean(tf.square(approx_obs - obs_list[0]))


class ModelRep(ModelRNNRep):
    def __init__(self, obs_dims):
        super().__init__(obs_dims)
        self.conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, kernel_size=8, strides=4, activation=tf.nn.relu),
            tf.keras.layers.Conv2D(32, kernel_size=4, strides=2, activation=tf.nn.relu),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation=tf.nn.relu),
        ])

        self.rnn_units = 64
        self.layer_rnn = tf.keras.layers.GRU(self.rnn_units, return_sequences=True, return_state=True)
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation=tf.nn.relu),
            tf.keras.layers.Dense(8)
        ])

        self.get_call_result_tensors()

    def call(self, obs_list, initial_state):
        vis_obs = obs_list[0]

        batch = tf.shape(vis_obs)[0]
        vis_obs = tf.reshape(vis_obs, [-1, *vis_obs.shape[2:]])
        vis_obs = self.conv(vis_obs)
        vis_obs = tf.reshape(vis_obs, [batch, -1, vis_obs.shape[-1]])

        outputs, next_rnn_state = self.layer_rnn(vis_obs, initial_state=initial_state)
        state = self.seq(outputs)

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

        return self.tfpd([tf.tanh(mean), tf.clip_by_value(tf.exp(logstd), 0.1, 1.0)])
