import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from algorithm.common_models import ModelTransition, ModelSimpleRep, ModelRNNRep


class ModelTransition(ModelTransition):
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
        self.vis_seq = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(2 * 2 * 32, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(2, 2, 32)),
            tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=4, strides=2, activation=tf.nn.relu),
            tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=8, strides=4, activation=tf.nn.relu),
            tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=3, strides=1, activation=tf.nn.relu),
        ])

        self(tf.keras.Input(shape=(state_dim,)))

    def call(self, state):
        batch = tf.shape(state)[0]
        t_state = tf.reshape(state, [-1, state.shape[-1]])
        vis_obs = self.vis_seq(t_state)
        vis_obs = tf.reshape(vis_obs, [batch, -1, *vis_obs.shape[1:]])

        return vis_obs

    def get_loss(self, state, obs_list):
        approx_vis_obs = self(state)
        vis_obs = obs_list[0]

        mse = tf.keras.losses.MeanSquaredError()

        return mse(approx_vis_obs, vis_obs)


class ModelRep(ModelRNNRep):
    def __init__(self, obs_dims, action_dim):
        super().__init__(obs_dims, action_dim)
        self.conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, activation=tf.nn.relu),
            tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, activation=tf.nn.relu),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(64, activation=tf.nn.relu)
        ])

        self.rnn_units = 64
        self.layer_rnn = tf.keras.layers.RNN(tf.keras.layers.GRUCell(self.rnn_units), return_sequences=True, return_state=True)
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation=tf.nn.relu),
        ])

        self.get_call_result_tensors()

    def call(self, obs_list, pre_action, rnn_state):
        vis_obs = obs_list[0]

        batch = tf.shape(vis_obs)[0]
        vis_obs = tf.reshape(vis_obs, [-1, *vis_obs.shape[2:]])
        vis_obs = self.conv(vis_obs)
        vis_obs = tf.reshape(vis_obs, [batch, -1, vis_obs.shape[-1]])

        outputs, next_rnn_state = self.layer_rnn(vis_obs, initial_state=rnn_state)
        state = self.seq(outputs)

        return state, next_rnn_state, outputs


class ModelQ(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
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
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(action_dim + action_dim)
        ])

        self.tfpd = tfp.layers.DistributionLambda(make_distribution_fn=lambda t: tfp.distributions.Normal(t[0], t[1]))

        self(tf.keras.Input(shape=(state_dim,)))

    def call(self, state):
        l = self.seq(state)
        mean, logstd = tf.split(l, num_or_size_splits=2, axis=-1)

        return self.tfpd([tf.tanh(mean), tf.clip_by_value(tf.exp(logstd), 0.1, 1.0)])
