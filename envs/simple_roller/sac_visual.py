import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from algorithm.common_models import ModelSimpleRep, ModelRNNRep


class ModelTransition(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation=tf.nn.tanh),
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
        super().__init__()
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
        super().__init__()
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


# # No RNN example
# class ModelRep(ModelSimpleRep):
#     def __init__(self, obs_dims):
#         super().__init__(obs_dims)
#         self.conv = tf.keras.Sequential([
#             tf.keras.layers.Conv2D(3, 3, activation=tf.nn.relu),
#             tf.keras.layers.MaxPooling2D(),
#             tf.keras.layers.Conv2D(3, 3, activation=tf.nn.relu),
#             tf.keras.layers.MaxPooling2D(),
#             tf.keras.layers.Conv2D(3, 3, activation=tf.nn.relu),
#             tf.keras.layers.MaxPooling2D(),
#             tf.keras.layers.Flatten(),
#             tf.keras.layers.Dense(64, activation=tf.nn.tanh)
#         ])

#         self.get_call_result_tensors()

#     def call(self, obs_list):
#         vis_obs = obs_list[0]
#         batch = tf.shape(vis_obs)[0]
#         if len(vis_obs.shape) == 5:
#             vis_obs = tf.reshape(vis_obs, [-1, *vis_obs.shape[2:]])
#             vec_state = self.conv(vis_obs)
#             vec_state = tf.reshape(vec_state, [batch, -1, vec_state.shape[-1]])
#         else:
#             vec_state = self.conv(vis_obs)

#         return vec_state

class ModelRep(ModelRNNRep):
    def __init__(self, obs_dims):
        super().__init__(obs_dims)
        self.conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(3, 3, activation=tf.nn.relu),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(3, 3, activation=tf.nn.relu),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(3, 3, activation=tf.nn.relu),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64)
        ])

        self.rnn_units = 64
        self.layer_rnn = tf.keras.layers.GRU(self.rnn_units, return_sequences=True, return_state=True)
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation=tf.nn.tanh)
        ])

        self.get_call_result_tensors()

    def call(self, obs_list, initial_state):
        vis_obs = obs_list[0]
        batch = tf.shape(vis_obs)[0]
        vis_obs = tf.reshape(vis_obs, [-1, *vis_obs.shape[2:]])
        vec_obs = self.conv(vis_obs)
        vec_obs = tf.reshape(vec_obs, [batch, -1, vec_obs.shape[-1]])

        outputs, next_rnn_state = self.layer_rnn(vec_obs, initial_state=initial_state)

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

        return self.tfpd([mean, tf.clip_by_value(tf.exp(logstd), 0.1, 1.0)])
