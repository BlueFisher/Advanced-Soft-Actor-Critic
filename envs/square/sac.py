import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from algorithm.common_models import ModelTransition, ModelRNNRep


class ModelTransition(ModelTransition):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(state_dim + state_dim)
        ])

        self.next_state_tfpd = tfp.layers.DistributionLambda(make_distribution_fn=lambda t: tfp.distributions.Normal(t[0], t[1]))

        self(tf.keras.Input(shape=(state_dim,)), tf.keras.Input(shape=(action_dim,)))

    def call(self, state, action):
        next_state = self.seq(tf.concat([state, action], -1))
        mean, logstd = tf.split(next_state, num_or_size_splits=2, axis=-1)
        next_state_dist = self.next_state_tfpd([mean, tf.clip_by_value(tf.exp(logstd), 0.1, 1.)])

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
        assert obs_dims[0] == (44, )
        assert obs_dims[1] == (6, )
        
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation=tf.nn.relu),
            tf.keras.layers.Dense(64, activation=tf.nn.relu),
            tf.keras.layers.Dense(6)
        ])

        self(tf.keras.Input(shape=(state_dim,)))

    def call(self, state):
        obs = self.seq(state[..., 44:])

        return [state[..., :44], obs]

    def get_loss(self, state, obs_list):
        approx_vec_obs = self.seq(state[..., 44:])

        ray_obs, vec_obs = obs_list

        mse = tf.losses.MeanSquaredError()

        return mse(approx_vec_obs, vec_obs)


class ModelRep(ModelRNNRep):
    def __init__(self, obs_dims):
        super().__init__(obs_dims)
        self.rnn_units = 16
        self.layer_rnn = tf.keras.layers.RNN(tf.keras.layers.GRUCell(self.rnn_units),
                                             return_sequences=True,
                                             return_state=True)

        self.get_call_result_tensors()

    def call(self, obs_list, initial_state):
        ray_obs, vec_obs = obs_list
        outputs, next_rnn_state = self.layer_rnn(vec_obs, initial_state=initial_state)

        state = tf.concat([ray_obs, outputs], -1)

        return state, next_rnn_state, outputs


class ModelQ(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.layer_state = tf.keras.layers.Dense(128, activation=tf.nn.relu)
        self.layer_action = tf.keras.layers.Dense(128, activation=tf.nn.relu)

        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(1)
        ])

        self(tf.keras.Input(shape=(state_dim,)), tf.keras.Input(shape=(action_dim,)))

    def call(self, state, action):
        state = self.layer_state(state)
        action = self.layer_action(action)
        l = tf.concat([state, action], -1)

        q = self.seq(l)
        return q


class ModelPolicy(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.common_model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(128, activation=tf.nn.relu)
        ])
        self.mean_model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(action_dim)
        ])
        self.logstd_model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(action_dim)
        ])

        self.tfpd = tfp.layers.DistributionLambda(make_distribution_fn=lambda t: tfp.distributions.Normal(t[0], t[1]))

        self(tf.keras.Input(shape=(state_dim,)))

    def call(self, state):
        l = self.common_model(state)

        mean = self.mean_model(l)
        logstd = self.logstd_model(l)

        return self.tfpd([tf.tanh(mean), tf.clip_by_value(tf.exp(logstd), 0.1, 1.0)])
