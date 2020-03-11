import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class ModelForward(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(ModelForward, self).__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(state_dim)
        ])

        self(tf.keras.Input(shape=(state_dim,)), tf.keras.Input(shape=(action_dim,)))

    def call(self, state, action):
        next_state = self.seq(tf.concat([state, action], -1))

        return next_state


class ModelTransition(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(ModelTransition, self).__init__()
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


class ModelObservation(tf.keras.Model):
    def __init__(self, state_dim, obs_dim):
        super(ModelObservation, self).__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(obs_dim)
        ])

        self(tf.keras.Input(shape=(state_dim,)))

    def call(self, state):
        obs = self.seq(state)

        return obs


class ModelReward(tf.keras.Model):
    def __init__(self, state_dim):
        super(ModelReward, self).__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(1)
        ])

        self(tf.keras.Input(shape=(state_dim,)))

    def call(self, state):
        reward = self.seq(state)

        return reward


class ModelRNN(tf.keras.Model):
    def __init__(self, obs_dim):
        super(ModelRNN, self).__init__()
        self.obs_dim = obs_dim
        self.rnn_units = 8
        self.layer_rnn = tf.keras.layers.RNN(tf.keras.layers.GRUCell(self.rnn_units),
                                             return_sequences=True,
                                             return_state=True)

        self.get_call_result_tensors()

    def call(self, obs, initial_state):
        outputs, next_rnn_state = self.layer_rnn(obs[:, :, -4:-2], initial_state=initial_state)

        state = tf.concat([outputs, obs], -1)

        return state, next_rnn_state, outputs

    def get_call_result_tensors(self):
        return self(tf.keras.Input(shape=(None, self.obs_dim,), dtype=tf.float32),
                    tf.keras.Input(shape=(self.rnn_units,), dtype=tf.float32))


class ModelQ(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(ModelQ, self).__init__()
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
        super(ModelPolicy, self).__init__()
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
