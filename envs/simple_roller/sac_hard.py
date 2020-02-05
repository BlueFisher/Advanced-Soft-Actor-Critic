import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


initializer_helper = {
    'kernel_initializer': tf.keras.initializers.TruncatedNormal(0, .1),
    'bias_initializer': tf.keras.initializers.Constant(0.1)
}


class ModelRNN(tf.keras.Model):
    def __init__(self, obs_dim):
        super(ModelRNN, self).__init__()
        self.obs_dim = obs_dim
        self.rnn_units = 32
        self.layer_rnn = tf.keras.layers.GRU(self.rnn_units, return_sequences=True, return_state=True)

        self.get_call_result_tensors()

    def call(self, obs, initial_state):
        outputs, next_rnn_state = self.layer_rnn(obs, initial_state=initial_state)

        state = tf.concat([obs, outputs], -1)
        return state, next_rnn_state, outputs

    def get_call_result_tensors(self):
        return self(tf.keras.Input(shape=(None, self.obs_dim,), dtype=tf.float32),
                    tf.keras.Input(shape=(self.rnn_units,), dtype=tf.float32))


class ModelPrediction(tf.keras.Model):
    def __init__(self, obs_dim, state_dim, action_dim):
        super(ModelPrediction, self).__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation=tf.nn.relu, **initializer_helper),
            tf.keras.layers.Dense(128, activation=tf.nn.relu, **initializer_helper),
            tf.keras.layers.Dense(128, activation=tf.nn.relu, **initializer_helper),
            tf.keras.layers.Dense(action_dim, **initializer_helper)
        ])

        self(tf.keras.Input(shape=(state_dim,)), tf.keras.Input(shape=(obs_dim,)))

    def call(self, state, obs_):
        return self.seq(tf.concat([state, obs_], -1))


class ModelQ(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(ModelQ, self).__init__()
        self.sequential_model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation=tf.nn.relu, **initializer_helper),
            tf.keras.layers.Dense(64, activation=tf.nn.relu, **initializer_helper),
            tf.keras.layers.Dense(1, **initializer_helper)
        ])

        self(tf.keras.Input(shape=(state_dim,)), tf.keras.Input(shape=(action_dim,)))

    def call(self, state, action):
        l = tf.concat([state, action], -1)

        q = self.sequential_model(l)
        return q


class ModelPolicy(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(ModelPolicy, self).__init__()
        self.common_model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation=tf.nn.relu, **initializer_helper),
            tf.keras.layers.Dense(64, activation=tf.nn.relu, **initializer_helper)
        ])
        self.mu_model = tf.keras.Sequential([
            tf.keras.layers.Dense(action_dim, activation=tf.nn.tanh, **initializer_helper)
        ])
        self.sigma_model = tf.keras.Sequential([
            tf.keras.layers.Dense(action_dim, activation=tf.nn.sigmoid, **initializer_helper)
        ])

        self.tfpd = tfp.layers.DistributionLambda(make_distribution_fn=lambda t: tfp.distributions.Normal(t[0], t[1]))

        self(tf.keras.Input(shape=(state_dim,)))

    def call(self, state):
        l = self.common_model(state)

        mu = self.mu_model(l)

        sigma = self.sigma_model(l)
        sigma = sigma + .1

        return self.tfpd([mu, sigma])
