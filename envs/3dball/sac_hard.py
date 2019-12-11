import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


initializer_helper = {
    'kernel_initializer': tf.keras.initializers.TruncatedNormal(0, .1),
    'bias_initializer': tf.keras.initializers.Constant(0.1)
}


class ModelRNN(tf.keras.Model):
    def __init__(self, state_dim):
        super(ModelRNN, self).__init__()
        self.state_dim = state_dim
        self.rnn_units = 64
        self.layer_rnn = tf.keras.layers.GRU(self.rnn_units, return_sequences=True, return_state=True)

        self.get_call_result_tensors()

    def call(self, inputs_s, initial_state):
        outputs, next_state = self.layer_rnn(inputs_s, initial_state=initial_state)

        # outputs = tf.concat([inputs_s, outputs], -1)

        return outputs, next_state

    def get_call_result_tensors(self):
        return self(tf.keras.Input(shape=(None, self.state_dim,), dtype=tf.float32),
                    tf.keras.Input(shape=(self.rnn_units,), dtype=tf.float32))


class ModelPrediction(tf.keras.Model):
    def __init__(self, state_dim, encoded_state_dim, action_dim):
        super(ModelPrediction, self).__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation=tf.nn.relu, **initializer_helper),
            tf.keras.layers.Dense(64, activation=tf.nn.relu, **initializer_helper),
            tf.keras.layers.Dense(64, activation=tf.nn.relu, **initializer_helper),
            tf.keras.layers.Dense(state_dim, **initializer_helper)
        ])

        self(tf.keras.Input(shape=(encoded_state_dim,)), tf.keras.Input(shape=(action_dim,)))

    def call(self, inputs_s, inputs_a):
        return self.seq(tf.concat([inputs_s, inputs_a], -1))


class ModelQ(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(ModelQ, self).__init__()
        self.layer_s = tf.keras.layers.Dense(64, activation=tf.nn.tanh, **initializer_helper)
        self.layer_a = tf.keras.layers.Dense(64, activation=tf.nn.tanh, **initializer_helper)
        self.sequential_model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation=tf.nn.relu, **initializer_helper),
            tf.keras.layers.Dense(64, activation=tf.nn.relu, **initializer_helper),
            tf.keras.layers.Dense(1, **initializer_helper)
        ])

        self(tf.keras.Input(shape=(state_dim,)), tf.keras.Input(shape=(action_dim,)))

    def call(self, inputs_s, inputs_a):
        ls = self.layer_s(inputs_s)
        la = self.layer_a(inputs_a)
        l = tf.concat([ls, la], -1)

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
            tf.keras.layers.Dense(64, activation=tf.nn.relu, **initializer_helper),
            tf.keras.layers.Dense(action_dim, activation=tf.nn.tanh, **initializer_helper)
        ])
        self.sigma_model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation=tf.nn.relu, **initializer_helper),
            tf.keras.layers.Dense(action_dim, activation=tf.nn.sigmoid, **initializer_helper)
        ])

        self.tfpd = tfp.layers.DistributionLambda(make_distribution_fn=lambda t: tfp.distributions.Normal(t[0], t[1]))

        self(tf.keras.Input(shape=(state_dim,)))

    def call(self, inputs_s):
        l = self.common_model(inputs_s)

        mu = self.mu_model(l)

        sigma = self.sigma_model(l)
        sigma = sigma + .1

        return self.tfpd([mu, sigma])
