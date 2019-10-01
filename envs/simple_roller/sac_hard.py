import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


initializer_helper = {
    'kernel_initializer': tf.keras.initializers.TruncatedNormal(0, .1),
    'bias_initializer': tf.keras.initializers.Constant(0.1)
}


class ModelLSTM(tf.keras.Model):
    def __init__(self, state_dim):
        super(ModelLSTM, self).__init__()
        self.state_dim = state_dim
        self.lstm_units = 2
        self.layer_lstm = tf.keras.layers.LSTM(self.lstm_units, return_sequences=True, return_state=True)

        self.get_call_result_tensors()

    def call(self, inputs_s, initial_state_h, initial_state_c):
        outputs, state_h, state_c = self.layer_lstm(inputs_s, initial_state=[initial_state_h, initial_state_c])

        encoded_s = tf.concat([inputs_s, outputs], -1)
        return encoded_s, state_h, state_c

    def get_call_result_tensors(self):
        return self(tf.keras.Input(shape=(None, self.state_dim,), dtype=tf.float32),
                    tf.keras.Input(shape=(self.lstm_units,), dtype=tf.float32),
                    tf.keras.Input(shape=(self.lstm_units,), dtype=tf.float32))


class ModelQ(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(ModelQ, self).__init__()
        self.layer_s = tf.keras.layers.Dense(64, activation=tf.nn.relu, **initializer_helper)
        self.layer_a = tf.keras.layers.Dense(64, activation=tf.nn.relu, **initializer_helper)
        self.layer_1 = tf.keras.layers.Dense(64, activation=tf.nn.relu, **initializer_helper)
        self.layer_2 = tf.keras.layers.Dense(64, activation=tf.nn.relu, **initializer_helper)
        self.layer_q = tf.keras.layers.Dense(1, **initializer_helper)

        self(tf.keras.Input(shape=(state_dim,)), tf.keras.Input(shape=(action_dim,)))

    def call(self, inputs_s, inputs_a):
        ls = self.layer_s(inputs_s)
        la = self.layer_a(inputs_a)
        l = tf.concat([ls, la], -1)

        l = self.layer_1(l)
        l = self.layer_2(l)
        q = self.layer_q(l)
        return q


class ModelPolicy(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(ModelPolicy, self).__init__()
        self.common_layer_1 = tf.keras.layers.Dense(64, activation=tf.nn.relu, **initializer_helper)
        self.common_layer_2 = tf.keras.layers.Dense(64, activation=tf.nn.relu, **initializer_helper)
        self.mu_layer_1 = tf.keras.layers.Dense(64, activation=tf.nn.relu, **initializer_helper)
        self.mu_layer_2 = tf.keras.layers.Dense(action_dim, activation=tf.nn.tanh, **initializer_helper)
        self.sigma_layer_1 = tf.keras.layers.Dense(64, activation=tf.nn.relu, **initializer_helper)
        self.sigma_layer_2 = tf.keras.layers.Dense(action_dim, activation=tf.nn.sigmoid, **initializer_helper)
        self.tfpd = tfp.layers.DistributionLambda(make_distribution_fn=lambda t: tfp.distributions.Normal(t[0], t[1]))

        self(tf.keras.Input(shape=(state_dim,)))

    def call(self, inputs_s):
        l = self.common_layer_1(inputs_s)
        l = self.common_layer_2(l)

        mu = self.mu_layer_1(l)
        mu = self.mu_layer_2(mu)

        sigma = self.sigma_layer_1(l)
        sigma = self.sigma_layer_2(sigma)
        sigma = sigma + .1

        return self.tfpd([mu, sigma])
