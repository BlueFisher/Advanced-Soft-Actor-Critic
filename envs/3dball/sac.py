import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

initializer_helper = {
    'kernel_initializer': tf.keras.initializers.TruncatedNormal(0, .1),
    'bias_initializer': tf.keras.initializers.Constant(0.1)
}


class ModelQ(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(ModelQ, self).__init__()
        self.layer_s = tf.keras.layers.Dense(32, activation=tf.nn.relu, **initializer_helper)
        self.layer_a = tf.keras.layers.Dense(32, activation=tf.nn.relu, **initializer_helper)
        self.sequential_model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation=tf.nn.relu, **initializer_helper),
            tf.keras.layers.Dense(32, activation=tf.nn.relu, **initializer_helper),
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
            tf.keras.layers.Dense(32, activation=tf.nn.relu, **initializer_helper),
            tf.keras.layers.Dense(32, activation=tf.nn.relu, **initializer_helper)
        ])
        self.mu_model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation=tf.nn.relu, **initializer_helper),
            tf.keras.layers.Dense(action_dim, activation=tf.nn.tanh, **initializer_helper)
        ])
        self.sigma_model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation=tf.nn.relu, **initializer_helper),
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
