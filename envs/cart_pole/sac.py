import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class ModelRep(tf.keras.Model):
    def __init__(self, obs_dim):
        super(ModelRep, self).__init__()
        self.obs_dim = obs_dim

        self.get_call_result_tensors()

    def call(self, obs):
        return obs

    def get_call_result_tensors(self):
        return self(tf.keras.Input(shape=(self.obs_dim,), dtype=tf.float32))


class ModelQ(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(ModelQ, self).__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation=tf.nn.relu),
            tf.keras.layers.Dense(action_dim)
        ])

        self(tf.keras.Input(shape=(state_dim,)))

    def call(self, state):
        q = self.seq(state)
        return q


class ModelPolicy(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(ModelPolicy, self).__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation=tf.nn.relu),
            tf.keras.layers.Dense(action_dim)
        ])

        self.tfpd = tfp.layers.DistributionLambda(make_distribution_fn=lambda t: tfp.distributions.Categorical(logits=t))

        self(tf.keras.Input(shape=(state_dim,)))

    def call(self, state):
        logits = self.seq(state)

        return self.tfpd(logits)
