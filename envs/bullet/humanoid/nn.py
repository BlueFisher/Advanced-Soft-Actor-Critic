import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import algorithm.nn_models as m

ModelRep = m.ModelSimpleRep


class ModelForward(m.ModelForward):
    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim,
                         dense_n=256, dense_depth=3)


class ModelRND(m.ModelBaseRND):
    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim)

        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation=tf.nn.relu),
            tf.keras.layers.Dense(256, activation=tf.nn.relu),
            tf.keras.layers.Dense(256),
        ])

    def call(self, state, action):
        return self.dense(tf.concat([state, action], axis=-1))


class ModelQ(m.ModelContinuousQ):
    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim,
                         dense_n=256, dense_depth=3)


class ModelPolicy(m.ModelContinuousPolicy):
    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim,
                         dense_n=256, dense_depth=3,
                         mean_n=256, mean_depth=1,
                         logstd_n=256, logstd_depth=1)


# class ModelPolicy(m.ModelBasePolicy):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#         self.dense = tf.keras.Sequential([
#             m.Noisy(512, tf.nn.relu) for _ in range(3)
#         ])
#         self.mean_model = tf.keras.Sequential([
#             m.Noisy(512, tf.nn.relu),
#             m.Noisy(self.action_dim)
#         ])
#         self.logstd_model = tf.keras.Sequential([
#             m.Noisy(512, tf.nn.relu),
#             m.Noisy(self.action_dim)
#         ])
#         self.tfpd = tfp.layers.DistributionLambda(
#             make_distribution_fn=lambda t: tfp.distributions.Normal(t[0], t[1]))

#     def call(self, state):
#         l = self.dense(state)

#         mean = self.mean_model(l)
#         logstd = self.logstd_model(l)

#         return self.tfpd([tf.tanh(mean), tf.clip_by_value(tf.exp(logstd), 0.1, 1.0)])
