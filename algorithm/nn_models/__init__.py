from .representation import *
from .policy import *
from .q import *
from .predictions import *
from .exploration import *
from .layers import *


# class Noisy(tf.keras.layers.Layer):
#     '''
#     Noisy Net: https://arxiv.org/abs/1706.10295
#     '''

#     def __init__(self, units,
#                  activation=None,
#                  use_bias=True,
#                  is_independent=True,
#                  noise_std=.4, **kwargs):
#         super().__init__(**kwargs)

#         self.units = units
#         self.activation = tf.keras.activations.get(activation)
#         self.use_bias = use_bias
#         self.is_independent = is_independent

#         self.noise_std = float(noise_std)

#     def build(self, input_shape):
#         self.dim = input_shape[-1]

#         self.sigma_w = self.add_weight('sigma_w',
#                                        shape=[self.dim, self.units],
#                                        initializer=tf.random_normal_initializer(0.0, .1),
#                                        dtype=self.dtype,
#                                        trainable=self.trainable)
#         self.mu_w = self.add_weight('mu_w',
#                                     shape=[self.dim, self.units],
#                                     initializer=tf.random_normal_initializer(0.0, .1),
#                                     dtype=self.dtype,
#                                     trainable=self.trainable)

#         if self.use_bias:
#             self.sigma_b = self.add_weight('sigma_b',
#                                            shape=[self.units, ],
#                                            initializer=tf.constant_initializer(self.noise_std / (self.units**0.5)),
#                                            dtype=self.dtype,
#                                            trainable=self.trainable)
#             self.mu_b = self.add_weight('mu_b',
#                                         shape=[self.units, ],
#                                         initializer=tf.constant_initializer(self.noise_std / (self.units**0.5)),
#                                         dtype=self.dtype,
#                                         trainable=self.trainable)

#     def f_for_factor(self, x):
#         return tf.sign(x) * tf.sqrt(tf.abs(x))

#     @property
#     def epsilon_w(self):
#         if self.is_independent:
#             return tf.random.truncated_normal([self.dim, self.units], stddev=self.noise_std)
#         else:
#             return self.f_for_factor(tf.random.truncated_normal([self.dim, 1], stddev=self.noise_std)) \
#                 * self.f_for_factor(tf.random.truncated_normal([1, self.units], stddev=self.noise_std))

#     @property
#     def epsilon_b(self):
#         return tf.random.truncated_normal([self.units, ], stddev=self.noise_std)

#     def noisy_layer(self, inputs):
#         return tf.matmul(inputs, self.noisy_w * self.epsilon_w) + self.noisy_b * self.epsilon_b

#     def call(self, inputs):
#         w = self.mu_w + self.sigma_w * self.epsilon_w
#         y = tf.matmul(inputs, w)

#         if self.use_bias:
#             b = self.mu_b + self.sigma_b * self.epsilon_b
#             y += b

#         if self.activation is not None:
#             return self.activation(y)
