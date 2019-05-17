import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from algorithm.sac_base import SAC_Base

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

initializer_helper = {
    'kernel_initializer': tf.truncated_normal_initializer(0, .1),
    'bias_initializer': tf.constant_initializer(.1)
}


class SAC(SAC_Base):
    def _build_q_net(self, s_input, a_input, scope, trainable=True, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            ls = tf.layers.dense(
                s_input, 64, activation=tf.nn.relu,
                trainable=trainable, **initializer_helper
            )
            la = tf.layers.dense(
                a_input, 64, activation=tf.nn.relu,
                trainable=trainable, **initializer_helper
            )
            l = tf.concat([ls, la], 1)
            l = tf.layers.dense(l, 64, activation=tf.nn.relu, trainable=trainable, **initializer_helper)
            l = tf.layers.dense(l, 64, activation=tf.nn.relu, trainable=trainable, **initializer_helper)
            q = tf.layers.dense(l, 1, **initializer_helper, trainable=trainable)

            variables = tf.get_variable_scope().global_variables()

        return q, variables

    def _build_policy_net(self, s_inputs, scope, trainable=True, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            l = tf.layers.dense(s_inputs, 64, tf.nn.relu, **initializer_helper, trainable=trainable)
            l = tf.layers.dense(l, 64, tf.nn.relu, **initializer_helper, trainable=trainable)

            mu = tf.layers.dense(l, 64, tf.nn.relu, **initializer_helper, trainable=trainable)
            mu = tf.layers.dense(mu, self.a_dim, tf.nn.tanh, **initializer_helper, trainable=trainable)

            sigma = tf.layers.dense(l, 64, tf.nn.relu, **initializer_helper, trainable=trainable)
            sigma = tf.layers.dense(sigma, self.a_dim, tf.nn.sigmoid, **initializer_helper, trainable=trainable)
            sigma = sigma + .1

            policy = tfp.distributions.Normal(loc=mu, scale=sigma)
            action = policy.sample()

            variables = tf.get_variable_scope().global_variables()

        return policy, action, variables

    def choose_action(self, s):
        assert len(s.shape) == 2

        a = self.sess.run(self.action_sampled, {
            self.pl_s: s,
        })

        return np.clip(a, -1, 1)
