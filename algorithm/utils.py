import numpy as np
import tensorflow as tf


def squash_correction_log_prob(dist, x):
    return dist.log_prob(x) - tf.math.log(tf.maximum(1 - tf.square(tf.tanh(x)), 1e-2))


def squash_correction_prob(dist, x):
    return dist.prob(x) / (tf.maximum(1 - tf.square(tf.tanh(x)), 1e-2))


def debug(name, x):
    tf.print(name, tf.reduce_min(x), tf.reduce_mean(x), tf.reduce_max(x))


def debug_grad(grads):
    for grad in grads:
        if grad is not None:
            debug(grad.name, grad)


def debug_grad_com(grads, grads1):
    for i, grad in enumerate(grads):
        debug(grad.name, grad - grads1[i])


def gen_pre_n_actions(n_actions, keep_last_action=False):
    return tf.concat([
        tf.zeros_like(n_actions[:, 0:1, ...]),
        n_actions if keep_last_action else n_actions[:, :-1, ...]
    ], axis=1)


def np_to_tensor(fn):
    def c(*args, **kwargs):
        return fn(*[k if k is not None else tf.zeros((0,)) for k in args],
                  **{k: v if v is not None else tf.zeros((0,)) for k, v in kwargs.items()})

    return c


def scale_h(x, epsilon=0.001):
    return tf.sign(x) * (tf.sqrt(tf.abs(x) + 1) - 1) + epsilon * x


def scale_inverse_h(x, epsilon=0.001):
    t = 1 + 4 * epsilon * (tf.abs(x) + 1 + epsilon)
    return tf.sign(x) * ((tf.sqrt(t) - 1) / (2 * epsilon) - 1)
