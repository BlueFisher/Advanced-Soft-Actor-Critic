import tensorflow as tf

from .policy import *
from .q import *
from .representation import *
from .predictions import *
from .exploration import *


def dense(n=64, depth=2, pri=None, post=None):
    l = [tf.keras.layers.Dense(n, tf.nn.relu) for _ in range(depth)]
    if pri:
        l.insert(0, pri)
    if post:
        l.append(post)

    return tf.keras.Sequential(l)


def through_conv(x, conv):
    if len(x.shape) > 4:
        batch = tf.shape(x)[0]
        x = tf.reshape(x, [-1, *x.shape[2:]])
        x = conv(x)
        x = tf.reshape(x, [batch, -1, x.shape[-1]])
    else:
        x = conv(x)

    return x
