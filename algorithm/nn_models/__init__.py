import tensorflow as tf

from .policy import *
from .q import *
from .representation import *
from .predictions import *


def dense(n=64, depth=2, pri=None, post=None):
    l = [tf.keras.layers.Dense(n, tf.nn.relu) for _ in range(depth)]
    if pri:
        l.insert(0, pri)
    if post:
        l.append(post)

    return tf.keras.Sequential(l)
