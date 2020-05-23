import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import algorithm.nn_models as m

ModelRep = m.ModelSimpleRep


class ModelQ(m.ModelDiscreteQ):
    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim,
                         dense_n=64, dense_depth=1)


class ModelPolicy(m.ModelDiscretePolicy):
    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim,
                         dense_n=64, dense_depth=1)
