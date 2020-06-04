import tensorflow as tf


class ModelBaseContinuesQ(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

    def init(self):
        self(tf.keras.Input(shape=(self.state_dim,)),
             tf.keras.Input(shape=(self.action_dim,)))

    def call(self, state, action):
        raise Exception("ModelContinuesQ not implemented")


class ModelContinuesQ(ModelBaseContinuesQ):
    def __init__(self, state_dim, action_dim,
                 state_n=64, state_depth=0,
                 action_n=64, action_depth=0,
                 dense_n=64, dense_depth=3):
        super().__init__(state_dim, action_dim)

        self.state_model = tf.keras.Sequential([
            tf.keras.layers.Dense(state_n, tf.nn.relu) for _ in range(state_depth)
        ])
        self.action_model = tf.keras.Sequential([
            tf.keras.layers.Dense(action_n, tf.nn.relu) for _ in range(action_depth)
        ])
        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(dense_n, tf.nn.relu) for _ in range(dense_depth)
        ] + [tf.keras.layers.Dense(1)])

    def call(self, state, action):
        state = self.state_model(state)
        action = self.action_model(action)

        q = self.dense(tf.concat([state, action], -1))

        return q


class ModelBaseDiscreteQ(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

    def init(self):
        self(tf.keras.Input(shape=(self.state_dim,)))

    def call(self, state):
        raise Exception("ModelDiscreteQ not implemented")


class ModelDiscreteQ(ModelBaseDiscreteQ):
    def __init__(self, state_dim, action_dim,
                 dense_n=64, dense_depth=3):
        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(dense_n, tf.nn.relu) for _ in range(dense_depth)
        ] + [tf.keras.layers.Dense(action_dim)])

    def call(self, state):
        q = self.dense(state)

        return q
