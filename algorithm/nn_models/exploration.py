import tensorflow as tf


class ModelBaseRND(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

    def init(self):
        return self(tf.keras.Input(shape=(self.state_dim,)),
                    tf.keras.Input(shape=(self.action_dim,)))

    def call(self, obs_list, action):
        raise Exception("ModelBaseRND not implemented")


class ModelBaseForward(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

    def init(self):
        self(tf.keras.Input(shape=(self.state_dim,)),
             tf.keras.Input(shape=(self.action_dim,)))

    def call(self, state, action):
        raise Exception("ModelBaseForward not implemented")


class ModelForward(ModelBaseForward):
    def __init__(self, state_dim, action_dim,
                 dense_n=64, dense_depth=2):
        super().__init__(state_dim, action_dim)

        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(dense_n, tf.nn.relu) for _ in range(dense_depth)
        ] + [tf.keras.layers.Dense(state_dim)])

    def call(self, state, action):
        next_state = self.dense(tf.concat([state, action], -1))

        return next_state
