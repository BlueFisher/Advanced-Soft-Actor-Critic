import tensorflow as tf


class ModelBaseQ(tf.keras.Model):
    def __init__(self, state_dim, d_action_dim, c_action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.d_action_dim = d_action_dim
        self.c_action_dim = c_action_dim

    def init(self):
        self(tf.keras.Input(shape=(self.state_dim,)),
             tf.keras.Input(shape=(self.c_action_dim,)))

    def call(self, state, action):
        raise Exception("ModelQ not implemented")


class ModelQ(ModelBaseQ):
    def __init__(self, state_dim, d_action_dim, c_action_dim,
                 state_n=64, state_depth=0,
                 action_n=64, action_depth=0,
                 dense_n=64, dense_depth=3):
        super().__init__(state_dim, d_action_dim, c_action_dim)

        if self.d_action_dim:
            self.d_dense = tf.keras.Sequential([
                tf.keras.layers.Dense(dense_n, tf.nn.relu) for _ in range(dense_depth)
            ] + [tf.keras.layers.Dense(d_action_dim)])

        if self.c_action_dim:
            self.c_state_model = tf.keras.Sequential([
                tf.keras.layers.Dense(state_n, tf.nn.relu) for _ in range(state_depth)
            ])
            self.c_action_model = tf.keras.Sequential([
                tf.keras.layers.Dense(action_n, tf.nn.relu) for _ in range(action_depth)
            ])
            self.c_dense = tf.keras.Sequential([
                tf.keras.layers.Dense(dense_n, tf.nn.relu) for _ in range(dense_depth)
            ] + [tf.keras.layers.Dense(1)])

    def call(self, state, c_action):
        if self.d_action_dim:
            d_q = self.d_dense(state)
        else:
            d_q = tf.zeros((0,))

        if self.c_action_dim:
            c_state = self.c_state_model(state)
            c_action = self.c_action_model(c_action)

            c_q = self.c_dense(tf.concat([c_state, c_action], -1))
        else:
            c_q = tf.zeros((0,))

        return d_q, c_q
