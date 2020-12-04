import tensorflow as tf


class ModelBaseQ(tf.keras.Model):
    def __init__(self, state_dim, d_action_dim, c_action_dim, name=None):
        super().__init__(name=name)
        self.state_dim = state_dim
        self.d_action_dim = d_action_dim
        self.c_action_dim = c_action_dim

    def init(self):
        self(tf.keras.Input(shape=(self.state_dim,)),
             tf.keras.Input(shape=(self.c_action_dim,)))

    def call(self, state, action):
        raise Exception("ModelQ not implemented")


class ModelQ(ModelBaseQ):
    def __init__(self, state_dim, d_action_dim, c_action_dim, name=None,
                 dense_n=64, dense_depth=0,
                 d_dense_n=64, d_dense_depth=3,
                 c_state_n=64, c_state_depth=0,
                 c_action_n=64, c_action_depth=0,
                 c_dense_n=64, c_dense_depth=3):
        super().__init__(state_dim, d_action_dim, c_action_dim, name)

        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(dense_n, tf.nn.relu) for _ in range(dense_depth)
        ], name='shared_seq')

        if self.d_action_dim:
            self.d_dense = tf.keras.Sequential([
                tf.keras.layers.Dense(d_dense_n, tf.nn.relu) for _ in range(d_dense_depth)
            ] + [tf.keras.layers.Dense(d_action_dim, name='d_q_output_dense')], name='d_seq')

        if self.c_action_dim:
            self.c_state_model = tf.keras.Sequential([
                tf.keras.layers.Dense(c_state_n, tf.nn.relu) for _ in range(c_state_depth)
            ], name='c_state_seq')
            self.c_action_model = tf.keras.Sequential([
                tf.keras.layers.Dense(c_action_n, tf.nn.relu) for _ in range(c_action_depth)
            ], name='c_action_seq')
            self.c_dense = tf.keras.Sequential([
                tf.keras.layers.Dense(c_dense_n, tf.nn.relu) for _ in range(c_dense_depth)
            ] + [tf.keras.layers.Dense(1, name='c_q_output_dense')], name='c_seq')

    def call(self, state, c_action):
        state = self.dense(state)
        
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
