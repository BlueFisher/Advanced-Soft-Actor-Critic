import tensorflow as tf


class ModelBaseSimpleRep(tf.keras.Model):
    def __init__(self, obs_dims):
        super().__init__()
        self.obs_dims = obs_dims

    def init(self):
        return self.call([tf.keras.Input(shape=o) for o in self.obs_dims])

    def call(self, obs_list):
        raise Exception("ModelSimpleRep not implemented")


class ModelSimpleRep(ModelBaseSimpleRep):
    def call(self, obs_list):
        return obs_list[0]


class ModelBaseRNNRep(tf.keras.Model):
    def __init__(self, obs_dims, action_dim, rnn_units):
        super().__init__()
        self.obs_dims = obs_dims
        self.action_dim = action_dim
        self.rnn_units = rnn_units

    def init(self):
        return self.call([tf.keras.Input(shape=(None, *o)) for o in self.obs_dims],
                         tf.keras.Input(shape=(None, self.action_dim)),
                         tf.keras.Input(shape=(self.rnn_units,)))

    def call(self, obs_list, pre_action, rnn_state):
        raise Exception("ModelRNNRep not implemented")


class ModelBaseGRURep(ModelBaseRNNRep):
    def __init__(self, obs_dims, action_dim,
                 rnn_units=64):
        super().__init__(obs_dims, action_dim, rnn_units)

        # TODO Disabled temporarily because of the issue
        # https://github.com/tensorflow/tensorflow/issues/39697
        self.gru = tf.keras.layers.RNN(tf.keras.layers.GRUCell(rnn_units),
                                       return_sequences=True,
                                       return_state=True)


class ModelBaseLSTMRep(ModelBaseRNNRep):
    def __init__(self, obs_dims, action_dim,
                 rnn_units=64):
        super().__init__(obs_dims, action_dim, rnn_units)

        # TODO Disabled temporarily because of the issue
        # https://github.com/tensorflow/tensorflow/issues/39697
        self.lstm = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(rnn_units),
                                        return_sequences=True,
                                        return_state=True)

    def init(self):
        return self.call([tf.keras.Input(shape=(None, *o)) for o in self.obs_dims],
                         tf.keras.Input(shape=(None, self.action_dim)),
                         tf.keras.Input(shape=(self.rnn_units * 2,)))
