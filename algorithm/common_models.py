import tensorflow as tf


class ModelSimpleRep(tf.keras.Model):
    def __init__(self, obs_dims):
        super(ModelSimpleRep, self).__init__()
        self.obs_dims = obs_dims

        self.get_call_result_tensors()

    def call(self, obs_list):
        return obs_list[0]

    def get_call_result_tensors(self):
        return self([tf.keras.Input(shape=t) for t in self.obs_dims])


class ModelRNNRep(tf.keras.Model):
    def __init__(self, obs_dims):
        super(ModelRNNRep, self).__init__()
        self.obs_dims = obs_dims

    def call(self, obs_list, initial_state):
        raise Exception("ModelRNNRep not implemented")

    def get_call_result_tensors(self):
        return self([tf.keras.Input(shape=(None, *t)) for t in self.obs_dims],
                    tf.keras.Input(shape=(self.rnn_units,)))
