import tensorflow as tf


class ModelTransition(tf.keras.Model):
    def extra_obs(self, obs_list):
        shape = tf.shape(obs_list[0])
        return tf.zeros([shape[0], shape[1], 0])


class ModelSimpleRep(tf.keras.Model):
    def __init__(self, obs_dims):
        super().__init__()
        self.obs_dims = obs_dims

    def call(self, obs_list):
        raise Exception("ModelSimpleRep not implemented")

    def get_call_result_tensors(self):
        return self([tf.keras.Input(shape=t) for t in self.obs_dims])


class ModelVoidRep(ModelSimpleRep):
    def __init__(self, obs_dims):
        super().__init__(obs_dims)

        self.get_call_result_tensors()

    def call(self, obs_list):
        return obs_list[0]


class ModelRNNRep(tf.keras.Model):
    def __init__(self, obs_dims, action_dim):
        super().__init__()
        self.obs_dims = obs_dims
        self.action_dim = action_dim

    def call(self, obs_list, pre_action, rnn_state):
        raise Exception("ModelRNNRep not implemented")

    def get_call_result_tensors(self):
        return self([tf.keras.Input(shape=(None, *t)) for t in self.obs_dims],
                    tf.keras.Input(shape=(None, self.action_dim)),
                    tf.keras.Input(shape=(self.rnn_units,)))
