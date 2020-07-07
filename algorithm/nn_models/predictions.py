import tensorflow as tf


class ModelBaseTransition(tf.keras.Model):
    def __init__(self, state_dim, action_dim, use_extra_data):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_extra_data = use_extra_data

    def init(self):
        return self(tf.keras.Input(shape=(self.state_dim,)),
                    tf.keras.Input(shape=(self.action_dim,)))

    def call(self, state, action):
        raise Exception("ModelBaseTransition not implemented")

    def extra_obs(self, obs_list):
        shape = tf.shape(obs_list[0])
        return tf.zeros([shape[0], shape[1], 0])


class ModelBaseReward(tf.keras.Model):
    def __init__(self, state_dim, use_extra_data,):
        super().__init__()
        self.state_dim = state_dim
        self.use_extra_data = use_extra_data

    def init(self):
        return self(tf.keras.Input(shape=(self.state_dim,)))

    def call(self, state):
        raise Exception("ModelBaseReward not implemented")


class ModelBaseObservation(tf.keras.Model):
    def __init__(self, state_dim, obs_dims, use_extra_data):
        super().__init__()
        self.state_dim = state_dim
        self.obs_dims = obs_dims
        self.use_extra_data = use_extra_data

    def init(self):
        return self(tf.keras.Input(shape=(self.state_dim,)))

    def call(self, state):
        raise Exception("ModelBaseObservation not implemented")

    def get_loss(self, state, obs_list):
        raise Exception("ModelBaseObservation not implemented")
