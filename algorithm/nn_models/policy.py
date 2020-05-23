import tensorflow as tf
import tensorflow_probability as tfp


class ModelBasePolicy(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

    def init(self):
        self(tf.keras.Input(shape=(self.state_dim,)))

    def call(self, state):
        raise Exception("ModelPolicy not implemented")


class ModelContinuesPolicy(ModelBasePolicy):
    def __init__(self, state_dim, action_dim,
                 dense_n=64, dense_depth=3,
                 mean_n=64, mean_depth=0,
                 logstd_n=64, logstd_depth=0):

        super().__init__(state_dim, action_dim)

        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(dense_n, tf.nn.relu) for _ in range(dense_depth)
        ])
        self.mean_model = tf.keras.Sequential([
            tf.keras.layers.Dense(mean_n, tf.nn.relu) for _ in range(mean_depth)
        ] + [tf.keras.layers.Dense(action_dim)])
        self.logstd_model = tf.keras.Sequential([
            tf.keras.layers.Dense(logstd_n, tf.nn.relu) for _ in range(logstd_depth)
        ] + [tf.keras.layers.Dense(action_dim)])

        self.tfpd = tfp.layers.DistributionLambda(
            make_distribution_fn=lambda t: tfp.distributions.Normal(t[0], t[1]))

    def call(self, state):
        l = self.dense(state)

        mean = self.mean_model(l)
        logstd = self.logstd_model(l)

        return self.tfpd([tf.tanh(mean), tf.clip_by_value(tf.exp(logstd), 0.1, 1.0)])


class ModelDiscretePolicy(ModelBasePolicy):
    def __init__(self, state_dim, action_dim,
                 dense_n=64, dense_depth=3):
        super().__init__(state_dim, action_dim,)

        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(dense_n, tf.nn.relu) for _ in range(dense_depth)
        ] + [tf.keras.layers.Dense(action_dim)])

        self.tfpd = tfp.layers.DistributionLambda(
            make_distribution_fn=lambda t: tfp.distributions.Categorical(logits=t))

    def call(self, state):
        logits = self.dense(state)

        return self.tfpd(logits)
