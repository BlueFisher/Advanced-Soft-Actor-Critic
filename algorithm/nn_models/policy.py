import tensorflow as tf
import tensorflow_probability as tfp


class ModelBasePolicy(tf.keras.Model):
    def __init__(self, state_dim, d_action_dim, c_action_dim, name=None):
        super().__init__(name=name)
        self.state_dim = state_dim
        self.d_action_dim = d_action_dim
        self.c_action_dim = c_action_dim

    def init(self):
        self(tf.keras.Input(shape=(self.state_dim,)))

    def call(self, state):
        raise Exception("ModelPolicy not implemented")


class ModelPolicy(ModelBasePolicy):
    def __init__(self, state_dim, d_action_dim, c_action_dim, name=None,
                 dense_n=64, dense_depth=0,
                 d_dense_n=64, d_dense_depth=3,
                 c_dense_n=64, c_dense_depth=3,
                 mean_n=64, mean_depth=0,
                 logstd_n=64, logstd_depth=0):
        super().__init__(state_dim, d_action_dim, c_action_dim, name)

        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(dense_n, tf.nn.relu) for _ in range(dense_depth)
        ], name='shared_seq')

        if self.d_action_dim:
            self.d_dense = tf.keras.Sequential([
                tf.keras.layers.Dense(d_dense_n, tf.nn.relu) for _ in range(d_dense_depth)
            ] + [tf.keras.layers.Dense(d_action_dim)], name='d_seq')
            self.d_tfpd = tfp.layers.DistributionLambda(
                make_distribution_fn=lambda t: tfp.distributions.Categorical(logits=t))

        if self.c_action_dim:
            self.c_dense = tf.keras.Sequential([
                tf.keras.layers.Dense(c_dense_n, tf.nn.relu) for _ in range(c_dense_depth)
            ], name='c_seq')
            self.mean_model = tf.keras.Sequential([
                tf.keras.layers.Dense(mean_n, tf.nn.relu) for _ in range(mean_depth)
            ] + [tf.keras.layers.Dense(c_action_dim, name='c_normal_output_dense')], name='c_mean_seq')
            self.logstd_model = tf.keras.Sequential([
                tf.keras.layers.Dense(logstd_n, tf.nn.relu) for _ in range(logstd_depth)
            ] + [tf.keras.layers.Dense(c_action_dim, name='c_normal_output_dense')], name='c_logstd_seq')
            self.c_tfpd = tfp.layers.DistributionLambda(
                make_distribution_fn=lambda t: tfp.distributions.Normal(t[0], t[1]))

    def call(self, state):
        state = self.dense(state)
        
        if self.d_action_dim:
            logits = self.d_dense(state)
            d_policy = self.d_tfpd(logits)
        else:
            d_policy = tf.zeros((0,))

        if self.c_action_dim:
            l = self.c_dense(state)
            mean = self.mean_model(l)
            logstd = self.logstd_model(l)
            c_policy = self.c_tfpd([tf.tanh(mean), tf.clip_by_value(tf.exp(logstd), 0.1, 1.0)])
        else:
            c_policy = tf.zeros((0,))

        return d_policy, c_policy
