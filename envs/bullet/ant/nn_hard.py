import torch
from torch import nn

import algorithm.nn_models as m


# class ModelForward(m.ModelForward):
#     def __init__(self, state_size, action_size):
#         super().__init__(state_size, action_size,
#                          dense_n=256, dense_depth=3)


# class ModelTransition(m.ModelBaseTransition):
#     def __init__(self, state_size, d_action_size, c_action_size, use_extra_data):
#         super().__init__(state_size, d_action_size, c_action_size, use_extra_data)

#         self.dense = tf.keras.Sequential([
#             tf.keras.layers.Dense(256, activation=tf.nn.relu),
#             tf.keras.layers.Dense(256, activation=tf.nn.relu),
#             tf.keras.layers.Dense(256, activation=tf.nn.relu),
#             tf.keras.layers.Dense(state_size + state_size)
#         ])

#         self.next_state_tfpd = tfp.layers.DistributionLambda(
#             make_distribution_fn=lambda t: tfp.distributions.Normal(t[0], t[1]))

#     def call(self, state, action):
#         next_state = self.dense(tf.concat([state, action], -1))
#         mean, logstd = tf.split(next_state, num_or_size_splits=2, axis=-1)
#         next_state_dist = self.next_state_tfpd([mean, tf.maximum(tf.exp(logstd), 1.0)])

#         return next_state_dist


# class ModelReward(m.ModelBaseReward):
#     def __init__(self, state_size, use_extra_data):
#         super().__init__(state_size, use_extra_data)

#         self.dense = tf.keras.Sequential([
#             tf.keras.layers.Dense(256, activation=tf.nn.relu),
#             tf.keras.layers.Dense(256, activation=tf.nn.relu),
#             tf.keras.layers.Dense(256, activation=tf.nn.relu),
#             tf.keras.layers.Dense(1)
#         ])

#     def call(self, state):
#         reward = self.dense(state)

#         return reward


# class ModelObservation(m.ModelBaseObservation):
#     def __init__(self, state_size, obs_shapes, use_extra_data):
#         super().__init__(state_size, obs_shapes, use_extra_data)

#         self.dense = tf.keras.Sequential([
#             tf.keras.layers.Dense(256, activation=tf.nn.relu),
#             tf.keras.layers.Dense(256, activation=tf.nn.relu),
#             tf.keras.layers.Dense(256, activation=tf.nn.relu),
#             tf.keras.layers.Dense(obs_shapes[0][0] if use_extra_data else obs_shapes[0][0] - 3)
#         ])

#     def call(self, state):
#         obs = self.dense(state)

#         return obs

#     def get_loss(self, state, obs_list):
#         approx_obs = self(state)

#         obs = obs_list[0]
#         if not self.use_extra_data:
#             obs = tf.concat([obs[..., :3], obs[..., 6:]], axis=-1)

#         return tf.reduce_mean(tf.square(approx_obs - obs))


class ModelRep(m.ModelBaseRNNRep):
    def __init__(self, obs_shapes, d_action_size, c_action_size):
        super().__init__(obs_shapes, d_action_size, c_action_size)

        self.rnn = m.GRU(obs_shapes[0][0] - 3 + d_action_size + c_action_size, 64, 1)

        self.dense = nn.Sequential(
            nn.Linear(obs_shapes[0][0] - 3 + 64, 32),
            nn.Tanh()
        )

    def forward(self, obs_list, pre_action, rnn_state=None):
        obs = obs_list[0]
        obs = torch.cat([obs[..., :3], obs[..., 6:]], dim=-1)

        output, hn = self.rnn(torch.cat([obs, pre_action], dim=-1), rnn_state)

        state = self.dense(torch.cat([obs, output], dim=-1))

        return state, hn


class ModelQ(m.ModelQ):
    def __init__(self, state_size, d_action_size, c_action_size):
        super().__init__(state_size, d_action_size, c_action_size,
                         c_dense_n=256, c_dense_depth=3)


class ModelPolicy(m.ModelPolicy):
    def __init__(self, state_size, d_action_size, c_action_size):
        super().__init__(state_size, d_action_size, c_action_size,
                         c_dense_n=256, c_dense_depth=3,
                         mean_n=256, mean_depth=1,
                         logstd_n=256, logstd_depth=1)
