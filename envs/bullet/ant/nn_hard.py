import torch
from torch import nn

import algorithm.nn_models as m


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
    def _build_model(self):
        super()._build_model(c_dense_n=256, c_dense_depth=3)


class ModelPolicy(m.ModelPolicy):
    def _build_model(self):
        super()._build_model(c_dense_n=256, c_dense_depth=3,
                             mean_n=256, mean_depth=1,
                             logstd_n=256, logstd_depth=1)


class ModelTransition(m.ModelTransition):
    def _build_model(self):
        super()._build_model(dense_n=256, dense_depth=3, extra_size=3)

    def extra_obs(self, obs_list):
        return obs_list[0][..., 3:6]


class ModelReward(m.ModelReward):
    def _build_model(self):
        super()._build_model(dense_n=256, dense_depth=3)


class ModelObservation(m.ModelBaseObservation):
    def _build_model(self):

        self.dense = m.LinearLayers(self.state_size,
                                    256, 3,
                                    self.obs_shapes[0][0] if self.use_extra_data else self.obs_shapes[0][0] - 3)

    def forward(self, state):
        obs = self.dense(state)

        return obs

    def get_loss(self, state, obs_list):
        obs = obs_list[0]
        if not self.use_extra_data:
            obs = torch.cat([obs[..., :3], obs[..., 6:]], dim=-1)

        mse = nn.MSELoss()
        return mse(self(state), obs)
