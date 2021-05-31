import torch
from torch import nn

import algorithm.nn_models as m

EXTRA_SIZE = 2


class ModelRep(m.ModelBaseRNNRep):
    def _build_model(self):
        assert self.obs_shapes[0] == (6, )

        self.rnn = m.GRU(self.obs_shapes[0][0] - EXTRA_SIZE + self.c_action_size, 64, 1)

        self.dense = nn.Sequential(
            nn.Linear(self.obs_shapes[0][0] - EXTRA_SIZE + 64, 64),
            nn.Tanh()
        )

    def forward(self, obs_list, pre_action, rnn_state=None):
        obs = obs_list[0][..., :-EXTRA_SIZE]

        output, hn = self.rnn(torch.cat([obs, pre_action], dim=-1), rnn_state)

        state = self.dense(torch.cat([obs, output], dim=-1))

        return state, hn


class ModelTransition(m.ModelTransition):
    def _build_model(self):
        return super()._build_model(dense_depth=2, extra_size=EXTRA_SIZE)

    def extra_obs(self, obs_list):
        return obs_list[0][..., -EXTRA_SIZE:]


class ModelReward(m.ModelReward):
    def _build_model(self):
        return super()._build_model(dense_depth=1)


class ModelObservation(m.ModelBaseObservation):
    def _build_model(self):
        self.dense = m.LinearLayers(self.state_size,
                                    64, 1,
                                    self.obs_shapes[0][0] if self.use_extra_data else self.obs_shapes[0][0] - EXTRA_SIZE)

    def forward(self, state):
        return self.dense(state)

    def get_loss(self, state, obs_list):
        mse = nn.MSELoss()

        return mse(self(state), obs_list[0] if self.use_extra_data else obs_list[0][..., :-EXTRA_SIZE])


class ModelQ(m.ModelQ):
    def _build_model(self):
        return super()._build_model(c_dense_n=64, c_dense_depth=2)


class ModelPolicy(m.ModelPolicy):
    def _build_model(self):
        return super()._build_model(c_dense_n=64, c_dense_depth=2)
