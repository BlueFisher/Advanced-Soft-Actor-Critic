import torch
from torch import nn

import algorithm.nn_models as m

EXTRA_SIZE = 2


class ModelRep(m.ModelBaseRNNRep):
    def _build_model(self):
        assert self.obs_shapes[0] == (30, 30, 3)
        assert self.obs_shapes[1] == (6,)

        self.conv = m.ConvLayers(30, 30, 3, 'simple',
                                 out_dense_n=64, out_dense_depth=2)

        self.rnn = m.GRU(self.conv.output_size + self.c_action_size, 64, 1)

        self.dense = m.LinearLayers(input_size=64 + self.obs_shapes[1][0] - EXTRA_SIZE,
                                    dense_n=128, dense_depth=2, output_size=64)

    def forward(self, obs_list, pre_action, rnn_state=None):
        vis_obs, vec_obs = obs_list
        vec_obs = vec_obs[..., :-EXTRA_SIZE]

        vis_obs = self.conv(vis_obs)

        output, hn = self.rnn(torch.cat([vis_obs, pre_action], dim=-1), rnn_state)

        state = self.dense(torch.cat([vec_obs, output], dim=-1))

        return state, hn


class ModelTransition(m.ModelTransition):
    def _build_model(self):
        return super()._build_model(dense_n=128, dense_depth=2, extra_size=EXTRA_SIZE)

    def extra_obs(self, obs_list):
        return obs_list[1][..., :-EXTRA_SIZE]


class ModelReward(m.ModelReward):
    def _build_model(self):
        return super()._build_model(dense_n=128, dense_depth=2)


class ModelObservation(m.ModelBaseObservation):
    def _build_model(self):
        self.conv_transpose = m.ConvTransposeLayers(
            self.state_size, 64, 1,
            2, 2, 32,
            conv_transpose=nn.Sequential(
                nn.ConvTranspose2d(32, 32, 4, 2),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(32, 16, 8, 4),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(16, 3, 3, 1),
                nn.LeakyReLU(),
            ))

        self.vec_dense = m.LinearLayers(self.state_size, dense_depth=2,
                                        output_size=self.obs_shapes[1][0] if self.use_extra_data else self.obs_shapes[1][0] - EXTRA_SIZE)

    def forward(self, state):
        vis_obs = self.conv_transpose(state)
        vec_obs = self.vec_dense(state)

        return vis_obs, vec_obs

    def get_loss(self, state, obs_list):
        approx_vis_obs, approx_vec_obs = self(state)
        vis_obs, vec_obs = obs_list
        if not self.use_extra_data:
            vec_obs = vec_obs[..., :-EXTRA_SIZE]

        mse = nn.MSELoss()

        return mse(approx_vis_obs, vis_obs) + mse(approx_vec_obs, vec_obs)


class ModelQ(m.ModelQ):
    def _build_model(self):
        return super()._build_model(c_dense_n=128, c_dense_depth=2)


class ModelPolicy(m.ModelPolicy):
    def _build_model(self):
        return super()._build_model(c_dense_n=128, c_dense_depth=2)
