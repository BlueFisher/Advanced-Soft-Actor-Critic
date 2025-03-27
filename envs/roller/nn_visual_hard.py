import torch
from torch import nn

import algorithm.nn_models as m

EXTRA_SIZE = 2


class ModelRep(m.ModelBaseRep):
    def _build_model(self):
        assert self.obs_shapes[0] == (3, 30, 30)
        assert self.obs_shapes[1] == (EXTRA_SIZE,)

        self.conv = m.ConvLayers(30, 30, 3, 'simple',
                                 out_dense_n=64, out_dense_depth=2)

        self.rnn = m.GRU(self.conv.output_size + self.c_action_size, 64, 1)

    def forward(self, obs_list, pre_action, rnn_state=None, padding_mask=None):
        vis_obs, vec_obs = obs_list

        vis_obs = self.conv(vis_obs)

        if rnn_state is not None:
            rnn_state = rnn_state[:, 0]

        output, hn = self.rnn(torch.cat([vis_obs, pre_action], dim=-1), rnn_state)

        return output, hn


class ModelTransition(m.ModelTransition):
    def _build_model(self):
        return super()._build_model(dense_depth=2, extra_size=EXTRA_SIZE)

    def extra_obs(self, obs_list):
        return obs_list[1]


class ModelReward(m.ModelReward):
    def _build_model(self):
        return super()._build_model(dense_depth=2)


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
                                        output_size=EXTRA_SIZE)

    def forward(self, state):
        vis_obs = self.conv_transpose(state)
        vec_obs = self.vec_dense(state)

        return vis_obs, vec_obs

    def get_loss(self, state, obs_list):
        approx_vis_obs, approx_vec_obs = self(state)
        vis_obs, vec_obs = obs_list

        mse = nn.MSELoss()

        if self.use_extra_data:
            return mse(approx_vis_obs, vis_obs) + mse(approx_vec_obs, vec_obs)
        else:
            return mse(approx_vis_obs, vis_obs)


class ModelQ(m.ModelQ):
    def _build_model(self):
        return super()._build_model(c_dense_n=128, c_dense_depth=2)


class ModelPolicy(m.ModelPolicy):
    def _build_model(self):
        return super()._build_model(c_dense_n=128, c_dense_depth=2)
