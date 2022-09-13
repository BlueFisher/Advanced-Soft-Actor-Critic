import torch

import algorithm.nn_models as m


class ModelRep(m.ModelBaseRNNRep):
    def _build_model(self):
        assert self.obs_shapes[0] == (84, 84, 3)
        assert self.obs_shapes[1] == (84,)
        assert self.obs_shapes[2] == (9,)

        self.conv = m.ConvLayers(84, 84, 3, 'simple',
                                 out_dense_n=64, out_dense_depth=2)

        self.rnn = m.GRU(self.conv.output_size + 84 + 9 + self.c_action_size, 64, 1)

    def forward(self, obs_list, pre_action, rnn_state=None):
        vis_obs, ray_obs, vec_obs = obs_list

        vis_obs = self.conv(vis_obs)

        state, hn = self.rnn(torch.concat([vis_obs, ray_obs, vec_obs, pre_action], dim=-1), rnn_state)

        return state, hn


class ModelQ(m.ModelQ):
    def _build_model(self):
        return super()._build_model(c_dense_n=128, c_dense_depth=2)


class ModelPolicy(m.ModelPolicy):
    def _build_model(self):
        return super()._build_model(c_dense_n=128, c_dense_depth=2)


class ModelRND(m.ModelRND):
    def _build_model(self):
        return super()._build_model(dense_n=128, dense_depth=2, output_size=128)
