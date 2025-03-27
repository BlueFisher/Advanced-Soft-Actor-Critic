import torch

import algorithm.nn_models as m


class ModelRep(m.ModelBaseRep):
    def _build_model(self):
        assert self.obs_shapes[0] == (3, 84, 84)
        assert self.obs_shapes[1] == (9,)

        self.conv = m.ConvLayers(84, 84, 3, 'simple',
                                 out_dense_n=64, out_dense_depth=2)

        if self.d_action_sizes:
            self.rnn = m.GRU(self.conv.output_size + 9 + sum(self.d_action_sizes), 128, 1)
        if self.c_action_size:
            self.rnn = m.GRU(self.conv.output_size + 9 + self.c_action_size, 128, 1)

    def forward(self, obs_list, pre_action, rnn_state=None, padding_mask=None):
        vis_obs, vec_obs = obs_list

        vis_obs = self.conv(vis_obs)

        if rnn_state is not None:
            rnn_state = rnn_state[:, 0]

        state, hn = self.rnn(torch.concat([vis_obs, vec_obs, pre_action], dim=-1), rnn_state)

        return state, hn


class ModelQ(m.ModelQ):
    def _build_model(self):
        return super()._build_model(d_dense_n=128, d_dense_depth=2,
                                    c_dense_n=128, c_dense_depth=2)


class ModelPolicy(m.ModelPolicy):
    def _build_model(self):
        return super()._build_model(d_dense_n=128, d_dense_depth=2,
                                    c_dense_n=128, c_dense_depth=2)
