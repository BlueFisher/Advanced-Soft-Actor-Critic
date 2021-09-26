import torch

import algorithm.nn_models as m

EXTRA_SIZE = 6


class ModelRep(m.ModelBaseRNNRep):
    def _build_model(self):
        assert self.obs_shapes[0] == (84, 84, 3)
        assert self.obs_shapes[1] == (2,)  # ray
        assert self.obs_shapes[2] == (8,)  # vector

        self.conv = m.ConvLayers(84, 84, 3, 'simple',
                                 out_dense_n=64, out_dense_depth=2)

        self.dense = m.LinearLayers(self.conv.output_size,
                                    dense_n=64, dense_depth=1)

        self.rnn = m.GRU(64 + self.c_action_size, 64, 1)

    def forward(self, obs_list, pre_action, rnn_state=None):
        vis_cam, ray, vec = obs_list
        vec = vec[..., :-EXTRA_SIZE]

        vis = self.conv(vis_cam)

        state, hn = self.rnn(torch.cat([self.dense(vis), pre_action], dim=-1), rnn_state)

        state = torch.cat([state, ray, vec], dim=-1)

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
