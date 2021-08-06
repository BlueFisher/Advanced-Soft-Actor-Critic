import torch

import algorithm.nn_models as m

EXTRA_SIZE = 2


class ModelRep(m.ModelBaseSimpleRep):
    def _bulid_model(self):
        assert self.obs_shapes[0] == (84, 84, 3)
        assert self.obs_shapes[1] == (84, 84, 3)
        assert self.obs_shapes[2] == (84, 84, 3)
        assert self.obs_shapes[3] == (8,)

        self.conv = m.ConvLayers(84, 84, 9, 'simple',
                                  out_dense_n=64, out_dense_depth=2)

        self.dense = m.LinearLayers(self.conv.output_size + 8 - EXTRA_SIZE,
                                    dense_n=64, dense_depth=1)

    def forward(self, obs_list):
        *vis, vec = obs_list
        vec = vec[..., :-EXTRA_SIZE]

        vis = self.conv(torch.cat(vis, dim=-1))

        state = self.dense(torch.cat([vis, vec], dim=-1))

        return state


class ModelQ(m.ModelQ):
    def _build_model(self):
        return super()._build_model(c_dense_n=128, c_dense_depth=2)


class ModelPolicy(m.ModelPolicy):
    def _build_model(self):
        return super()._build_model(c_dense_n=128, c_dense_depth=2)


class ModelRND(m.ModelRND):
    def _build_model(self):
        return super()._build_model(dense_n=128, dense_depth=2, output_size=128)
