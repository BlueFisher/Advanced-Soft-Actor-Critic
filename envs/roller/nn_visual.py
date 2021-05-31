import torch

import algorithm.nn_models as m


class ModelRep(m.ModelBaseSimpleRep):
    def _bulid_model(self):
        assert self.obs_shapes[0] == (30, 30, 3)
        assert self.obs_shapes[1] == (2,)

        self.conv = m.ConvLayers(30, 30, 3, 'simple',
                                 out_dense_n=64, out_dense_depth=2)

        self.dense = m.LinearLayers(self.conv.output_size + 2,
                                    dense_n=64, dense_depth=1)

    def forward(self, obs_list):
        vis_obs, vec_obs = obs_list

        vis_obs = self.conv(vis_obs)

        state = self.dense(torch.cat([vis_obs, vec_obs], dim=-1))

        return state


class ModelQ(m.ModelQ):
    def _build_model(self):
        return super()._build_model(c_dense_n=128, c_dense_depth=2)


class ModelPolicy(m.ModelPolicy):
    def _build_model(self):
        return super()._build_model(c_dense_n=128, c_dense_depth=2)
