import torch

import algorithm.nn_models as m
from algorithm.utils.visualization.image import ImageVisual


class ModelRep(m.ModelBaseRep):
    def _build_model(self):
        assert self.obs_shapes[0] == (3, 30, 30)
        assert self.obs_shapes[1] == (6,)

        self.conv = m.ConvLayers(30, 30, 3, 'simple',
                                 out_dense_n=64, out_dense_depth=2)

        self.dense = m.LinearLayers(self.conv.output_size + 2,
                                    dense_n=64, dense_depth=1)

    def forward(self, obs_list):
        vis_obs, vec_obs = obs_list
        vec_obs = vec_obs[..., -2:]

        vis_obs = self.conv(vis_obs)

        state = self.dense(torch.cat([vis_obs, vec_obs], dim=-1))

        return state


class ModelQ(m.ModelQ):
    def _build_model(self):
        return super()._build_model(c_dense_n=128, c_dense_depth=2)


class ModelPolicy(m.ModelPolicy):
    def _build_model(self):
        return super()._build_model(c_dense_n=128, c_dense_depth=2)
