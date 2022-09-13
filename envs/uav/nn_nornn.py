import torch

from algorithm.utils.image_visual import ImageVisual
import algorithm.nn_models as m


class ModelRep(m.ModelBaseSimpleRep):
    def _build_model(self):
        assert self.obs_shapes[0] == (84, 84, 3)
        assert self.obs_shapes[1] == (84,)
        assert self.obs_shapes[2] == (9,)

        self._image_visual = ImageVisual()

        self.conv = m.ConvLayers(84, 84, 3, 'simple',
                                 out_dense_n=64, out_dense_depth=2)

    def forward(self, obs_list):
        vis_obs, ray_obs, vec_obs = obs_list

        # self._image_visual(vis_obs)

        vis_obs = self.conv(vis_obs)

        return torch.concat([vis_obs, vec_obs], dim=-1)


class ModelQ(m.ModelQ):
    def _build_model(self):
        return super()._build_model(c_dense_n=128, c_dense_depth=2)


class ModelPolicy(m.ModelPolicy):
    def _build_model(self):
        return super()._build_model(c_dense_n=128, c_dense_depth=2)


class ModelRND(m.ModelRND):
    def _build_model(self):
        return super()._build_model(dense_n=128, dense_depth=2, output_size=128)
