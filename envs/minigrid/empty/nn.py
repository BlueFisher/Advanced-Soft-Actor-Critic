import torch
import algorithm.nn_models as m


class ModelRep(m.ModelBaseSimpleRep):
    def _build_model(self):
        assert self.obs_shapes[0] == (7, 7, 3)
        assert self.obs_shapes[1] == (1,)

        self.conv = m.ConvLayers(7, 7, 3, 'small',
                                 out_dense_n=64, out_dense_depth=2)

    def forward(self, obs_list):
        vis_obs, vec_obs = obs_list

        vis_obs = self.conv(vis_obs)

        state = torch.cat([vis_obs, vec_obs], dim=-1)

        return state


class ModelQ(m.ModelQ):
    def _build_model(self):
        return super()._build_model(d_dense_depth=1)


class ModelPolicy(m.ModelPolicy):
    def _build_model(self):
        return super()._build_model(d_dense_depth=1)


class ModelRND(m.ModelRND):
    def _build_model(self):
        return super()._build_model(output_size=64)