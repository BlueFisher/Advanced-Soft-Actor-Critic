import torch

import algorithm.nn_models as m

EXTRA_SIZE = 4


class ModelRep(m.ModelBaseSimpleRep):
    def _bulid_model(self):
        assert self.obs_shapes[2] == (11,)

    def forward(self, obs_list):
        ray_1, ray_2, vec = obs_list
        vec = vec[..., :-EXTRA_SIZE]

        return torch.cat([ray_1, ray_2, vec], dim=-1)


class ModelQ(m.ModelQ):
    def _build_model(self):
        return super()._build_model(c_state_n=128, c_state_depth=1,
                                    c_action_n=128, c_action_depth=1,
                                    c_dense_n=128, c_dense_depth=2)


class ModelPolicy(m.ModelPolicy):
    def _build_model(self):
        return super()._build_model(c_dense_n=128, c_dense_depth=2,
                                    mean_n=128, mean_depth=1,
                                    logstd_n=128, logstd_depth=1)


class ModelRND(m.ModelRND):
    def _build_model(self):
        return super()._build_model(dense_n=128, dense_depth=2, output_size=128)
