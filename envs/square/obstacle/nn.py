import torch
from torch import nn

import algorithm.nn_models as m


class ModelRep(m.ModelBaseSimpleRep):
    def _build_model(self):
        assert self.obs_shapes[0] == (55,)
        assert self.obs_shapes[1] == (6,)

    def forward(self, obs_list):
        ray_obs, vec_obs = obs_list

        return torch.concat([ray_obs, vec_obs], dim=-1)


class ModelQ(m.ModelQ):
    def _build_model(self):
        return super()._build_model(c_state_n=128, c_state_depth=1,
                                    c_action_n=128, c_action_depth=1,
                                    c_dense_n=128, c_dense_depth=3)


class ModelPolicy(m.ModelPolicy):
    def _build_model(self):
        return super()._build_model(c_dense_n=128, c_dense_depth=2,
                                    mean_n=128, mean_depth=1,
                                    logstd_n=128, logstd_depth=1)
