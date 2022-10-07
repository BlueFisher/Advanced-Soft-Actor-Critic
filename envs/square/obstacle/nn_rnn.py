import torch
from torch import nn

import algorithm.nn_models as m


class ModelRep(m.ModelBaseRNNRep):
    def _build_model(self):
        assert self.obs_shapes[0] == (55,)
        assert self.obs_shapes[1] == (6,)

        self.rnn = m.GRU(self.obs_shapes[0][0] + self.obs_shapes[1][0] + self.c_action_size, 64, 1)

    def forward(self, obs_list, pre_action, rnn_state=None):
        ray_obs, vec_obs = obs_list

        output, hn = self.rnn(torch.cat([ray_obs, vec_obs, pre_action], dim=-1), rnn_state)

        return output, hn


class ModelQ(m.ModelQ):
    def _build_model(self):
        return super()._build_model(c_state_n=64, c_state_depth=1,
                                    c_action_n=64, c_action_depth=1,
                                    c_dense_n=64, c_dense_depth=1)


class ModelPolicy(m.ModelPolicy):
    def _build_model(self):
        return super()._build_model(c_dense_n=64, c_dense_depth=1,
                                    mean_n=64, mean_depth=1,
                                    logstd_n=64, logstd_depth=1)
