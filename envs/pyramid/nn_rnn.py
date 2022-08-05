import torch

import algorithm.nn_models as m

EXTRA_SIZE = 6


class ModelRep(m.ModelBaseRNNRep):
    def _build_model(self):
        assert self.obs_shapes[0] == (55,)
        assert self.obs_shapes[1] == (11,)

        self.rnn = m.GRU(55 + 11 - EXTRA_SIZE + self.c_action_size, 64, 1)

        self.dense = m.LinearLayers(input_size=64,
                                    dense_n=128, dense_depth=2, output_size=128)

    def forward(self, obs_list, pre_action, rnn_state=None):
        ray_obs, vec_obs = obs_list
        vec_obs = vec_obs[..., :-EXTRA_SIZE]

        x = torch.concat([ray_obs, vec_obs, pre_action], dim=-1)

        state, hn = self.rnn(x, rnn_state)
        state = self.dense(state)

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
