import torch

import algorithm.nn_models as m

ModelVOverOption = m.ModelVOverOption
ModelTermination = m.ModelTermination


class ModelRep(m.ModelBaseRNNRep):
    def _build_model(self):
        assert self.obs_shapes[0] == (2, 3, 4)

        self.rnn = m.GRU(2 * 3 * 4, 64, 1)

    def forward(self, obs_list, pre_action, rnn_state=None, padding_mask=None):
        vec_obs = obs_list[0]
        vec_obs = vec_obs.reshape(*vec_obs.shape[:-3], 2 * 3 * 4)

        output, hn = self.rnn(vec_obs, rnn_state)

        return output, hn


class ModelOptionRep(m.ModelBaseRNNRep):
    def _build_model(self):
        assert self.obs_shapes[1] == (2, 3, 4)

        self.rnn = m.GRU(2 * 3 * 4, 64, 1)

    def forward(self, obs_list, pre_action, rnn_state=None, padding_mask=None):
        high_state, vec_obs = obs_list
        vec_obs = vec_obs.reshape(*vec_obs.shape[:-3], 2 * 3 * 4)

        output, hn = self.rnn(vec_obs, rnn_state)

        state = torch.cat([high_state, output], dim=-1)

        return state, hn


class ModelQ(m.ModelQ):
    def _build_model(self):
        return super()._build_model(d_dense_n=128, d_dense_depth=2)


class ModelPolicy(m.ModelPolicy):
    def _build_model(self):
        return super()._build_model(d_dense_n=128, d_dense_depth=2)


class ModelRND(m.ModelRND):
    def _build_model(self):
        return super()._build_model(dense_n=128, dense_depth=2, output_size=128)
