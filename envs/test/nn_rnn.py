import torch

import algorithm.nn_models as m

ModelVOverOption = m.ModelVOverOption
ModelTermination = m.ModelTermination


class ModelRep(m.ModelBaseRNNRep):
    def _build_model(self):
        if self.use_dilation:
            self.rnn = m.GRU(self.obs_shapes[0][0], 8, 2)
        else:
            self.rnn = m.GRU(self.obs_shapes[0][0] + sum(self.d_action_sizes) + self.c_action_size, 8, 2)

    def forward(self, obs_list, pre_action, rnn_state=None, padding_mask=None):
        obs = obs_list[0]

        if self.use_dilation:
            state, hn = self.rnn(obs, rnn_state)
        else:
            state, hn = self.rnn(torch.cat([obs, pre_action], dim=-1), rnn_state)

        return state, hn


class ModelOptionRep(ModelRep):
    pass


ModelQ = m.ModelQ
ModelPolicy = m.ModelPolicy
