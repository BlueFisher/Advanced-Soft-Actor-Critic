import torch

import algorithm.nn_models as m

N_OTHER_AGENTS = 3


class ModelRep(m.ModelBaseRNNRep):
    def _build_model(self):
        assert self.obs_shapes[0] == (N_OTHER_AGENTS, 9)  # AgentsBufferSensor
        assert self.obs_shapes[1] == (84, 84, 3)
        assert self.obs_shapes[2] == (9, )

        self.conv = m.ConvLayers(84, 84, 3, 'simple',
                                 out_dense_n=64, out_dense_depth=2)

        if self.d_action_sizes:
            self.rnn = m.GRU(N_OTHER_AGENTS * 9 + self.conv.output_size + 9 + sum(self.d_action_sizes), 128, 1)
        else:
            self.rnn = m.GRU(N_OTHER_AGENTS * 9 + self.conv.output_size + 9 + self.c_action_size, 128, 1)

    def forward(self, obs_list, pre_action, rnn_state=None, padding_mask=None):
        feature_agents, vis_obs, vec_obs = obs_list

        feature_agents = feature_agents.reshape(*feature_agents.shape[:-2], -1)

        vis_obs = self.conv(vis_obs)

        state, hn = self.rnn(torch.concat([feature_agents,
                                           vis_obs,
                                           vec_obs,
                                           pre_action], dim=-1), rnn_state)

        return state, hn


class ModelQ(m.ModelQ):
    def _build_model(self):
        return super()._build_model(d_dense_n=128, d_dense_depth=2,
                                    c_dense_n=128, c_dense_depth=2)


class ModelPolicy(m.ModelPolicy):
    def _build_model(self):
        return super()._build_model(d_dense_n=128, d_dense_depth=2,
                                    c_dense_n=128, c_dense_depth=2)


class ModelRND(m.ModelRND):
    def _build_model(self):
        return super()._build_model(dense_n=128, dense_depth=3, output_size=128)
