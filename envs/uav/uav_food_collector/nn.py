import torch

import algorithm.nn_models as m


class ModelRep(m.ModelBaseRNNRep):
    def _build_model(self):
        assert self.obs_shapes[0] == (2, 9)  # AgentsBufferSensor
        assert self.obs_shapes[1] == (10, 3)  # TargetsBufferSensor
        assert self.obs_shapes[2] == (9, )

        self.rnn = m.GRU(2 * 9 + 10 * 3 + 9 + self.c_action_size, 128, 1)

    def forward(self, obs_list, pre_action, rnn_state=None, padding_mask=None):
        feature_agents, feature_targets, vec_obs = obs_list

        feature_agents = feature_agents.reshape(*feature_agents.shape[:-2], -1)
        feature_targets = feature_targets.reshape(*feature_targets.shape[:-2], -1)

        state, hn = self.rnn(torch.concat([feature_agents, feature_targets, vec_obs, pre_action], dim=-1), rnn_state)

        return state, hn


class ModelQ(m.ModelQ):
    def _build_model(self):
        return super()._build_model(c_dense_n=128, c_dense_depth=3)


class ModelPolicy(m.ModelPolicy):
    def _build_model(self):
        return super()._build_model(c_dense_n=128, c_dense_depth=3)


class ModelRND(m.ModelRND):
    def _build_model(self):
        return super()._build_model(dense_n=128, dense_depth=3, output_size=128)
