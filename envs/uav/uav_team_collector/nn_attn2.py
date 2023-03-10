import torch

import algorithm.nn_models as m


class ModelRep(m.ModelBaseRNNRep):
    def _build_model(self):
        assert self.obs_shapes[0] == (2, 3)  # AgentsBufferSensor
        assert self.obs_shapes[1] == (3, 3)  # TargetsBufferSensor
        assert self.obs_shapes[2] == (9, )

        self.attn = m.MultiheadAttention(4, 1)

        self.attn_self = m.MultiheadAttention(9, 1, kdim=4, vdim=4)

        self.rnn = m.GRU(self.obs_shapes[2][0] + 4 + self.c_action_size, 32, 1)

    def forward(self, obs_list, pre_action, rnn_state=None, padding_mask=None):
        feature_agents, feature_targets, vec_obs = obs_list

        feature_agents_mask = ~feature_agents.any(dim=-1)
        feature_targets_mask = ~feature_targets.any(dim=-1)
        feature_self_mask = torch.zeros((*feature_agents_mask.shape[:-1], 1), dtype=torch.bool, device=feature_agents_mask.device)
        mask = torch.concat([feature_self_mask, feature_agents_mask, feature_targets_mask], dim=-1)

        feature_self_indicator = torch.ones((*vec_obs.shape[:-1], 1), device=vec_obs.device)
        feature_self = torch.concat([feature_self_indicator, vec_obs[..., -3:]], dim=-1)
        feature_self = feature_self.unsqueeze(-2)

        feature_agents_indicator = torch.ones((*feature_agents.shape[:-1], 1), device=feature_agents.device)
        feature_agents = torch.concat([feature_agents_indicator, feature_agents], dim=-1)

        feature_targets_indicator = torch.zeros((*feature_targets.shape[:-1], 1), device=feature_targets.device)
        feature_targets = torch.concat([feature_targets_indicator, feature_targets], dim=-1)

        features = torch.concat([feature_self, feature_agents, feature_targets], dim=-2)

        features_attn, _ = self.attn(features, features, features, key_padding_mask=mask)
        features = features + features_attn

        features[mask] = 0.
        features = features[..., 1:, :]
        mask = mask[..., 1:]
        features = features.mean(-2)

        state, hn = self.rnn(torch.concat([vec_obs,
                                           features,
                                           pre_action], dim=-1), rnn_state)

        return state, hn


class ModelQ(m.ModelQ):
    def _build_model(self):
        return super()._build_model(c_dense_n=32, c_dense_depth=2)


class ModelPolicy(m.ModelPolicy):
    def _build_model(self):
        return super()._build_model(c_dense_n=32, c_dense_depth=2)


class ModelRND(m.ModelRND):
    def _build_model(self):
        return super()._build_model(dense_n=32, dense_depth=2, output_size=32)
