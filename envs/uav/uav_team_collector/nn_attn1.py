import torch

import algorithm.nn_models as m


class ModelRep(m.ModelBaseRNNRep):
    def _build_model(self):
        assert self.obs_shapes[0] == (2, 3)  # AgentsBufferSensor
        assert self.obs_shapes[1] == (3, 3)  # TargetsBufferSensor
        assert self.obs_shapes[2] == (9, )

        self.agents_attn = m.MultiheadAttention(9, 1, kdim=3, vdim=3)
        self.targets_attn = m.MultiheadAttention(9, 1, kdim=3, vdim=3)

        self.rnn = m.GRU(self.obs_shapes[2][0] + self.c_action_size, 16, 1)

    def forward(self, obs_list, pre_action, rnn_state=None, padding_mask=None):
        feature_agents, feature_targets, vec_obs = obs_list

        feature_agents_mask = ~feature_agents.any(dim=-1)
        feature_agents_mask[..., 0] = False
        feature_targets_mask = ~feature_targets.any(dim=-1)
        feature_targets_mask[..., 0] = False

        attned_agents, _ = self.agents_attn(vec_obs.unsqueeze(-2), feature_agents, feature_agents, key_padding_mask=feature_agents_mask)
        attned_targets, _ = self.targets_attn(vec_obs.unsqueeze(-2), feature_targets, feature_targets, key_padding_mask=feature_targets_mask)

        attned_agents = attned_agents.squeeze(-2)
        attned_targets = attned_targets.squeeze(-2)

        if padding_mask is not None:
            attned_agents[padding_mask] = 0.
            attned_targets[padding_mask] = 0.

        state, hn = self.rnn(torch.concat([vec_obs,
                                           pre_action], dim=-1), rnn_state)

        if padding_mask is not None:
            state = state * (~padding_mask).to(state.dtype).unsqueeze(-1)

        state = torch.concat([state, attned_agents, attned_targets,], dim=-1)

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
