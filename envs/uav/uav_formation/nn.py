import torch

import algorithm.nn_models as m


class ModelRep(m.ModelBaseRNNRep):
    def _build_model(self):
        assert self.obs_shapes[0] == (3, 9)  # AgentsBufferSensor
        assert self.obs_shapes[1] == (12, )

        self.attn_envs = m.MultiheadAttention(9, 1)

        if self.d_action_sizes:
            self.rnn = m.GRU(9 + 12 + sum(self.d_action_sizes), 128, 1)
        else:
            self.rnn = m.GRU(9 + 12 + self.c_action_size, 128, 1)

    def forward(self, obs_list, pre_action, rnn_state=None, padding_mask=None):
        feature_agents, vec_obs = obs_list

        feature_agents_mask = ~feature_agents.any(dim=-1)
        feature_agents_mask[..., 0] = False

        attned_agents, _ = self.attn_envs(vec_obs[..., :-3].unsqueeze(-2), feature_agents, feature_agents,
                                          key_padding_mask=feature_agents_mask)

        attned_agents = attned_agents.squeeze(-2)

        if padding_mask is not None:
            attned_agents[padding_mask] = 0.

        state, hn = self.rnn(torch.concat([attned_agents,
                                           vec_obs,
                                           pre_action], dim=-1), rnn_state)

        if padding_mask is not None:
            state = state * (~padding_mask).to(state.dtype).unsqueeze(-1)

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
        return super()._build_model(dense_n=128, dense_depth=2, output_size=128)
