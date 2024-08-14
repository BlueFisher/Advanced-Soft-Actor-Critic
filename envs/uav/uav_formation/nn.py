import torch

import algorithm.nn_models as m


class ModelRep(m.ModelBaseRep):
    def _build_model(self):
        assert self.obs_shapes[0] == (3, 9)  # AgentsBufferSensor
        assert self.obs_shapes[1] == (22,)  # ray1 (1 + 5 + 5) * 2
        assert self.obs_shapes[2] == (22,)  # ray2 (1 + 5 + 5) * 2
        assert self.obs_shapes[3] == (22,)  # ray3 (1 + 5 + 5) * 2
        assert self.obs_shapes[4] == (12, )

        self.attn_envs = m.MultiheadAttention(9, 1)

        self.ray_dense = m.LinearLayers(input_size=22 * 3,
                                        dense_n=64,
                                        dense_depth=1)

        if self.d_action_sizes:
            self.rnn = m.GRU(9 + 12 + 64 + sum(self.d_action_sizes), 128, 1)
        else:
            self.rnn = m.GRU(9 + 12 + 64 + self.c_action_size, 128, 1)

    def forward(self, obs_list, pre_action, rnn_state=None, padding_mask=None):
        feature_agents, ray_obs_1, ray_obs_2, ray_obs_3, vec_obs = obs_list

        feature_agents_mask = ~feature_agents.any(dim=-1)
        feature_agents_mask[..., 0] = False

        attned_agents, _ = self.attn_envs(vec_obs[..., :-3].unsqueeze(-2), feature_agents, feature_agents,
                                          key_padding_mask=feature_agents_mask)

        attned_agents = attned_agents.squeeze(-2)

        if padding_mask is not None:
            attned_agents[padding_mask] = 0.

        ray = self.ray_dense(torch.concat([ray_obs_1, ray_obs_2, ray_obs_3], dim=-1))

        state, hn = self.rnn(torch.concat([attned_agents,
                                           vec_obs,
                                           ray,
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
