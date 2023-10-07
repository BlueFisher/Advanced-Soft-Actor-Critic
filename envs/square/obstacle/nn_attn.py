import torch
from torch import nn

import algorithm.nn_models as m


class ModelRep(m.ModelBaseAttentionRep):
    def _build_model(self):
        assert self.obs_shapes[0] == (55,)
        assert self.obs_shapes[1] == (6,)

        if self.use_dilation:
            self.mlp = m.LinearLayers(self.obs_shapes[0][0] + self.obs_shapes[1][0], 64, 1)
        else:
            self.mlp = m.LinearLayers(self.obs_shapes[0][0] + self.obs_shapes[1][0] + self.c_action_size, 64, 1)
        self.attn = m.EpisodeMultiheadAttention(64, 8, num_layers=2)
        self.pos = m.AbsolutePositionalEncoding(64)

    def forward(self, index, obs_list, pre_action=None,
                seq_q_len=1,
                hidden_state=None,
                is_prev_hidden_state=False,
                query_only_attend_to_rest_key=False,
                padding_mask=None):
        ray_obs, vec_obs = obs_list

        if self.use_dilation:
            x = self.mlp(torch.cat([ray_obs, vec_obs], dim=-1))
        else:
            x = self.mlp(torch.cat([ray_obs, vec_obs, pre_action], dim=-1))
        pe = self.pos(index)
        x = x + pe

        output, hn, attn_weights_list = self.attn(x,
                                                  seq_q_len,
                                                  hidden_state,
                                                  is_prev_hidden_state,
                                                  query_only_attend_to_rest_key,
                                                  padding_mask)

        return output, hn, attn_weights_list


class ModelOptionRep(m.ModelBaseRNNRep):
    def _build_model(self):
        assert self.obs_shapes[0] == (64,)
        assert self.obs_shapes[1] == (55,)
        assert self.obs_shapes[2] == (6,)

        self.rnn = m.GRU(self.obs_shapes[0][0] + self.obs_shapes[1][0] + self.c_action_size, 8, 1)

    def forward(self, obs_list, pre_action, rnn_state=None, padding_mask=None):
        high_state, ray_obs, vec_obs = obs_list

        output, hn = self.rnn(torch.cat([high_state, ray_obs, pre_action], dim=-1), rnn_state)

        state = torch.cat([vec_obs, output], dim=-1)

        return state, hn


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
