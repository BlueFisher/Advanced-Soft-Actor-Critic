import torch
from torch import nn

import algorithm.nn_models as m

ModelVOverOption = m.ModelVOverOption
ModelTermination = m.ModelTermination

EXTRA_SIZE = 2


class ModelRep(m.ModelBaseAttentionRep):
    def _build_model(self):
        assert self.obs_shapes[0] == (6, )

        embed_dim = 6 - EXTRA_SIZE + 2  # 6
        self.mlp = m.LinearLayers(embed_dim, output_size=32)
        self.attn = m.EpisodeMultiheadAttention(32, 2, num_layers=2)
        self.pos = m.AbsolutePositionalEncoding(32)

    def forward(self, index, obs_list, pre_action,
                query_length=1,
                hidden_state=None,
                is_prev_hidden_state=False,
                query_only_attend_to_rest_key=False,
                padding_mask=None):
        obs = obs_list[0][..., :-EXTRA_SIZE]

        x = torch.concat([obs, pre_action], dim=-1)
        x = self.mlp(x)

        pe = self.pos(index)
        # x = torch.concat([x, pe], dim=-1)
        x = x + pe

        output, hn, attn_weights_list = self.attn(x,
                                                  query_length,
                                                  hidden_state,
                                                  is_prev_hidden_state,
                                                  query_only_attend_to_rest_key,
                                                  padding_mask)

        return output, hn, attn_weights_list


class ModelOptionRep(ModelRep):
    def _build_model(self):
        assert self.obs_shapes[0] == (32, )
        assert self.obs_shapes[1] == (6, )

        self.rnn = m.GRU(self.obs_shapes[1][0] - EXTRA_SIZE + self.c_action_size, 32, 1)

    def forward(self, obs_list, pre_action, rnn_state=None, padding_mask=None):
        high_rep, obs = obs_list
        obs = obs[..., :-EXTRA_SIZE]

        state, hn = self.rnn(torch.cat([obs, pre_action], dim=-1), rnn_state)

        state = high_rep + state

        return state, hn


class ModelQ(m.ModelQ):
    def _build_model(self):
        return super()._build_model(c_dense_n=64, c_dense_depth=2)


class ModelPolicy(m.ModelPolicy):
    def _build_model(self):
        return super()._build_model(c_dense_n=64, c_dense_depth=2)
