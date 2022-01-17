import math

import torch
from torch import nn

import algorithm.nn_models as m

EXTRA_SIZE = 2


class ModelRep(m.ModelBaseAttentionRep):
    def _build_model(self):
        assert self.obs_shapes[0] == (6, )

        embed_dim = 6 - EXTRA_SIZE + 2
        self.attn = m.EpisodeMultiheadAttention(embed_dim*2, 2, num_layers=3)
        self.pos = PositionalEncoding(embed_dim)

    def forward(self, index, obs_list, pre_action,
                query_length=1,
                hidden_state=None,
                is_prev_hidden_state=False,
                padding_mask=None):
        obs = obs_list[0][..., :-EXTRA_SIZE]

        x = torch.concat([obs, pre_action], dim=-1)

        pe = self.pos(index)
        # i = x + pe
        i = torch.concat([x, pe], dim=-1)
        output, hn, attn_weights_list = self.attn(i,
                                                  query_length,
                                                  hidden_state,
                                                  is_prev_hidden_state,
                                                  padding_mask)

        return output, hn, attn_weights_list


class ModelQ(m.ModelQ):
    def _build_model(self):
        return super()._build_model(c_dense_n=64, c_dense_depth=2)


class ModelPolicy(m.ModelPolicy):
    def _build_model(self):
        return super()._build_model(c_dense_n=64, c_dense_depth=2)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        self.register_buffer('pe', pe)

    def forward(self, indexes):
        with torch.no_grad():
            return self.pe[indexes.type(torch.int64)]
