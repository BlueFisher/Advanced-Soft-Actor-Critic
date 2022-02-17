import torch
from torch import nn

import algorithm.nn_models as m

EXTRA_SIZE = 3


class ModelRep(m.ModelBaseAttentionRep):
    def _build_model(self):
        assert self.obs_shapes[0] == (8, )

        embed_dim = 8 - EXTRA_SIZE + 2
        self.attn = m.EpisodeMultiheadAttention(embed_dim + 8, 3, num_layers=2)
        self.pos = m.AbsolutePositionalEncoding(8)

    def forward(self, index, obs_list, pre_action,
                query_length=1,
                hidden_state=None,
                is_prev_hidden_state=False,
                padding_mask=None):
        obs = obs_list[0][..., :-EXTRA_SIZE]

        x = torch.concat([obs, pre_action], dim=-1)

        pe = self.pos(index)
        x = torch.concat([x, pe], dim=-1)
        output, hn, attn_weights_list = self.attn(x,
                                                  query_length,
                                                  hidden_state,
                                                  is_prev_hidden_state,
                                                  padding_mask)

        return output, hn, attn_weights_list


class ModelQ(m.ModelQ):
    def _build_model(self):
        return super()._build_model(c_dense_n=64, c_dense_depth=1)


class ModelPolicy(m.ModelPolicy):
    def _build_model(self):
        return super()._build_model(c_dense_n=64, c_dense_depth=1)
