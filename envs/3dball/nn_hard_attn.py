import torch
from torch import nn

import algorithm.nn_models as m

EXTRA_SIZE = 3


class ModelRep(m.ModelBaseAttentionRep):
    def _build_model(self):
        assert self.obs_shapes[0] == (8, )

        embed_dim = 8 - EXTRA_SIZE + 2  # 7
        self.mlp = m.LinearLayers(embed_dim, output_size=32)
        self.attn = m.EpisodeMultiheadAttention(32, 2, num_layers=2)
        self.pos = m.AbsolutePositionalEncoding(32)

    def forward(self, index, obs_list, pre_action,
                seq_q_len=1,
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
                                                  seq_q_len,
                                                  hidden_state,
                                                  is_prev_hidden_state,
                                                  query_only_attend_to_rest_key,
                                                  padding_mask)

        return output, hn, attn_weights_list


class ModelQ(m.ModelQ):
    def _build_model(self):
        return super()._build_model(c_dense_n=64, c_dense_depth=2)


class ModelPolicy(m.ModelPolicy):
    def _build_model(self):
        return super()._build_model(c_dense_n=64, c_dense_depth=2)
