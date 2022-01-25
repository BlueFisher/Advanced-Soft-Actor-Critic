import math

import torch
from torch import nn

import algorithm.nn_models as m

EXTRA_SIZE = 4


class ModelRep(m.ModelBaseAttentionRep):
    def _build_model(self):
        assert self.obs_shapes[0] == (44,)
        assert self.obs_shapes[1] == (8,)

        embed_dim = self.obs_shapes[1][0] - EXTRA_SIZE + self.c_action_size
        self.pos = m.AbsolutePositionalEncoding(embed_dim)
        self.attn = m.EpisodeMultiheadAttention(embed_dim, 2, num_layers=2)

    def forward(self, index, obs_list, pre_action,
                query_length=1,
                hidden_state=None,
                is_prev_hidden_state=False,
                padding_mask=None):

        ray_obs, vec_obs = obs_list
        vec_obs = vec_obs[..., :-EXTRA_SIZE]

        x = torch.concat([vec_obs, pre_action], dim=-1)
        pe = self.pos(index)
        x = x + pe
        output, hn, attn_weights_list = self.attn(x,
                                                  query_length,
                                                  hidden_state,
                                                  is_prev_hidden_state,
                                                  padding_mask)

        return torch.concat([output, ray_obs[:, -query_length:]], dim=-1), hn, attn_weights_list


class ModelRND(m.ModelRND):
    def _build_model(self):
        return super()._build_model(dense_n=128, dense_depth=2, output_size=128)


class ModelQ(m.ModelQ):
    def _build_model(self):
        return super()._build_model(c_state_n=128, c_state_depth=1,
                                    c_action_n=128, c_action_depth=1,
                                    c_dense_n=128, c_dense_depth=3)


class ModelPolicy(m.ModelPolicy):
    def _build_model(self):
        return super()._build_model(c_dense_n=128, c_dense_depth=2,
                                    mean_n=128, mean_depth=1,
                                    logstd_n=128, logstd_depth=1)
