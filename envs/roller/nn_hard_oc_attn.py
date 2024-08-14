import torch
from torch import nn

import algorithm.nn_models as m
from algorithm.nn_models.layers.seq_layers import POSITIONAL_ENCODING


ModelTermination = m.ModelTermination

EXTRA_SIZE = 2


class ModelRep(m.ModelBaseAttentionRep):
    def _build_model(self):
        assert self.obs_shapes[0] == (2 + 1, )
        assert self.obs_shapes[1] == (6, )

        embed_dim = 2 + 1 + 6 - EXTRA_SIZE

        self.mlp = m.LinearLayers(embed_dim, output_size=8)

        self.attn = m.EpisodeMultiheadAttention(8, num_layers=2,
                                                num_heads=1,
                                                pe=POSITIONAL_ENCODING.ABSOLUTE_CAT,
                                                use_residual=True,
                                                use_gated=True,
                                                use_layer_norm=False)

    def forward(self, index, obs_list, pre_action=None,
                seq_q_len=1,
                hidden_state=None,
                is_prev_hidden_state=False,
                query_only_attend_to_rest_key=False,
                padding_mask=None):
        option_index, vec_obs = obs_list
        vec_obs = vec_obs[..., :-EXTRA_SIZE]

        vec_obs = torch.concat([option_index, vec_obs], dim=-1)
        vec_obs = self.mlp(vec_obs)

        output, hn, attn_weights_list = self.attn(vec_obs,
                                                  seq_q_len=seq_q_len,
                                                  hidden_state=hidden_state,
                                                  is_prev_hidden_state=is_prev_hidden_state,

                                                  query_only_attend_to_rest_key=query_only_attend_to_rest_key,
                                                  key_index=index,
                                                  key_padding_mask=padding_mask)

        return output, hn, attn_weights_list


class ModelOptionRep(m.ModelBaseRep):
    def _build_model(self):
        assert self.obs_shapes[1] == (6, )

    def forward(self, obs_list):
        high_state, vec_obs = obs_list
        vec_obs = vec_obs[..., :-EXTRA_SIZE]

        output = torch.concat([high_state, vec_obs], dim=-1)

        return output


class ModelQ(m.ModelQ):
    def _build_model(self):
        return super()._build_model(c_dense_n=64, c_dense_depth=2)


class ModelPolicy(m.ModelPolicy):
    def _build_model(self):
        return super()._build_model(c_dense_n=64, c_dense_depth=2)
