import torch
from torch import nn

import algorithm.nn_models as m
from algorithm.nn_models.layers.seq_layers import AbsolutePositionalEncoding

ModelVOverOption = m.ModelVOverOption
ModelTermination = m.ModelTermination


class ModelRep(m.ModelBaseAttentionRep):
    def _build_model(self):
        assert self.obs_shapes[0] == (2, 3, 4)

        embed_size = 2 * 3 * 4

        # self.pe = AbsolutePositionalEncoding(embed_size, 1000)

        self.attn = m.EpisodeMultiheadAttention(embed_size, num_layers=3,
                                                num_heads=[1, 4, 8],
                                                use_residual=False,
                                                use_gated=False,
                                                use_layer_norm=False)

    def forward(self, index, obs_list, pre_action=None,
                seq_q_len=1,
                hidden_state=None,
                is_prev_hidden_state=False,
                query_only_attend_to_rest_key=False,
                padding_mask=None):
        vec_obs = obs_list[0]
        vec_obs = vec_obs.reshape(*vec_obs.shape[:-3], 2 * 3 * 4)

        output, hn, attn_weights_list = self.attn(vec_obs,
                                                  seq_q_len=seq_q_len,
                                                  hidden_state=hidden_state,
                                                  is_prev_hidden_state=is_prev_hidden_state,

                                                  query_only_attend_to_rest_key=query_only_attend_to_rest_key,
                                                  key_index=index,
                                                  key_padding_mask=padding_mask)

        return output, hn, attn_weights_list


class ModelOptionRep(m.ModelBaseSimpleRep):
    def _build_model(self):
        assert self.obs_shapes[1] == (2, 3, 4)

        embed_size = 2 * 3 * 4

        self.mlp = m.LinearLayers(embed_size, dense_depth=2, output_size=embed_size * 2)

    def forward(self, obs_list):
        high_state, vec_obs = obs_list
        vec_obs = vec_obs.reshape(*vec_obs.shape[:-3], 2 * 3 * 4)

        output = torch.concat([high_state, self.mlp(vec_obs)], dim=-1)

        return output


class ModelQ(m.ModelQ):
    def _build_model(self):
        return super()._build_model(d_dense_n=128, d_dense_depth=2)


class ModelPolicy(m.ModelPolicy):
    def _build_model(self):
        return super()._build_model(d_dense_n=128, d_dense_depth=2)


class ModelRND(m.ModelRND):
    def _build_model(self):
        return super()._build_model(dense_n=128, dense_depth=2, output_size=128)
