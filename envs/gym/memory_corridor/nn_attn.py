import torch
from torch import nn

import algorithm.nn_models as m

ModelVOverOption = m.ModelVOverOption
ModelTermination = m.ModelTermination


class ModelRep(m.ModelBaseAttentionRep):
    def _build_model(self):
        assert self.obs_shapes[0] == (4, 3, 4)

        self.ray_mlp = m.LinearLayers(35, output_size=16)

        embed_size = 4 * 3 * 4

        self.pos = m.AbsolutePositionalEncoding(embed_size)
        self.attn = m.EpisodeMultiheadAttention(embed_size, 1, num_layers=2, use_layer_norm=True)
        self.layer_norm = nn.LayerNorm(embed_size)

        self.mlp = m.LinearLayers(embed_size, output_size=embed_size)

    def forward(self, index, obs_list, pre_action=None,
                query_length=1,
                hidden_state=None,
                is_prev_hidden_state=False,
                query_only_attend_to_reset_key=False,
                padding_mask=None):
        vec_obs = obs_list[0]
        vec_obs = vec_obs.reshape(*vec_obs.shape[:-3], 4 * 3 * 4)

        x = vec_obs

        pe = self.pos(index)
        x = x + pe

        output, hn, attn_weights_list = self.attn(x,
                                                  query_length,
                                                  hidden_state,
                                                  is_prev_hidden_state,
                                                  query_only_attend_to_reset_key,
                                                  padding_mask)

        output = output + self.mlp(self.layer_norm(output))

        return output, hn, attn_weights_list


class ModelOptionRep(m.ModelBaseRNNRep):
    def _build_model(self):
        assert self.obs_shapes[1] == (4, 3, 4)

        self.rnn = m.GRU(4 * 3 * 4, 64, 1)

    def forward(self, obs_list, pre_action, rnn_state=None, padding_mask=None):
        high_state, vec_obs = obs_list
        vec_obs = vec_obs.reshape(*vec_obs.shape[:-3], 4 * 3 * 4)

        output, hn = self.rnn(vec_obs, rnn_state)

        state = torch.cat([high_state, output], dim=-1)

        return state, hn


class ModelQ(m.ModelQ):
    def _build_model(self):
        return super()._build_model(d_dense_n=128, d_dense_depth=2)


class ModelPolicy(m.ModelPolicy):
    def _build_model(self):
        return super()._build_model(d_dense_n=128, d_dense_depth=2)


class ModelRND(m.ModelRND):
    def _build_model(self):
        return super()._build_model(dense_n=128, dense_depth=2, output_size=128)