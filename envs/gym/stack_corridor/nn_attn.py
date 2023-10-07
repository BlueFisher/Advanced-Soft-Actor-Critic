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

        self.attn = m.EpisodeMultiheadAttention(embed_size, 2,
                                                num_layers=2,
                                                use_residual=True,
                                                use_gated=False,
                                                use_layer_norm=False)

        self.mlp = m.LinearLayers(embed_size, output_size=embed_size)

    def forward(self, index, obs_list, pre_action=None,
                seq_q_len=1,
                hidden_state=None,
                is_prev_hidden_state=False,
                query_only_attend_to_rest_key=False,
                padding_mask=None):
        vec_obs = obs_list[0]
        vec_obs = vec_obs.reshape(*vec_obs.shape[:-3], 2 * 3 * 4)

        # pe = self.pe(index)
        # vec_obs = torch.concat([vec_obs, pe], dim=-1)

        output, hn, attn_weights_list = self.attn(vec_obs,
                                                  seq_q_len=seq_q_len,
                                                  hidden_state=hidden_state,
                                                  is_prev_hidden_state=is_prev_hidden_state,

                                                  query_only_attend_to_rest_key=query_only_attend_to_rest_key,
                                                  key_index=index,
                                                  key_padding_mask=padding_mask)

        return output, hn, attn_weights_list


class ModelOptionRep(m.ModelBaseRNNRep):
    def _build_model(self):
        assert self.obs_shapes[1] == (2, 3, 4)

        embed_size = 2 * 3 * 4

        self.rnn = m.GRU(embed_size * 2, 64, 1)

    def forward(self, obs_list, pre_action, rnn_state=None, padding_mask=None):
        high_state, vec_obs = obs_list
        vec_obs = vec_obs.reshape(*vec_obs.shape[:-3], 2 * 3 * 4)

        if padding_mask is not None:
            high_state = high_state * (~padding_mask).to(vec_obs.dtype).unsqueeze(-1)
            vec_obs = vec_obs * (~padding_mask).to(vec_obs.dtype).unsqueeze(-1)

        output, hn = self.rnn(torch.concat([high_state, vec_obs], dim=-1),
                              rnn_state)

        return output, hn


# class ModelOptionRep(m.ModelBaseRNNRep):
#     def _build_model(self):
#         assert self.obs_shapes[1] == (2, 3, 4)

#         self.rnn = nn.GRUCell(2 * 3 * 4, 64)

#     def forward(self, obs_list, pre_action, rnn_state=None, padding_mask=None):
#         high_state, vec_obs = obs_list
#         vec_obs = vec_obs.reshape(*vec_obs.shape[:-3], 2 * 3 * 4)

#         if padding_mask is not None:
#             vec_obs = vec_obs * (~padding_mask).to(vec_obs.dtype).unsqueeze(-1)

#         output = []

#         for t in range(vec_obs.shape[1]):
#             if padding_mask is not None:
#                 mask = padding_mask[:, t]
#                 rnn_state = rnn_state.clone()
#                 rnn_state[mask] = 0.

#             rnn_state = self.rnn(vec_obs[:, t], rnn_state)

#             output.append(rnn_state)

#         output = torch.stack(output, dim=1)

#         state = torch.cat([high_state, output], dim=-1)

#         return state, rnn_state


class ModelQ(m.ModelQ):
    def _build_model(self):
        return super()._build_model(d_dense_n=128, d_dense_depth=2)


class ModelPolicy(m.ModelPolicy):
    def _build_model(self):
        return super()._build_model(d_dense_n=128, d_dense_depth=2)


class ModelRND(m.ModelRND):
    def _build_model(self):
        return super()._build_model(dense_n=128, dense_depth=2, output_size=128)
