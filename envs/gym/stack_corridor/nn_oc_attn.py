import torch
from torch import nn

import algorithm.nn_models as m
from algorithm.nn_models.layers.seq_layers import GATE, POSITIONAL_ENCODING


class ModelRep(m.ModelBaseAttentionRep):
    def _build_model(self):
        assert self.obs_shapes[0] == (3 + 1, )
        assert self.obs_shapes[1] == (3, 4)

        embed_size = 3 + 1 + 3 * 4  # 16
        self.mlp = m.LinearLayers(embed_size, dense_n=embed_size, dense_depth=2,
                                  dropout=0.01)

        self.attn = m.EpisodeMultiheadAttention(embed_size, num_layers=4,
                                                num_heads=1,
                                                pe=POSITIONAL_ENCODING.ROPE,
                                                dropout=0.01,
                                                gate=GATE.CAT,
                                                use_layer_norm=False)

    def forward(self, index, obs_list, pre_action=None,
                seq_q_len=1,
                hidden_state=None,
                is_prev_hidden_state=False,
                query_only_attend_to_rest_key=False,
                padding_mask=None):
        if self._offline_action_index != -1:
            option_index, vec_obs, _ = obs_list
        else:
            option_index, vec_obs = obs_list

        vec_obs = vec_obs.reshape(*vec_obs.shape[:-2], 3 * 4)
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


class ModelOptionRep(m.ModelBaseSimpleRep):
    def _build_model(self):
        assert self.obs_shapes[1] == (3, 4)

        embed_size = 3 * 4

        self.mlp = m.LinearLayers(embed_size, dense_n=embed_size, dense_depth=1)

    def forward(self, obs_list):
        if self._offline_action_index != -1:
            high_state, vec_obs, _ = obs_list
        else:
            high_state, vec_obs = obs_list

        vec_obs = vec_obs.reshape(*vec_obs.shape[:-2], 3 * 4)

        output = torch.concat([high_state, self.mlp(vec_obs)], dim=-1)

        return output


class ModelQ(m.ModelQ):
    def _build_model(self):
        return super()._build_model(d_dense_n=64, d_dense_depth=2, dropout=0.01)


class ModelTermination(m.ModelTermination):
    def _build_model(self):
        return super()._build_model(dense_n=64, dense_depth=2, dropout=0.01)

    def forward(self, state):
        l = self.dense(state)
        return torch.sigmoid(l)


class ModelPolicy(m.ModelPolicy):
    def _build_model(self):
        return super()._build_model(d_dense_n=32, d_dense_depth=1)


class ModelRND(m.ModelRND):
    def _build_model(self):
        return super()._build_model(dense_n=32, dense_depth=2, output_size=32)
