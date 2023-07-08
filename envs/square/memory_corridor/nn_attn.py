import torch
from torch import nn

import algorithm.nn_models as m

ModelVOverOption = m.ModelVOverOption
ModelTermination = m.ModelTermination


class ModelRep(m.ModelBaseAttentionRep):
    def _build_model(self):
        assert self.obs_shapes[0] == (35,)
        assert self.obs_shapes[1] == (5,)

        self.ray_mlp = m.LinearLayers(35, output_size=16)

        if self.use_dilation:
            embed_size = 16
        else:
            embed_size = 16 + 6 + self.c_action_size

        self.pos = m.AbsolutePositionalEncoding(embed_size)
        self.attn = m.EpisodeMultiheadAttention(embed_size, 2, num_layers=2, use_layer_norm=True)
        self.layer_norm = nn.LayerNorm(embed_size)

        self.mlp = m.LinearLayers(embed_size, output_size=embed_size)

    def forward(self, index, obs_list, pre_action=None,
                query_length=1,
                hidden_state=None,
                is_prev_hidden_state=False,
                query_only_attend_to_rest_key=False,
                padding_mask=None):
        ray_obs, vec_obs = obs_list
        vec_obs = torch.concat([vec_obs, torch.zeros_like(vec_obs)[..., :1]], dim=-1)

        ray_obs = self.ray_mlp(ray_obs)

        if self.use_dilation:
            x = ray_obs
        else:
            x = torch.cat([ray_obs, vec_obs, pre_action], dim=-1)

        pe = self.pos(index)
        x = x + pe

        output, hn, attn_weights_list = self.attn(x,
                                                  query_length,
                                                  hidden_state,
                                                  is_prev_hidden_state,
                                                  query_only_attend_to_rest_key,
                                                  padding_mask)

        output = output + self.mlp(self.layer_norm(output))

        return output, hn, attn_weights_list


class ModelOptionRep(m.ModelBaseRNNRep):
    def _build_model(self):
        # assert self.obs_shapes[0] == embed_size
        assert self.obs_shapes[1] == (35,)
        assert self.obs_shapes[2] == (5,)

        self.ray_mlp = m.LinearLayers(35, output_size=16)

        embed_size = 16 + 5 + self.c_action_size

        self.rnn = m.GRU(embed_size, embed_size, 1)

    def forward(self, obs_list, pre_action, rnn_state=None, padding_mask=None):
        high_state, ray_obs, vec_obs = obs_list

        ray_obs = self.ray_mlp(ray_obs)

        output, hn = self.rnn(torch.cat([ray_obs, vec_obs, pre_action], dim=-1), rnn_state)

        state = torch.cat([high_state, output], dim=-1)

        return state, hn


class ModelQ(m.ModelQ):
    def _build_model(self):
        return super()._build_model(c_state_n=64, c_state_depth=1,
                                    c_action_n=64, c_action_depth=1,
                                    c_dense_n=64, c_dense_depth=1)


class ModelPolicy(m.ModelPolicy):
    def _build_model(self):
        return super()._build_model(c_dense_n=64, c_dense_depth=1,
                                    mean_n=64, mean_depth=1,
                                    logstd_n=64, logstd_depth=1)


class ModelRND(m.ModelRND):
    def _build_model(self):
        return super()._build_model(dense_n=128, dense_depth=2, output_size=128)
