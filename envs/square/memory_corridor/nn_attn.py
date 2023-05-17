import torch

import algorithm.nn_models as m


class ModelRep(m.ModelBaseAttentionRep):
    def _build_model(self):
        assert self.obs_shapes[0] == (2, 7)
        assert self.obs_shapes[1] == (4,)

        self.bbox_mlp = m.LinearLayers(2 * 7, output_size=16)

        if self.use_dilation:
            self.mlp = m.LinearLayers(16 + self.obs_shapes[1][0], 64, 1)
        else:
            self.mlp = m.LinearLayers(16 + self.obs_shapes[1][0] + self.c_action_size, 64, 1)
        self.attn = m.EpisodeMultiheadAttention(64, 8, num_layers=2)
        self.pos = m.AbsolutePositionalEncoding(64)

    def forward(self, index, obs_list, pre_action=None,
                query_length=1,
                hidden_state=None,
                is_prev_hidden_state=False,
                query_only_attend_to_reset_key=False,
                padding_mask=None):
        bbox_obs, vec_obs = obs_list

        bbox_obs = bbox_obs.reshape(*bbox_obs.shape[:-2], -1)
        bbox_obs = self.bbox_mlp(bbox_obs)

        if self.use_dilation:
            x = self.mlp(torch.cat([bbox_obs, vec_obs], dim=-1))
        else:
            x = self.mlp(torch.cat([bbox_obs, vec_obs, pre_action], dim=-1))
        pe = self.pos(index)
        x = x + pe

        output, hn, attn_weights_list = self.attn(x,
                                                  query_length,
                                                  hidden_state,
                                                  is_prev_hidden_state,
                                                  query_only_attend_to_reset_key,
                                                  padding_mask)

        return output, hn, attn_weights_list


class ModelOptionRep(m.ModelBaseRNNRep):
    def _build_model(self):
        assert self.obs_shapes[0] == (64,)
        assert self.obs_shapes[1] == (2, 7)
        assert self.obs_shapes[2] == (4,)

        self.bbox_mlp = m.LinearLayers(2 * 7, output_size=16)

        self.rnn = m.GRU(self.obs_shapes[0][0] + 16 + self.c_action_size, 64, 1)

    def forward(self, obs_list, pre_action, rnn_state=None, padding_mask=None):
        high_state, bbox_obs, vec_obs = obs_list

        bbox_obs = bbox_obs.reshape(*bbox_obs.shape[:-2], -1)
        bbox_obs = self.bbox_mlp(bbox_obs)

        output, hn = self.rnn(torch.cat([high_state, bbox_obs, pre_action], dim=-1), rnn_state)

        state = torch.cat([vec_obs, output], dim=-1)

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
