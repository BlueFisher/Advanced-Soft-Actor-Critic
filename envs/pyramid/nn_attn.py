import torch

import algorithm.nn_models as m

EXTRA_SIZE = 6


class ModelRep(m.ModelBaseAttentionRep):
    def _build_model(self, pe: str):
        assert self.obs_shapes[0] == (55,)
        assert self.obs_shapes[1] == (11,)

        self.pe = pe

        embed_dim = self.obs_shapes[0][0] + self.obs_shapes[1][0] - EXTRA_SIZE + self.c_action_size

        self.mlp = m.LinearLayers(embed_dim, output_size=64)

        if pe == 'cat':
            self.pos = m.AbsolutePositionalEncoding(64)
            self.attn = m.EpisodeMultiheadAttention(64 * 2, 2, num_layers=2)
        elif pe == 'add':
            self.pos = m.AbsolutePositionalEncoding(64)
            self.attn = m.EpisodeMultiheadAttention(64, 2, num_layers=2)
        else:
            self.attn = m.EpisodeMultiheadAttention(64, 2, num_layers=2)

    def forward(self, index, obs_list, pre_action,
                query_length=1,
                hidden_state=None,
                is_prev_hidden_state=False,
                query_only_attend_to_reset_key=False,
                padding_mask=None):

        ray_obs, vec_obs = obs_list
        vec_obs = vec_obs[..., :-EXTRA_SIZE]

        x = torch.concat([ray_obs, vec_obs, pre_action], dim=-1)
        x = self.mlp(x)

        if self.pe == 'cat':
            pe = self.pos(index)
            x = torch.concat([x, pe], dim=-1)
        elif self.pe == 'add':
            pe = self.pos(index)
            x = x + pe

        output, hn, attn_weights_list = self.attn(x,
                                                  query_length,
                                                  hidden_state,
                                                  is_prev_hidden_state,
                                                  query_only_attend_to_reset_key,
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
