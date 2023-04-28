import torch

import algorithm.nn_models as m


class ModelRep(m.ModelBaseAttentionRep):
    def _build_model(self, pe: str):
        self.pe = pe

        embed_dim = self.obs_shapes[0][0]

        self.mlp = m.LinearLayers(embed_dim, output_size=8)

        if pe == 'cat':
            self.pos = m.AbsolutePositionalEncoding(8)
            self.attn = m.EpisodeMultiheadAttention(8 * 2, 2, num_layers=2)
        elif pe == 'add':
            self.pos = m.AbsolutePositionalEncoding(8)
            self.attn = m.EpisodeMultiheadAttention(8, 2, num_layers=2)
        else:
            self.attn = m.EpisodeMultiheadAttention(8, 2, num_layers=2)

    def forward(self, index, obs_list,
                pre_action=None,
                query_length=1,
                hidden_state=None,
                is_prev_hidden_state=False,
                padding_mask=None):

        x = torch.concat([obs_list[0]], dim=-1)
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
                                                  padding_mask)

        return output, hn, attn_weights_list


class ModelOptionRep(m.ModelBaseRNNRep):
    def _build_model(self):
        self.rnn = m.GRU(self.obs_shapes[0][0] + self.obs_shapes[1][0] + sum(self.d_action_sizes) + self.c_action_size, 8, 2)

    def forward(self, obs_list, pre_action, rnn_state=None, padding_mask=None):
        high_state, obs = obs_list

        state, hn = self.rnn(torch.cat([high_state, obs, pre_action], dim=-1), rnn_state)

        return state, hn


ModelQ = m.ModelQ
ModelPolicy = m.ModelPolicy
