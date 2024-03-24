import torch

import algorithm.nn_models as m

ModelVOverOption = m.ModelVOverOption
ModelTermination = m.ModelTermination

# from nn_attn import ModelOptionRep


class ModelRep(m.ModelBaseAttentionRep):
    def _build_model(self):
        if self.use_dilation:
            embed_dim = self.obs_shapes[0][0]
        else:
            embed_dim = self.obs_shapes[0][0] + self.c_action_size + sum(self.d_action_sizes)

        self.attn = m.EpisodeMultiheadAttention(embed_dim)

    def forward(self, index, obs_list,
                pre_action=None,
                seq_q_len=1,
                hidden_state=None,
                is_prev_hidden_state=False,
                query_only_attend_to_rest_key=False,
                padding_mask=None):

        if self.use_dilation:
            x = torch.concat([obs_list[0]], dim=-1)
        else:
            x = torch.concat([obs_list[0], pre_action], dim=-1)

        output, hn, attn_weights_list = self.attn(x,
                                                  seq_q_len=seq_q_len,
                                                  hidden_state=hidden_state,
                                                  is_prev_hidden_state=is_prev_hidden_state,

                                                  query_only_attend_to_rest_key=query_only_attend_to_rest_key,
                                                  key_index=index,
                                                  key_padding_mask=padding_mask)

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
