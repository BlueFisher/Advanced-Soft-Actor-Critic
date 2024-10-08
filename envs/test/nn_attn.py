import torch

import algorithm.nn_models as m


ModelTermination = m.ModelTermination


class ModelRep(m.ModelBaseAttentionRep):
    def _build_model(self):
        embed_dim = self.obs_shapes[0][0] + self.c_action_size + sum(self.d_action_sizes)

        self.attn = m.EpisodeMultiheadAttention(embed_dim)

    def forward(self,
                seq_q_len: int,
                index: torch.Tensor,
                obs_list: list[torch.Tensor],
                pre_action: torch.Tensor,
                pre_seq_hidden_state: torch.Tensor | None,
                is_prev_hidden_state=False,
                query_only_attend_to_rest_key=False,
                padding_mask: torch.Tensor | None = None):

        x = torch.concat([obs_list[0], pre_action], dim=-1)

        output, hn, attn_weights_list = self.attn(x,
                                                  seq_q_len=seq_q_len,
                                                  hidden_state=pre_seq_hidden_state,
                                                  is_prev_hidden_state=is_prev_hidden_state,

                                                  query_only_attend_to_rest_key=query_only_attend_to_rest_key,
                                                  key_index=index,
                                                  key_padding_mask=padding_mask)

        return output, hn, attn_weights_list


class ModelOptionRep(m.ModelBaseRep):
    def forward(self,
                obs_list: list[torch.Tensor],
                pre_action: torch.Tensor,
                pre_seq_hidden_state: torch.Tensor | None,
                padding_mask: torch.Tensor | None = None):
        high_state, vec_obs = obs_list

        output = torch.concat([high_state, vec_obs], dim=-1)

        return output


ModelQ = m.ModelQ
ModelPolicy = m.ModelPolicy
