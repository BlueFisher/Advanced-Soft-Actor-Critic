import torch

import algorithm.nn_models as m


class ModelRep(m.ModelBaseOptionSelectorAttentionRep):
    def _build_model(self):
        embed_dim = self.obs_shapes[0][0]

        self.attn = m.EpisodeMultiheadAttention(embed_dim)

    def forward(self,
                seq_q_len: int,
                index: torch.Tensor,
                obs_list: list[torch.Tensor],
                pre_action: torch.Tensor,
                pre_seq_hidden_state: torch.Tensor | None,
                pre_termination_mask: torch.Tensor | None = None,
                is_prev_hidden_state=False,
                query_only_attend_to_rest_key=False,
                padding_mask: torch.Tensor | None = None):

        if pre_action is not None:
            assert index.shape[1] == obs_list[0].shape[1] == pre_action.shape[1]

        x = obs_list[0]

        output, hn, attn_weights_list = self.attn(x,
                                                  seq_q_len=seq_q_len,
                                                  hidden_state=pre_seq_hidden_state,
                                                  is_prev_hidden_state=is_prev_hidden_state,

                                                  query_only_attend_to_rest_key=query_only_attend_to_rest_key,
                                                  key_index=index,
                                                  key_padding_mask=padding_mask)

        return output, hn, attn_weights_list


ModelOptionRep = m.ModelSimpleRep


ModelQ = m.ModelQ
ModelPolicy = m.ModelPolicy

ModelTermination = m.ModelTermination
