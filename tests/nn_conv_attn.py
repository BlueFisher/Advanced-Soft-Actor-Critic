import torch
from torch import nn

import algorithm.nn_models as m

EXTRA_SIZE = 3


class ModelRep(m.ModelBaseAttentionRep):
    def _build_model(self):
        self.conv = m.ConvLayers(30, 30, 3, 'simple', out_dense_depth=2, output_size=8)

        embed_dim = self.conv.output_size

        self.attn = m.EpisodeMultiheadAttention(embed_dim)

        self.dense = nn.Sequential(
            nn.Linear(embed_dim, 8),
            nn.Tanh()
        )

    def forward(self,
                seq_q_len: int,
                index: torch.Tensor,
                obs_list: list[torch.Tensor],
                pre_action: torch.Tensor,
                pre_seq_hidden_state: torch.Tensor | None,
                is_prev_hidden_state=False,
                query_only_attend_to_rest_key=False,
                padding_mask: torch.Tensor | None = None):
        if self.obs_names[0] == '_OPTION_INDEX':
            _, obs_vec, obs_vis = obs_list
        else:
            obs_vec, obs_vis = obs_list

        obs_vec = obs_vec[..., EXTRA_SIZE:]

        vis = self.conv(obs_vis)

        state, hn, attn_weights_list = self.attn(vis,
                                                 seq_q_len=seq_q_len,
                                                 hidden_state=pre_seq_hidden_state,
                                                 is_prev_hidden_state=is_prev_hidden_state,

                                                 query_only_attend_to_rest_key=query_only_attend_to_rest_key,
                                                 key_index=index,
                                                 key_padding_mask=padding_mask)

        state = self.dense(state)

        return state, hn, attn_weights_list

    def get_augmented_encoders(self, obs_list):
        obs_vec, obs_vis = obs_list

        vis_encoder = self.conv(obs_vis)

        return vis_encoder

    def get_state_from_encoders(self,
                                seq_q_len: int,
                                encoders: torch.Tensor | tuple[torch.Tensor],
                                index: torch.Tensor,
                                obs_list: list[torch.Tensor],
                                pre_action: torch.Tensor,
                                pre_seq_hidden_state: torch.Tensor | None,
                                is_prev_hidden_state=False,
                                query_only_attend_to_rest_key=False,
                                padding_mask: torch.Tensor | None = None):
        if self.obs_names[0] == '_OPTION_INDEX':
            _, obs_vec, obs_vis = obs_list
        else:
            obs_vec, obs_vis = obs_list

        obs_vec = obs_vec[..., EXTRA_SIZE:]

        vis_encoder = encoders

        state, hn, attn_weights_list = self.attn(vis_encoder,
                                                 seq_q_len=seq_q_len,
                                                 hidden_state=pre_seq_hidden_state,
                                                 is_prev_hidden_state=is_prev_hidden_state,

                                                 query_only_attend_to_rest_key=query_only_attend_to_rest_key,
                                                 key_index=index,
                                                 key_padding_mask=padding_mask)

        state = self.dense(state)

        return state


ModelQ = m.ModelQ
ModelPolicy = m.ModelPolicy
ModelForwardDynamic = m.ModelForwardDynamic
ModelRND = m.ModelRND
ModelRepProjection = m.ModelRepProjection
ModelRepPrediction = m.ModelRepPrediction
