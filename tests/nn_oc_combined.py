import torch
from torch import nn

import algorithm.nn_models as m

from .nn_conv_vanilla import ModelRep as ModelVanillaRep
from .nn_conv_rnn import ModelRep as ModelRNNRep
from .nn_conv_attn import ModelRep as ModelAttentionRep


EXTRA_SIZE = 3


class ModelOptionSelectorVanillaRep(ModelVanillaRep, m.ModelBaseOptionSelectorRep):
    def forward(self,
                obs_list: list[torch.Tensor],
                pre_action: torch.Tensor,
                pre_seq_hidden_state: torch.Tensor | None,
                pre_termination_mask: torch.Tensor | None = None,
                padding_mask: torch.Tensor | None = None):
        return super().forward(
            obs_list,
            pre_action,
            pre_seq_hidden_state,
            padding_mask
        )


class ModelOptionSelectorRNNRep(ModelRNNRep, m.ModelBaseOptionSelectorRep):
    def forward(self,
                obs_list: list[torch.Tensor],
                pre_action: torch.Tensor,
                pre_seq_hidden_state: torch.Tensor | None,
                pre_termination_mask: torch.Tensor | None = None,
                padding_mask: torch.Tensor | None = None):
        return super().forward(
            obs_list,
            pre_action,
            pre_seq_hidden_state,
            padding_mask
        )


class ModelOptionSelectorAttentionRep(ModelAttentionRep, m.ModelBaseOptionSelectorAttentionRep):
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
        return super().forward(
            seq_q_len,
            index,
            obs_list,
            pre_action,
            pre_seq_hidden_state,
            is_prev_hidden_state,
            query_only_attend_to_rest_key,
            padding_mask
        )


class ModelVanillaOptionRep(m.ModelBaseRep):
    def _build_model(self):
        self.conv = m.ConvLayers(30, 30, 3, 'simple', out_dense_depth=2, output_size=8)

        self.dense = nn.Sequential(
            nn.Linear(self.conv.output_size + self.obs_shapes[1][0], 8),
            nn.Tanh()
        )

    def forward(self,
                obs_list: list[torch.Tensor],
                pre_action: torch.Tensor,
                pre_seq_hidden_state: torch.Tensor | None,
                padding_mask: torch.Tensor | None = None):
        high_state, obs_vec, obs_vis = obs_list

        vis = self.conv(obs_vis)

        state = self.dense(torch.cat([obs_vec, vis], dim=-1))

        return state, self._get_empty_seq_hidden_state(state)

    def get_augmented_encoders(self, obs_list):
        high_state, obs_vec, obs_vis = obs_list

        vis_encoder = self.conv(obs_vis)

        return vis_encoder

    def get_state_from_encoders(self,
                                encoders: torch.Tensor | tuple[torch.Tensor],
                                obs_list: list[torch.Tensor],
                                pre_action: torch.Tensor,
                                pre_seq_hidden_state: torch.Tensor | None,
                                padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        high_state, obs_vec, obs_vis = obs_list

        vis_encoder = encoders

        state = self.dense(torch.cat([obs_vec, vis_encoder], dim=-1))

        return state


class ModelRNNOptionRep(m.ModelBaseRep):
    def _build_model(self):
        self.conv = m.ConvLayers(30, 30, 3, 'simple', out_dense_depth=2, output_size=8)

        self.rnn = m.GRU(self.conv.output_size + sum(self.d_action_sizes) + self.c_action_size, 8, 2)

        self.dense = nn.Sequential(
            nn.Linear(self.obs_shapes[1][0] - EXTRA_SIZE + 8 + 8, 8),
            nn.Tanh()
        )

    def forward(self,
                obs_list: list[torch.Tensor],
                pre_action: torch.Tensor,
                pre_seq_hidden_state: torch.Tensor | None,
                padding_mask: torch.Tensor | None = None):
        high_state, obs_vec, obs_vis = obs_list
        obs_vec = obs_vec[..., EXTRA_SIZE:]

        vis = self.conv(obs_vis)

        if pre_seq_hidden_state is not None:
            pre_seq_hidden_state = pre_seq_hidden_state[:, 0]
        state, hn = self.rnn(torch.cat([vis, pre_action], dim=-1), pre_seq_hidden_state)

        state = self.dense(torch.cat([obs_vec, state, high_state], dim=-1))

        return state, hn

    def get_augmented_encoders(self, obs_list):
        high_state, obs_vec, obs_vis = obs_list

        vis_encoder = self.conv(obs_vis)

        return vis_encoder

    def get_state_from_encoders(self,
                                encoders: torch.Tensor | tuple[torch.Tensor],
                                obs_list: list[torch.Tensor],
                                pre_action: torch.Tensor,
                                pre_seq_hidden_state: torch.Tensor | None,
                                padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        high_state, obs_vec, obs_vis = obs_list
        obs_vec = obs_vec[..., EXTRA_SIZE:]

        vis_encoder = encoders

        state, _ = self.rnn(torch.cat([vis_encoder, pre_action], dim=-1), pre_seq_hidden_state)

        state = self.dense(torch.cat([obs_vec, state, high_state], dim=-1))

        return state


ModelQ = m.ModelQ
ModelPolicy = m.ModelPolicy
ModelForwardDynamic = m.ModelForwardDynamic
ModelOptionSelectorRND = m.ModelOptionSelectorRND
ModelRND = m.ModelRND
ModelRepProjection = m.ModelRepProjection
ModelRepPrediction = m.ModelRepPrediction
ModelTermination = m.ModelTermination
ModelVOverOptions = m.ModelVOverOptions