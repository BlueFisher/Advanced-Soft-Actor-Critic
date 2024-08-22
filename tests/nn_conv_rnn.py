import torch
from torch import nn

import algorithm.nn_models as m

EXTRA_SIZE = 3


class ModelRep(m.ModelBaseRep):
    def _build_model(self):
        self.conv = m.ConvLayers(30, 30, 3, 'simple', out_dense_depth=2, output_size=8)

        self.rnn = m.GRU(self.conv.output_size, 8, 2)

        self.dense = nn.Sequential(
            nn.Linear(self.obs_shapes[0][0] - EXTRA_SIZE + 8, 8),
            nn.Tanh()
        )

    def forward(self,
                obs_list: list[torch.Tensor],
                pre_action: torch.Tensor,
                pre_seq_hidden_state: torch.Tensor | None,
                padding_mask: torch.Tensor | None = None):
        obs_vec, obs_vis = obs_list
        obs_vec = obs_vec[..., EXTRA_SIZE:]

        vis = self.conv(obs_vis)

        if pre_seq_hidden_state is not None:
            pre_seq_hidden_state = pre_seq_hidden_state[:, 0]
        state, hn = self.rnn(vis, pre_seq_hidden_state)

        state = self.dense(torch.cat([obs_vec, state], dim=-1))

        return state, hn

    def get_augmented_encoders(self, obs_list):
        obs_vec, obs_vis = obs_list

        vis_encoder = self.conv(obs_vis)

        return vis_encoder

    def get_state_from_encoders(self,
                                encoders: torch.Tensor | tuple[torch.Tensor],
                                obs_list: list[torch.Tensor],
                                pre_action: torch.Tensor,
                                pre_seq_hidden_state: torch.Tensor | None,
                                padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        obs_vec, obs_vis = obs_list
        obs_vec = obs_vec[..., EXTRA_SIZE:]

        vis_encoder = encoders

        state, _ = self.rnn(torch.cat([vis_encoder], dim=-1), pre_seq_hidden_state)

        state = self.dense(torch.cat([obs_vec, state], dim=-1))

        return state


ModelQ = m.ModelQ
ModelPolicy = m.ModelPolicy
ModelForwardDynamic = m.ModelForwardDynamic
ModelRND = m.ModelRND
ModelRepProjection = m.ModelRepProjection
ModelRepPrediction = m.ModelRepPrediction
