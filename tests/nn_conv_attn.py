import torch
from torch import nn

import algorithm.nn_models as m

from .nn_conv_rnn import ModelOptionRep

EXTRA_SIZE = 3


class ModelRep(m.ModelBaseAttentionRep):
    def _build_model(self):
        self.conv = m.ConvLayers(30, 30, 3, 'simple', out_dense_depth=2, output_size=8)

        if self.use_dilation:
            embed_dim = self.conv.output_size
        else:
            embed_dim = self.conv.output_size + sum(self.d_action_sizes) + self.c_action_size

        self.attn = m.EpisodeMultiheadAttention(embed_dim, 1, num_layers=3)

        self.dense = nn.Sequential(
            nn.Linear(self.obs_shapes[0][0] - EXTRA_SIZE + embed_dim, 8),
            nn.Tanh()
        )

    def forward(self, index, obs_list, pre_action,
                query_length=1,
                hidden_state=None,
                is_prev_hidden_state=False,
                padding_mask=None):
        obs_vec, obs_vis = obs_list
        obs_vec = obs_vec[..., EXTRA_SIZE:]

        vis = self.conv(obs_vis)

        if self.use_dilation:
            state, hn, attn_weights_list = self.attn(vis,
                                                     query_length,
                                                     hidden_state,
                                                     is_prev_hidden_state,
                                                     padding_mask)
        else:
            state, hn, attn_weights_list = self.attn(torch.cat([vis, pre_action], dim=-1),
                                                     query_length,
                                                     hidden_state,
                                                     is_prev_hidden_state,
                                                     padding_mask)

        state = self.dense(torch.cat([obs_vec[:, -query_length:], state], dim=-1))

        return state, hn, attn_weights_list

    def get_augmented_encoders(self, obs_list):
        obs_vec, obs_vis = obs_list

        vis_encoder = self.conv(obs_vis)

        return vis_encoder

    def get_state_from_encoders(self, index, obs_list, encoders, pre_action,
                                query_length=1,
                                hidden_state=None,
                                is_prev_hidden_state=False,
                                padding_mask=None):
        obs_vec, obs_vis = obs_list
        obs_vec = obs_vec[..., EXTRA_SIZE:]

        vis_encoder = encoders

        if self.use_dilation:
            state, *_ = self.attn(vis_encoder,
                                  query_length,
                                  hidden_state,
                                  is_prev_hidden_state,
                                  padding_mask)
        else:
            state, *_ = self.attn(torch.cat([vis_encoder, pre_action], dim=-1),
                                  query_length,
                                  hidden_state,
                                  is_prev_hidden_state,
                                  padding_mask)

        state = self.dense(torch.cat([obs_vec, state], dim=-1))

        return state


class ModelTransition(m.ModelTransition):
    def _build_model(self):
        return super()._build_model(extra_size=EXTRA_SIZE if self.use_extra_data else 0)

    def extra_obs(self, obs_list):
        return obs_list[0][..., :EXTRA_SIZE]


ModelReward = m.ModelReward


class ModelObservation(m.ModelBaseObservation):
    def _build_model(self):
        self.dense = m.LinearLayers(self.state_size,
                                    8, 2,
                                    self.obs_shapes[0][0] if self.use_extra_data else self.obs_shapes[0][0] - EXTRA_SIZE)

    def forward(self, state):
        return self.dense(state)

    def get_loss(self, state, obs_list):
        mse = torch.nn.MSELoss()

        return mse(self(state), obs_list[0] if self.use_extra_data else obs_list[0][..., EXTRA_SIZE:])


ModelQ = m.ModelQ
ModelPolicy = m.ModelPolicy
ModelForwardDynamic = m.ModelForwardDynamic
ModelRND = m.ModelRND
ModelRepProjection = m.ModelRepProjection
ModelRepPrediction = m.ModelRepPrediction
