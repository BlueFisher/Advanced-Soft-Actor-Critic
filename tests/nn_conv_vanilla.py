import torch
from torch import nn

import algorithm.nn_models as m


class ModelRep(m.ModelBaseSimpleRep):
    def _build_model(self):
        self.conv = m.ConvLayers(30, 30, 3, 'simple', out_dense_depth=2, output_size=8)

        self.dense = nn.Sequential(
            nn.Linear(self.conv.output_size + self.obs_shapes[0][0], 8),
            nn.Tanh()
        )

    def forward(self, obs_list):
        obs_vec, obs_vis = obs_list

        vis = self.conv(obs_vis)

        state = self.dense(torch.cat([obs_vec, vis], dim=-1))

        return state

    def get_augmented_encoders(self, obs_list):
        obs_vec, obs_vis = obs_list

        vis_encoder = self.conv(obs_vis)

        return vis_encoder

    def get_state_from_encoders(self, obs_list, encoders):
        obs_vec, obs_vis = obs_list

        vis_encoder = encoders

        state = self.dense(torch.cat([obs_vec, vis_encoder], dim=-1))

        return state


ModelQ = m.ModelQ
ModelPolicy = m.ModelPolicy
ModelForwardDynamic = m.ModelForwardDynamic
ModelRND = m.ModelRND
ModelRepProjection = m.ModelRepProjection
ModelRepPrediction = m.ModelRepPrediction
