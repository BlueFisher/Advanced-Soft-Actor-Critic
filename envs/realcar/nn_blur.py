import torch
from torchvision import transforms

from algorithm.nn_models.layers import Transform
from envs.realcar.nn import *


class ModelRep(ModelRep):
    def _build_model(self):
        super()._build_model()

        self.blurrer = Transform(transforms.GaussianBlur(9, sigma=9))

    def forward(self, obs_list, pre_action, rnn_state=None):
        vis_cam, ray, vec = obs_list
        vec = vec[..., :-EXTRA_SIZE]

        vis_cam = self.blurrer(vis_cam)
        vis = self.conv(vis_cam)

        state, hn = self.rnn(torch.cat([self.dense(vis), pre_action], dim=-1), rnn_state)

        state = torch.cat([state, ray, vec], dim=-1)

        return state, hn
