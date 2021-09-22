import torch
from torchvision import transforms

from algorithm.nn_models.layers import Transform
from envs.realcar.nn import *
from algorithm.utils.image_visual import ImageVisual


class ModelRep(ModelRep):
    def _build_model(self):
        super()._build_model()

        self.blurrer = Transform(transforms.GaussianBlur(9, sigma=9))
        self.image_visual = ImageVisual()

    def forward(self, obs_list, pre_action, rnn_state=None):
        vis_bounding, vis_ori, ray, vis_seg, vec = obs_list
        vec = vec[..., :-EXTRA_SIZE]

        vis_ori = self.blurrer(vis_ori)
        self.image_visual.show(vis_ori[:, 0, ...].cpu().numpy())

        vis = self.conv(torch.cat([vis_bounding, vis_ori, vis_seg], dim=-1))

        state = self.dense(torch.cat([vis, ray, vec], dim=-1))

        state, hn = self.rnn(torch.cat([state, pre_action], dim=-1), rnn_state)

        return state, hn
