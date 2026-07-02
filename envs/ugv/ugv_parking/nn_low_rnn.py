import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2 as T

import algorithm.nn_models as m
from algorithm.nn_models.layers.seq_layers import POSITIONAL_ENCODING
from algorithm.utils.visualization.image import ImageVisual
from algorithm.utils.visualization.ray import RayVisual

from .obs_preprocessor import NUM_CLASSES

OBS_SHAPES = [(1,), (NUM_CLASSES, 84, 84), (NUM_CLASSES, 84, 84), (802,), (6,)]

RAY_SIZE = 400
AUG_RAY_RANDOM_PROB = 0.4


class ModelRep(m.ModelBaseRep):
    def _build_model(self, ray_random):
        for u_s, s in zip(self.obs_shapes, OBS_SHAPES):
            assert u_s == s, f'{u_s} {s}'

        self._image_visual = ImageVisual()
        self._ray_visual = RayVisual()

        self.ray_random = ray_random

        self.conv_cam_seg = m.ConvLayers(84, 84, NUM_CLASSES, 'simple',
                                         out_dense_n=64, out_dense_depth=2)

        self.conv_third_cam_seg = m.ConvLayers(84, 84, NUM_CLASSES, 'simple',
                                               out_dense_n=64, out_dense_depth=2)

        self.ray_conv = m.Conv1dLayers(RAY_SIZE, 2, 'default',
                                       out_dense_n=64, out_dense_depth=2)

        self.dense = m.LinearLayers(6, output_size=64)

        self.attn = m.MultiheadAttention(64, 8, pe=POSITIONAL_ENCODING.ROPE)

        self.rnn = m.GRU(64 + self.c_action_size, 64, 1)

    def forward(self,
                obs_list: list[torch.Tensor],
                pre_action: torch.Tensor,
                rnn_state: torch.Tensor | None,
                padding_mask: torch.Tensor | None = None):
        llm_obs, vis, vis_third, ray, vec = obs_list

        # remove the center ray
        ray = torch.cat([ray[..., :RAY_SIZE], ray[..., RAY_SIZE + 2:]], dim=-1)
        ray = ray.view(*ray.shape[:-1], RAY_SIZE, 2)

        """ ENCODE """
        vis_encoder = self.conv_cam_seg(vis)
        vis_third_encoder = self.conv_third_cam_seg(vis_third)

        # self._ray_visual(ray, max_batch=3)
        ray_encoder = self.ray_conv(ray)

        x = self.dense(vec)

        sensor_f = torch.cat([vis_encoder.unsqueeze(-2),
                              vis_third_encoder.unsqueeze(-2),
                              ray_encoder.unsqueeze(-2)], dim=-2)
        attn_x, _ = self.attn(x.unsqueeze(-2), sensor_f, sensor_f)
        x = x + attn_x[..., 0, :]

        if rnn_state is not None:
            rnn_state = rnn_state[:, 0]

        output, hn = self.rnn(torch.cat([x, pre_action], dim=-1),
                              rnn_state,
                              padding_mask=padding_mask)

        return output, hn


class ModelQ(m.ModelQ):
    def _build_model(self):
        return super()._build_model(c_dense_n=128, c_dense_depth=2)


class ModelPolicy(m.ModelPolicy):
    def _build_model(self):
        return super()._build_model(c_dense_n=128, c_dense_depth=2)


class ModelRND(m.ModelRND):
    def _build_model(self):
        return super()._build_model(dense_n=128, dense_depth=2, output_size=128)
