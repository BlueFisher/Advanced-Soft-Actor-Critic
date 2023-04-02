import torch
from torchvision import transforms as T

import algorithm.nn_models as m
from algorithm.utils.image_visual import ImageVisual
from algorithm.utils.ray import RayVisual
from algorithm.utils.transform import GaussianNoise, SaltAndPepperNoise

RAY_SIZE = 400
AUG_RAY_RANDOM_SIZE = 250


class ModelRep(m.ModelBaseRNNRep):
    def _build_model(self, ray_random):
        assert self.obs_shapes[0] == (10, 7)  # BoundingBoxSensor
        assert self.obs_shapes[1] == (802,)  # ray (1 + 200 + 200) * 2
        assert self.obs_shapes[2] == (21, 7)  # ThirdPersonBoundingBoxSensor
        assert self.obs_shapes[3] == (6,)  # vector

        self.ray_random = ray_random

        self.bbox_attn = m.MultiheadAttention(7, 1)
        self.bbox_attn_tp = m.MultiheadAttention(7, 1)
        self.bbox_dense = m.LinearLayers(7, output_size=64)
        self.bbox_dense_tp = m.LinearLayers(7, output_size=64)

        self.ray_conv = m.Conv1dLayers(RAY_SIZE, 2, 'default',
                                       out_dense_n=64, out_dense_depth=2)

        self.rnn = m.GRU(64 + self.obs_shapes[3][0] + sum(self.d_action_sizes), 64, 1)

    def _handle_bbox(self, bbox, attn):
        bbox_mask = ~bbox.any(dim=-1)
        bbox_mask[..., 0] = False
        bbox, _ = attn(bbox, bbox, bbox, key_padding_mask=bbox_mask)
        return bbox.mean(-2)

    def forward(self, obs_list, pre_action, rnn_state=None, padding_mask=None):
        bbox_cam, ray, bbox_tp_cam, vec = obs_list
        ray = torch.cat([ray[..., :RAY_SIZE], ray[..., RAY_SIZE + 2:]], dim=-1)

        bbox_cam = self._handle_bbox(bbox_cam, self.bbox_attn)
        bbox_tp_cam = self._handle_bbox(bbox_tp_cam, self.bbox_attn_tp)
        bbox_cam = self.bbox_dense(bbox_cam)
        bbox_tp_cam = self.bbox_dense_tp(bbox_tp_cam)

        ray = ray.view(*ray.shape[:-1], RAY_SIZE, 2)
        random_index = torch.randperm(RAY_SIZE)[:self.ray_random]
        ray[..., random_index, 0] = 1.
        ray[..., random_index, 1] = 1.
        ray = self.ray_conv(ray)

        state, hn = self.rnn(torch.cat([ray + bbox_cam + bbox_tp_cam, vec, pre_action], dim=-1),
                             rnn_state)

        return state, hn


class ModelQ(m.ModelQ):
    def _build_model(self):
        return super()._build_model(d_dense_n=128, d_dense_depth=2)


class ModelPolicy(m.ModelPolicy):
    def _build_model(self):
        return super()._build_model(d_dense_n=128, d_dense_depth=2)


class ModelRND(m.ModelRND):
    def _build_model(self):
        return super()._build_model(dense_n=128, dense_depth=2, output_size=128)


ModelRepProjection = m.ModelRepProjection
ModelRepPrediction = m.ModelRepPrediction
