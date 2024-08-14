import torch
from torchvision import transforms as T

import algorithm.nn_models as m
from algorithm.utils.image_visual import ImageVisual
from algorithm.utils.ray import RayVisual
from algorithm.utils.transform import GaussianNoise, SaltAndPepperNoise


# OBS_NAMES = ['CameraSensor', 'RayPerceptionSensor', 'SegmentationSensor',
#              'ThirdPersonCameraSensor', 'ThirdPersonSegmentationSensor',
#              'VectorSensor_size6']
# OBS_SHAPES = [(84, 84, 3), (802,), (84, 84, 3), (84, 84, 3), (84, 84, 3), (6,)]

OBS_NAMES = ['RayPerceptionSensor',
             'SegmentationSensor',
             'ThirdPersonSegmentationSensor',
             'VectorSensor_size6']
OBS_SHAPES = [(802,), (84, 84, 3), (84, 84, 3), (6,)]

RAY_SIZE = 400
AUG_RAY_RANDOM_SIZE = 250


class ModelRep(m.ModelBaseRep):
    def _build_model(self):
        for u_s, s in zip(self.obs_shapes, OBS_SHAPES):
            assert u_s == s, f'{u_s} {s}'

        self._ray_visual = RayVisual()

        self._image_visual = ImageVisual()

        self.conv_cam_seg = m.ConvLayers(84, 84, 3, 'simple',
                                         out_dense_n=64, out_dense_depth=2)

        self.conv_third_cam_seg = m.ConvLayers(84, 84, 3, 'simple',
                                               out_dense_n=64, out_dense_depth=2)

        self.ray_conv = m.Conv1dLayers(RAY_SIZE, 2, 'default',
                                       out_dense_n=64, out_dense_depth=2)

        self.dense = m.LinearLayers(64 * 3, dense_n=128, dense_depth=1)

    def forward(self,
                obs_list: list[torch.Tensor],
                pre_action: torch.Tensor,
                pre_seq_hidden_state: torch.Tensor | None,
                padding_mask: torch.Tensor | None = None):
        ray, vis_seg, vis_third_seg, vec = obs_list
        ray = torch.cat([ray[..., :RAY_SIZE], ray[..., RAY_SIZE + 2:]], dim=-1)

        # self._image_visual(vis_cam, vis_seg, vis_third_cam, vis_third_seg, max_batch=3)
        vis_seg = self.conv_cam_seg(vis_seg)
        vis_third_seg = self.conv_third_cam_seg(vis_third_seg)

        ray = ray.view(*ray.shape[:-1], RAY_SIZE, 2)
        # self._ray_visual(ray, max_batch=3)
        ray = self.ray_conv(ray)

        x = self.dense(torch.cat([vis_seg, vis_third_seg, ray], dim=-1))
        x = torch.cat([x, vec], dim=-1)

        return x, self._get_empty_seq_hidden_state(x)


class ModelQ(m.ModelQ):
    def _build_model(self):
        return super()._build_model(c_dense_n=128, c_dense_depth=2)


class ModelPolicy(m.ModelPolicy):
    def _build_model(self):
        return super()._build_model(c_dense_n=128, c_dense_depth=2)


class ModelRND(m.ModelRND):
    def _build_model(self):
        return super()._build_model(dense_n=128, dense_depth=2, output_size=128)


ModelRepProjection = m.ModelRepProjection
ModelRepPrediction = m.ModelRepPrediction
