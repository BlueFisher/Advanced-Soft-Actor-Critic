import torch
from torchvision.transforms import v2 as T
from torchvision.transforms import InterpolationMode

import algorithm.nn_models as m
from algorithm.utils.visualization.image import ImageVisual
from algorithm.utils.visualization.ray import RayVisual


OBS_SHAPES = [(1,), (3, 84, 84), (3, 84, 84), (802,), (6,)]

RAY_SIZE = 400
AUG_RAY_RANDOM_SIZE = 250


class ModelRep(m.ModelBaseRep):
    def _build_model(self, ray_random):
        for u_s, s in zip(self.obs_shapes, OBS_SHAPES):
            assert u_s == s, f'{u_s} {s}'

        self._image_visual = ImageVisual()
        self._ray_visual = RayVisual()

        self.ray_random = ray_random

        self.conv_cam_seg = m.ConvLayers(84, 84, 3, 'simple',
                                         out_dense_n=64, out_dense_depth=2)

        self.conv_third_cam_seg = m.ConvLayers(84, 84, 3, 'simple',
                                               out_dense_n=64, out_dense_depth=2)

        self.ray_conv = m.Conv1dLayers(RAY_SIZE, 2, 'default',
                                       out_dense_n=64, out_dense_depth=2)

        self.dense = m.LinearLayers(64 * 3, dense_n=128, dense_depth=1)

        self._vis_random_transformers = T.RandomChoice([
            m.Transform(T.RandomResizedCrop(size=(84, 84), scale=(0.8, 0.9), interpolation=InterpolationMode.NEAREST)),
            m.Transform(T.ElasticTransform(alpha=100, sigma=5, interpolation=InterpolationMode.NEAREST))
        ])

    def forward(self,
                obs_list: list[torch.Tensor],
                pre_action: torch.Tensor,
                pre_seq_hidden_state: torch.Tensor | None,
                padding_mask: torch.Tensor | None = None):
        llm_obs, vis, vis_third, ray, vec = obs_list

        """ PREPROCESSING """
        # remove the center ray
        ray = torch.cat([ray[..., :RAY_SIZE], ray[..., RAY_SIZE + 2:]], dim=-1)
        ray = ray.view(*ray.shape[:-1], RAY_SIZE, 2)

        """ DOMAIN RANDOMIZATION """
        random_index = torch.randperm(RAY_SIZE)[:self.ray_random]
        ray[..., random_index, 0] = 1.
        ray[..., random_index, 1] = 1.

        """ ENCODE """
        # self._image_visual(vis, vis_third, max_batch=3)
        vis_encoder = self.conv_cam_seg(vis)
        vis_third_encoder = self.conv_third_cam_seg(vis_third)

        # self._ray_visual(ray, max_batch=3)
        ray_encoder = self.ray_conv(ray)

        x = self.dense(torch.cat([vis_encoder, vis_third_encoder, ray_encoder], dim=-1))
        x = torch.cat([x, vec], dim=-1)

        return x, self._get_empty_seq_hidden_state(x)

    def get_state_from_encoders(self,
                                encoders: torch.Tensor | tuple[torch.Tensor],
                                obs_list: list[torch.Tensor],
                                pre_action: torch.Tensor,
                                pre_seq_hidden_state: torch.Tensor | None,
                                padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        llm_obs, vis, vis_third, ray, vec = obs_list

        vis_encoder, vis_third_encoder, ray_encoder = encoders

        x = self.dense(torch.cat([vis_encoder, vis_third_encoder, ray_encoder], dim=-1))
        x = torch.cat([x, vec], dim=-1)

        return x

    def get_augmented_encoders(self,
                               obs_list: list[torch.Tensor]) -> torch.Tensor | tuple[torch.Tensor]:
        llm_obs, vis, vis_third, ray, vec = obs_list

        """ PREPROCESSING """
        # remove the center ray
        ray = torch.cat([ray[..., :RAY_SIZE], ray[..., RAY_SIZE + 2:]], dim=-1)
        ray = ray.view(*ray.shape[:-1], RAY_SIZE, 2)

        """ AUGMENTATION """
        vis_aug = self._vis_random_transformers(vis)
        vis_third_aug = self._vis_random_transformers(vis_third)
        ray_random = (torch.rand(1) * AUG_RAY_RANDOM_SIZE).int()
        random_index = torch.randperm(RAY_SIZE)[:ray_random]
        ray[..., random_index, 0] = 1.
        ray[..., random_index, 1] = 1.

        """ ENCODE """
        vis_encoder = self.conv_cam_seg(vis_aug)
        vis_third_encoder = self.conv_third_cam_seg(vis_third_aug)

        ray_encoder = self.ray_conv(ray)

        return vis_encoder, vis_third_encoder, ray_encoder


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
