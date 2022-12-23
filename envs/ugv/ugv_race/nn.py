import torch
from torchvision import transforms as T

import algorithm.nn_models as m
from algorithm.utils.image_visual import ImageVisual
from algorithm.utils.ray import RayVisual
from algorithm.utils.transform import GaussianNoise, SaltAndPepperNoise

RAY_SIZE = 400
AUG_RAY_RANDOM_SIZE = 250


class ModelRep(m.ModelBaseSimpleRep):
    def _build_model(self, blur, brightness, ray_random):
        assert self.obs_shapes[0] == (84, 84, 3)
        assert self.obs_shapes[1] == (802,)  # ray (1 + 200 + 200) * 2
        assert self.obs_shapes[2] == (2,)  # vector

        self.ray_random = ray_random
        if self.train_mode:
            self.ray_random = 150
        if blur != 0:
            self.blurrer = m.Transform(T.GaussianBlur(blur, sigma=blur))
        else:
            self.blurrer = None

        self.brightness = m.Transform(T.ColorJitter(brightness=(brightness, brightness)))

        self._ray_visual = RayVisual()

        self._image_visual = ImageVisual()

        self.conv = m.ConvLayers(84, 84, 3, 'simple',
                                 out_dense_n=64, out_dense_depth=2)

        self.ray_conv = m.Conv1dLayers(RAY_SIZE, 2, 'default',
                                       out_dense_n=64, out_dense_depth=2)

        self.vis_ray_dense = m.LinearLayers(64, dense_n=64, dense_depth=1)

        cropper = torch.nn.Sequential(
            T.RandomCrop(size=(50, 50)),
            T.Resize(size=(84, 84))
        )
        self.vis_cam_random_transformers = T.RandomChoice([
            m.Transform(SaltAndPepperNoise(0.2, 0.5)),
            m.Transform(GaussianNoise()),
            m.Transform(T.GaussianBlur(9, sigma=9)),
            m.Transform(cropper)
        ])

    def forward(self, obs_list):
        vis_cam, ray, vec = obs_list
        ray = torch.cat([ray[..., :RAY_SIZE], ray[..., RAY_SIZE + 2:]], dim=-1)

        if self.blurrer:
            vis_cam = self.blurrer(vis_cam)
        vis_cam = self.brightness(vis_cam)
        # self._image_visual(vis_cam)

        vis = self.conv(vis_cam)

        ray = ray.view(*ray.shape[:-1], RAY_SIZE, 2)
        random_index = torch.randperm(RAY_SIZE)[:self.ray_random]
        ray[..., random_index, 0] = 1.
        ray[..., random_index, 1] = 1.
        # self._ray_visual(ray)
        ray = self.ray_conv(ray)

        vis_ray_concat = self.vis_ray_dense(vis + ray)

        return torch.concat([vis_ray_concat, vec], dim=-1)

    def get_state_from_encoders(self, obs_list, encoders):
        vis_cam, ray, vec = obs_list
        vis_cam_encoder, ray_encoder = encoders

        vis_ray = self.vis_ray_dense(vis_cam_encoder + ray_encoder)

        return torch.concat([vis_ray, vec], dim=-1)

    def get_augmented_encoders(self, obs_list):
        vis_cam, ray, vec = obs_list
        ray = torch.cat([ray[..., :RAY_SIZE], ray[..., RAY_SIZE + 2:]], dim=-1)

        aug_vis_cam = self.vis_cam_random_transformers(vis_cam)
        vis_cam_encoder = self.conv(aug_vis_cam)

        ray = ray.view(*ray.shape[:-1], RAY_SIZE, 2)
        ray_random = (torch.rand(1) * AUG_RAY_RANDOM_SIZE).int()
        random_index = torch.randperm(RAY_SIZE)[:ray_random]
        ray[..., random_index, 0] = 1.
        ray[..., random_index, 1] = 1.
        ray_encoder = self.ray_conv(ray)

        return vis_cam_encoder, ray_encoder


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
