import torch
from torchvision import transforms as T

import algorithm.nn_models as m
from algorithm.utils.image_visual import ImageVisual
from algorithm.utils.ray import RayVisual, generate_unity_to_nn_ray_index
from algorithm.utils.transform import GaussianNoise, SaltAndPepperNoise

RAY_SIZE = 400


class ModelRep(m.ModelBaseRNNRep):
    def _build_model(self, blur, brightness, ray_random, need_speed):
        assert self.obs_shapes[0] == (84, 84, 3)
        assert self.obs_shapes[1] == (802,)  # ray (1 + 200 + 200) * 2
        assert self.obs_shapes[2] == (6,)  # vector

        self.ray_random = ray_random
        self.need_speed = need_speed
        if blur != 0:
            self.blurrer = m.Transform(T.GaussianBlur(blur, sigma=blur))
        else:
            self.blurrer = None

        self.brightness = m.Transform(T.ColorJitter(brightness=(brightness, brightness)))

        self.ray_index = generate_unity_to_nn_ray_index(RAY_SIZE)
        # self._ray_visual = RayVisual()

        self.conv = m.ConvLayers(84, 84, 3, 'simple',
                                 out_dense_n=64, out_dense_depth=2)

        self.ray_conv = m.Conv1dLayers(RAY_SIZE, 2, 'default',
                                       out_dense_n=64, out_dense_depth=2)

        self.vis_ray_dense = m.LinearLayers(self.conv.output_size + self.ray_conv.output_size,
                                            dense_n=64, dense_depth=1)

        self.rnn = m.GRU(64 + self.c_action_size, 64, 1)

        cropper = torch.nn.Sequential(
            T.RandomCrop(size=(50, 50)),
            T.Resize(size=(84, 84))
        )
        self.random_transformers = T.RandomChoice([
            m.Transform(SaltAndPepperNoise(0.2, 0.5)),
            m.Transform(GaussianNoise()),
            m.Transform(T.GaussianBlur(9, sigma=9)),
            m.Transform(cropper)
        ])

    def forward(self, obs_list, pre_action, rnn_state=None):
        vis_cam, ray, vec = obs_list
        ray = ray[..., self.ray_index]

        if self.blurrer:
            vis_cam = self.blurrer(vis_cam)
        vis_cam = self.brightness(vis_cam)

        vis = self.conv(vis_cam)

        ray = ray.view(*ray.shape[:-1], RAY_SIZE, 2)
        ray_random = (torch.rand(1) * self.ray_random).int()
        random_index = torch.randperm(RAY_SIZE)[:ray_random]
        ray[..., random_index, 0] = 1.
        ray[..., random_index, 1] = 1.
        # self._ray_visual(ray)
        ray = self.ray_conv(ray)

        vis_ray_concat = self.vis_ray_dense(torch.cat([vis, ray], dim=-1))
        state, hn = self.rnn(torch.cat([vis_ray_concat, pre_action], dim=-1), rnn_state)

        state = torch.cat([state, vec], dim=-1)

        return state, hn

    def get_state_from_encoders(self, obs_list, encoders, pre_action, rnn_state=None):
        vis_cam, ray, vec = obs_list
        vis_cam_encoder, ray_encoder = encoders

        vis_ray_concat = self.vis_ray_dense(torch.cat([vis_cam_encoder, ray_encoder], dim=-1))
        state, _ = self.rnn(torch.cat([vis_ray_concat, pre_action], dim=-1), rnn_state)

        state = torch.cat([state, vec], dim=-1)

        return state

    def get_augmented_encoders(self, obs_list):
        vis_cam, ray, vec = obs_list
        ray = ray[..., self.ray_index]

        aug_vis_cam = self.random_transformers(vis_cam)
        vis_cam_encoder = self.conv(aug_vis_cam)

        ray = ray.view(*ray.shape[:-1], RAY_SIZE, 2)
        ray_random = (torch.rand(1) * self.ray_random).int()
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
