import torch
from torchvision import transforms as T

import algorithm.nn_models as m
from algorithm.nn_models.representation import ModelRepPrediction
from algorithm.utils.transform import GaussianNoise, SaltAndPepperNoise


class ModelRep(m.ModelBaseRNNRep):
    def _build_model(self, blur, brightness, need_ray, need_speed):
        assert self.obs_shapes[0] == (84, 84, 3)
        assert self.obs_shapes[1] == (1442,)  # ray (1 + 360 + 360) * 2
        assert self.obs_shapes[2] == (6,)  # vector

        self.need_ray = need_ray
        self.need_speed = need_speed
        if blur != 0:
            self.blurrer = m.Transform(T.GaussianBlur(blur, sigma=blur))
        else:
            self.blurrer = None

        self.brightness = m.Transform(T.ColorJitter(brightness=(brightness, brightness)))

        self.ray_index = []
        self.ray_size = 720
        for i in reversed(range(360)):
            self.ray_index.append((i * 2 + 1) * 2)
            self.ray_index.append((i * 2 + 1) * 2 + 1)
        for i in range(360):
            self.ray_index.append((i * 2 + 2) * 2)
            self.ray_index.append((i * 2 + 2) * 2 + 1)

        self.conv = m.ConvLayers(84, 84, 3, 'simple',
                                 out_dense_n=64, out_dense_depth=2)

        self.ray_conv = m.Conv1dLayers(720, 2, 'default',
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

        if not self.need_ray:
            ray = torch.ones_like(ray)

        vis = self.conv(vis_cam)

        ray = ray.view(*ray.shape[:-1], self.ray_size, 2)
        ray = self.ray_conv(ray)

        vis_ray_concat = self.vis_ray_dense(torch.cat([vis, ray], dim=-1))
        state, hn = self.rnn(torch.cat([vis_ray_concat, pre_action], dim=-1), rnn_state)

        state = torch.cat([state, vec], dim=-1)

        return state, hn

    def get_augmented_encoder(self, obs_list):
        vis_cam, ray, vec = obs_list
        ray = ray[..., self.ray_index]

        transformed_vis_cam = self.random_transformers(vis_cam)
        vis_encoder = self.conv(transformed_vis_cam)

        ray = ray.view(*ray.shape[:-1], self.ray_size, 2)
        ray[..., torch.randperm(720)[:72], 1] = 0.03
        ray_encoder = self.ray_conv(ray)

        return vis_encoder, ray_encoder


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
