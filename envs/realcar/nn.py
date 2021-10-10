import torch
from torchvision import transforms as T

import algorithm.nn_models as m
from algorithm.utils.transform import GaussianNoise, SaltAndPepperNoise

EXTRA_SIZE = 6


class ModelRep(m.ModelBaseRNNRep):
    def _build_model(self, blur, brightness, need_ray, need_speed):
        assert self.obs_shapes[0] == (84, 84, 3)
        assert self.obs_shapes[1] == (2,)  # ray
        assert self.obs_shapes[2] == (8,)  # vector

        self.need_ray = need_ray
        self.need_speed = need_speed

        self.conv = m.ConvLayers(84, 84, 3, 'simple',
                                 out_dense_n=64, out_dense_depth=2)

        self.dense = m.LinearLayers(self.conv.output_size,
                                    dense_n=64, dense_depth=1)

        self.rnn = m.GRU(64 + self.c_action_size, 64, 1)

        if blur != 0:
            self.blurrer = m.Transform(T.GaussianBlur(blur, sigma=blur))
        else:
            self.blurrer = None

        self.brightness = m.Transform(T.ColorJitter(brightness=(brightness, brightness)))

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
        vec = vec[..., :-EXTRA_SIZE]

        if self.blurrer:
            vis_cam = self.blurrer(vis_cam)

        vis_cam = self.brightness(vis_cam)

        if not self.need_ray:
            ray = torch.ones_like(ray)

        vis = self.conv(vis_cam)

        state, hn = self.rnn(torch.cat([self.dense(vis), pre_action], dim=-1), rnn_state)

        state = torch.cat([state, ray, vec], dim=-1)

        return state, hn

    def get_contrastive_encoder(self, obs_list):
        vis_cam, *_ = obs_list

        return self.conv(self.random_transformers(vis_cam))


class ModelQ(m.ModelQ):
    def _build_model(self):
        return super()._build_model(c_dense_n=128, c_dense_depth=2)


class ModelPolicy(m.ModelPolicy):
    def _build_model(self):
        return super()._build_model(c_dense_n=128, c_dense_depth=2)


class ModelRND(m.ModelRND):
    def _build_model(self):
        return super()._build_model(dense_n=128, dense_depth=2, output_size=128)
