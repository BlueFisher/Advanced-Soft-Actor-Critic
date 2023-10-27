import torch
from torchvision import transforms as T

import algorithm.nn_models as m
from algorithm.utils.image_visual import ImageVisual
from algorithm.utils.transform import GaussianNoise, SaltAndPepperNoise


class ModelRep(m.ModelBaseRNNRep):
    def _build_model(self, blur, brightness):
        assert self.obs_shapes[0] == (84, 84, 3)

        if blur != 0:
            self.blurrer = m.Transform(T.GaussianBlur(blur, sigma=blur))
        else:
            self.blurrer = None

        self.brightness = m.Transform(T.ColorJitter(brightness=(brightness, brightness)))

        # self._image_visual = ImageVisual(self.model_abs_dir)

        self.conv = m.ConvLayers(84, 84, 3, 'simple',
                                 out_dense_n=64, out_dense_depth=2)

        self.rnn = m.GRU(64 + sum(self.d_action_sizes) + self.c_action_size, 64, 1)

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

    def forward(self, obs_list, pre_action, rnn_state=None, padding_mask=None):
        vis_cam = obs_list[0]

        if self.blurrer:
            vis_cam = self.blurrer(vis_cam)
        vis_cam = self.brightness(vis_cam)
        # self._image_visual(vis_cam, max_batch=1)

        vis = self.conv(vis_cam)
        state, hn = self.rnn(torch.cat([vis, pre_action], dim=-1), rnn_state)

        return state, hn

    def get_state_from_encoders(self, obs_list, encoders, pre_action, rnn_state=None, padding_mask=None):
        vis_cam_encoder = encoders
        state, hn = self.rnn(torch.cat([vis_cam_encoder, pre_action], dim=-1), rnn_state)

        return state

    def get_augmented_encoders(self, obs_list):
        vis_cam = obs_list[0]

        aug_vis_cam = self.vis_cam_random_transformers(vis_cam)
        vis_cam_encoder = self.conv(aug_vis_cam)

        return vis_cam_encoder


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
