import torch
from torchvision import transforms as T

import algorithm.nn_models as m
from algorithm.utils.image_visual import ImageVisual
from algorithm.utils.transform import GaussianNoise, SaltAndPepperNoise


class ModelRep(m.ModelBaseRNNRep):
    def _build_model(self, blur, brightness, need_speed):
        assert self.obs_shapes[0] == (84, 84, 3)  # BoundingBoxSensor
        assert self.obs_shapes[1] == (84, 84, 3)  # CameraSensor
        assert self.obs_shapes[2] == (84, 84, 3)  # SegmentationSensor
        assert self.obs_shapes[3] == (7,)

        if blur != 0:
            self.blurrer = m.Transform(T.GaussianBlur(blur, sigma=blur))
        else:
            self.blurrer = None
        self.brightness = m.Transform(T.ColorJitter(brightness=(brightness, brightness)))
        self.need_speed = need_speed

        self.conv = m.ConvLayers(84, 84, 3 * 3, 'simple',
                                 out_dense_n=64, out_dense_depth=2)

        self.dense = m.LinearLayers(self.conv.output_size + 7,
                                    dense_n=64, dense_depth=1)

        self.rnn = m.GRU(64 + self.c_action_size, 64, 1)

        cropper = torch.nn.Sequential(
            T.RandomCrop(size=(50, 50)),
            T.Resize(size=(84, 84))
        )
        self.bounding_box_transformer = m.Transform(cropper)
        self.camera_transformers = T.RandomChoice([
            m.Transform(SaltAndPepperNoise(0.2, 0.5)),
            m.Transform(GaussianNoise()),
            m.Transform(T.GaussianBlur(9, sigma=9)),
            m.Transform(cropper)
        ])
        self.segmentation_transformers = T.RandomChoice([
            m.Transform(T.GaussianBlur(9, sigma=9)),
            m.Transform(cropper)
        ])

    def forward(self, obs_list, pre_action, rnn_state=None):
        vis_bounding_box, vis_camera, vis_segmentation, vec = obs_list

        if self.blurrer:
            vis_camera = self.blurrer(vis_camera)
        vis_camera = self.brightness(vis_camera)

        vis = self.conv(torch.cat([vis_bounding_box, vis_camera, vis_segmentation], dim=-1))

        state = self.dense(torch.cat([vis, vec], dim=-1))

        state, hn = self.rnn(torch.cat([state, pre_action], dim=-1), rnn_state)

        return state, hn

    def get_state_from_encoders(self, obs_list, encoders, pre_action, rnn_state=None):
        *_, vec = obs_list
        vis_encoder = encoders

        state = self.dense(torch.cat([vis_encoder, vec], dim=-1))

        state, _ = self.rnn(torch.cat([state, pre_action], dim=-1), rnn_state)

        return state

    def get_augmented_encoders(self, obs_list):
        vis_bounding_box, vis_camera, vis_segmentation, vec = obs_list

        aug_vis_bounding_box = self.bounding_box_transformer(vis_bounding_box)
        aug_vis_camera = self.camera_transformers(vis_camera)
        aug_vis_segmentation = self.segmentation_transformers(vis_segmentation)
        vis_encoder = self.conv(torch.cat([aug_vis_bounding_box,
                                           aug_vis_camera,
                                           aug_vis_segmentation], dim=-1))

        return vis_encoder


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
