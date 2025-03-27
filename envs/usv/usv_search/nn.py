import torch
from torchvision import transforms as T

import algorithm.nn_models as m
from algorithm.utils.transform import GaussianNoise, SaltAndPepperNoise


class ModelRep(m.ModelBaseRep):
    def _build_model(self, blur, brightness, need_speed):
        assert self.obs_shapes[0] == (6, 6)  # BoundingBoxSensor
        assert self.obs_shapes[1] == (3, 84, 84)  # CameraSensor
        assert self.obs_shapes[2] == (3, 84, 84)  # SegmentationSensor
        assert self.obs_shapes[3] == (7,)

        if blur != 0:
            self.blurrer = m.Transform(T.GaussianBlur(blur, sigma=blur))
        else:
            self.blurrer = None
        self.brightness = m.Transform(T.ColorJitter(brightness=(brightness, brightness)))
        self.need_speed = need_speed

        self.bbox_attn = m.MultiheadAttention(6, 1)

        self.conv = m.ConvLayers(84, 84, 3 * 2, 'simple',
                                 out_dense_n=64, out_dense_depth=1,
                                 output_size=16)

        self.dense = m.LinearLayers(6 + self.conv.output_size + 7,
                                    dense_n=64, dense_depth=1)

        self.rnn = m.GRU(self.dense.output_size + self.c_action_size, 64, 1)

        cropper = torch.nn.Sequential(
            T.RandomCrop(size=(50, 50)),
            T.Resize(size=(84, 84))
        )
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

    def _handle_bbox(self, bbox):
        bbox_mask = ~bbox.any(dim=-1)
        bbox_mask[..., 0] = False
        bbox, _ = self.bbox_attn(bbox, bbox, bbox, key_padding_mask=bbox_mask)
        bbox = bbox.mean(-2)

        return bbox

    def forward(self, obs_list, pre_action, rnn_state=None, padding_mask=None):
        bbox, vis_camera, vis_segmentation, vec = obs_list

        if self.blurrer:
            vis_camera = self.blurrer(vis_camera)
        vis_camera = self.brightness(vis_camera)

        bbox = self._handle_bbox(bbox)

        vis = self.conv(torch.cat([vis_camera, vis_segmentation], dim=-1))

        state = self.dense(torch.cat([bbox, vis, vec], dim=-1))

        state, hn = self.rnn(torch.cat([state, pre_action], dim=-1), rnn_state)

        return state, hn

    def get_state_from_encoders(self, obs_list, encoders, pre_action, rnn_state=None):
        bbox, vis_camera, vis_segmentation, vec = obs_list
        vis_encoder = encoders

        bbox = self._handle_bbox(bbox)

        state = self.dense(torch.cat([bbox, vis_encoder, vec], dim=-1))

        state, _ = self.rnn(torch.cat([state, pre_action], dim=-1), rnn_state)

        return state

    def get_augmented_encoders(self, obs_list):
        bbox, vis_camera, vis_segmentation, vec = obs_list

        aug_vis_camera = self.camera_transformers(vis_camera)
        aug_vis_segmentation = self.segmentation_transformers(vis_segmentation)
        vis_encoder = self.conv(torch.cat([aug_vis_camera,
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
