import torch
from torchvision import transforms as T

import algorithm.nn_models as m
from algorithm.utils.elapse_timer import unified_elapsed_timer
from algorithm.utils.image_visual import ImageVisual
from algorithm.utils.ray import RayVisual, generate_unity_to_nn_ray_index
from algorithm.utils.transform import GaussianNoise, SaltAndPepperNoise

RAY_SIZE = 400
AUG_RAY_RANDOM_SIZE = 250


class ModelRep(m.ModelBaseRNNRep):
    def _build_model(self, blur, brightness, ray_random, need_speed):
        assert self.obs_shapes[0] == (10, 7)  # BoundingBoxSensor
        assert self.obs_shapes[1] == (84, 84, 3)  # CameraSensor
        assert self.obs_shapes[2] == (802,)  # ray (1 + 200 + 200) * 2
        assert self.obs_shapes[3] == (10, 7)  # ThirdPersonBoundingBoxSensor
        assert self.obs_shapes[4] == (84, 84, 3)  # ThirdPersonCameraSensor
        assert self.obs_shapes[5] == (6,)  # vector

        self.ray_random = ray_random
        if self.train_mode:
            self.ray_random = 150
        self.need_speed = need_speed
        if blur != 0:
            self.blurrer = m.Transform(T.GaussianBlur(blur, sigma=blur))
        else:
            self.blurrer = None

        self.brightness = m.Transform(T.ColorJitter(brightness=(brightness, brightness)))

        self.ray_index = generate_unity_to_nn_ray_index(RAY_SIZE)

        self.bbox_attn = m.MultiheadAttention(7, 1)
        self.tp_bbox_attn = m.MultiheadAttention(7, 1)
        self.bbox_dense = m.LinearLayers(7, output_size=64)
        self.tp_bbox_dense = m.LinearLayers(7, output_size=64)

        self.conv = m.ConvLayers(84, 84, 3, 'simple',
                                 out_dense_n=64, out_dense_depth=2)
        self.tp_conv = m.ConvLayers(84, 84, 3, 'simple',
                                    out_dense_n=64, out_dense_depth=2)

        self.ray_conv = m.Conv1dLayers(RAY_SIZE, 2, 'default',
                                       out_dense_n=64, out_dense_depth=2)

        self.vis_ray_dense = m.LinearLayers(64, dense_n=64, dense_depth=1)
        self.tp_vis_dense = m.LinearLayers(64, dense_n=64, dense_depth=1)

        self.rnn = m.GRU(64 + self.obs_shapes[5][0] + self.c_action_size, 64, 1)

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

        self._timer = unified_elapsed_timer()

    def forward(self, obs_list, pre_action, rnn_state=None):
        bbox, vis_cam, ray, tp_bbox, vis_tp_cam, vec = obs_list
        ray = ray[..., self.ray_index]

        for v in [vis_cam, vis_tp_cam]:
            if self.blurrer:
                v = self.blurrer(v)
            v = self.brightness(v)

        bbox_mask = ~bbox.any(dim=-1)
        tp_bbox_mask = ~tp_bbox.any(dim=-1)
        bbox_mask[..., 0] = False
        tp_bbox_mask[..., 0] = False
        bbox, _ = self.bbox_attn(bbox, bbox, bbox, key_padding_mask=bbox_mask)
        tp_bbox, _ = self.bbox_attn(tp_bbox, tp_bbox, tp_bbox, key_padding_mask=tp_bbox_mask)
        bbox = self.bbox_dense(bbox.mean(-2))
        tp_bbox = self.bbox_dense(tp_bbox.mean(-2))

        vis_cam = self.conv(vis_cam)
        vis_tp_cam = self.conv(vis_tp_cam)

        ray = ray.view(*ray.shape[:-1], RAY_SIZE, 2)
        random_index = torch.randperm(RAY_SIZE)[:self.ray_random]
        ray[..., random_index, 0] = 1.
        ray[..., random_index, 1] = 1.
        ray = self.ray_conv(ray)

        vis_ray_concat = self.vis_ray_dense(vis_cam + ray)
        tp_vis_concat = self.tp_vis_dense(vis_tp_cam)

        state, hn = self.rnn(torch.cat([vis_ray_concat + tp_vis_concat + bbox + tp_bbox, vec, pre_action], dim=-1), rnn_state)

        return state, hn

    def get_state_from_encoders(self, obs_list, encoders, pre_action, rnn_state=None):
        vis_bbox, vis_cam, ray, vis_tp_bbox, vis_tp_cam, vec = obs_list
        vis_cam_encoder, vis_tp_cam_encoder, ray_encoder = encoders

        vis_bbox = self.conv(vis_bbox)
        vis_tp_bbox = self.conv(vis_tp_bbox)

        vis_ray_concat = self.vis_ray_dense(vis_bbox + vis_cam_encoder + ray_encoder)
        tp_vis_concat = self.tp_vis_dense(vis_tp_bbox + vis_tp_cam_encoder)

        state, _ = self.rnn(torch.cat([vis_ray_concat + tp_vis_concat, vec, pre_action], dim=-1), rnn_state)

        return state

    def get_augmented_encoders(self, obs_list):
        vis_bbox, vis_cam, ray, vis_tp_bbox, vis_tp_cam, vec = obs_list
        ray = ray[..., self.ray_index]

        aug_vis_cam = self.vis_cam_random_transformers(vis_cam)
        vis_cam_encoder = self.conv(aug_vis_cam)

        aug_vis_tp_cam = self.vis_cam_random_transformers(vis_tp_cam)
        vis_tp_cam_encoder = self.conv(aug_vis_tp_cam)

        ray = ray.view(*ray.shape[:-1], RAY_SIZE, 2)
        ray_random = (torch.rand(1) * AUG_RAY_RANDOM_SIZE).int()
        random_index = torch.randperm(RAY_SIZE)[:ray_random]
        ray[..., random_index, 0] = 1.
        ray[..., random_index, 1] = 1.
        ray_encoder = self.ray_conv(ray)

        return vis_cam_encoder, vis_tp_cam_encoder, ray_encoder


class ModelOptionRep(m.ModelBaseRNNRep):
    def _build_model(self, blur, brightness, ray_random, need_speed):
        assert self.obs_shapes[0] == (64,)
        assert self.obs_shapes[1] == (84, 84, 3)
        assert self.obs_shapes[2] == (802,)  # ray (1 + 200 + 200) * 2
        assert self.obs_shapes[3] == (6,)  # vector

        self.ray_random = ray_random
        if self.train_mode:
            self.ray_random = 150
        self.need_speed = need_speed
        if blur != 0:
            self.blurrer = m.Transform(T.GaussianBlur(blur, sigma=blur))
        else:
            self.blurrer = None

        self.brightness = m.Transform(T.ColorJitter(brightness=(brightness, brightness)))

        self.ray_index = generate_unity_to_nn_ray_index(RAY_SIZE)

        self.conv = m.ConvLayers(84, 84, 3, 'simple',
                                 out_dense_n=64, out_dense_depth=2)

        self.ray_conv = m.Conv1dLayers(RAY_SIZE, 2, 'default',
                                       out_dense_n=64, out_dense_depth=2)

        self.vis_ray_dense = m.LinearLayers(64, dense_n=64, dense_depth=1)

        self.rnn = m.GRU(64 + self.obs_shapes[3][0] + self.c_action_size, 64, 1)

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

    def forward(self, obs_list, pre_action, rnn_state=None):
        high_state, vis_cam, ray, vec = obs_list
        ray = ray[..., self.ray_index]

        if self.blurrer:
            vis_cam = self.blurrer(vis_cam)
        vis_cam = self.brightness(vis_cam)

        vis = self.conv(vis_cam)

        ray = ray.view(*ray.shape[:-1], RAY_SIZE, 2)
        random_index = torch.randperm(RAY_SIZE)[:self.ray_random]
        ray[..., random_index, 0] = 1.
        ray[..., random_index, 1] = 1.
        # self._ray_visual(ray)
        ray = self.ray_conv(ray)

        vis_ray_concat = self.vis_ray_dense(vis + ray) + high_state
        state, hn = self.rnn(torch.cat([vis_ray_concat, vec, pre_action], dim=-1), rnn_state)

        return state, hn

    def get_state_from_encoders(self, obs_list, encoders, pre_action, rnn_state=None):
        high_state, vis_cam, ray, vec = obs_list
        vis_cam_encoder, ray_encoder = encoders

        vis_ray_concat = self.vis_ray_dense(vis_cam_encoder + ray_encoder) + high_state
        state, _ = self.rnn(torch.cat([vis_ray_concat, vec, pre_action], dim=-1), rnn_state)

        return state

    def get_augmented_encoders(self, obs_list):
        high_state, vis_cam, ray, vec = obs_list
        ray = ray[..., self.ray_index]

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
