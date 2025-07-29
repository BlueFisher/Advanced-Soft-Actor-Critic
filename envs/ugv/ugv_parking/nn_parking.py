import torch
from torch.nn import functional
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2 as T

import algorithm.nn_models as m
from algorithm.nn_models.layers.seq_layers import POSITIONAL_ENCODING

from .nn_low import AUG_RAY_RANDOM_PROB, NUM_CLASSES, OBS_SHAPES, RAY_SIZE

NUM_OPTIONS = 5


class ModelOptionSelectorRep(m.ModelBaseOptionSelectorRep):
    def _build_model(self, ray_random):
        for u_s, s in zip(self.obs_shapes, OBS_SHAPES):
            assert u_s == s, f'{u_s} {s}'

        option_eye = torch.eye(NUM_OPTIONS, dtype=torch.float32)
        self.register_buffer('option_eye', option_eye)

        self.ray_random = ray_random

        self.conv_cam_seg = m.ConvLayers(84, 84, NUM_CLASSES, 'simple',
                                         out_dense_n=64, out_dense_depth=2)

        self.conv_third_cam_seg = m.ConvLayers(84, 84, NUM_CLASSES, 'simple',
                                               out_dense_n=64, out_dense_depth=2)

        self.ray_conv = m.Conv1dLayers(RAY_SIZE, 2, 'default',
                                       out_dense_n=64, out_dense_depth=2)

        self.dense = m.LinearLayers(6, output_size=64)

        self.attn = m.MultiheadAttention(64, 8, pe=POSITIONAL_ENCODING.ROPE)

        self._vis_random_transformers = T.RandomChoice([
            m.Transform(T.RandomResizedCrop(size=(84, 84), scale=(0.8, 0.9), interpolation=InterpolationMode.NEAREST)),
            m.Transform(T.ElasticTransform(alpha=100, sigma=5, interpolation=InterpolationMode.NEAREST))
        ])

    def forward(self,
                obs_list: list[torch.Tensor],
                pre_action: torch.Tensor,
                pre_seq_hidden_state: torch.Tensor | None,
                pre_termination_mask: torch.Tensor | None = None,
                padding_mask: torch.Tensor | None = None):
        llm_obs, vis, vis_third, ray, vec = obs_list
        # ClockwiseRace = 0
        # AnticlockwiseRace = 1
        # ReachingPark = 2
        # Parking = 3
        # TurnAround = 4

        """ PREPROCESSING """
        llm_state = llm_obs.to(torch.long)
        llm_state = self.option_eye[llm_state.squeeze(-1)]

        # remove the center ray
        ray = torch.cat([ray[..., :RAY_SIZE], ray[..., RAY_SIZE + 2:]], dim=-1)
        ray = ray.view(*ray.shape[:-1], RAY_SIZE, 2)

        """ DOMAIN RANDOMIZATION """
        ray_random = torch.rand((ray.shape[0], ray.shape[1], RAY_SIZE, 1), device=ray.device)
        ray_random = ray_random < self.ray_random
        ray = ray * (~ray_random) + 1. * ray_random

        """ ENCODE """
        vis_encoder = self.conv_cam_seg(vis)
        vis_third_encoder = self.conv_third_cam_seg(vis_third)

        ray_encoder = self.ray_conv(ray)

        x = self.dense(vec)

        sensor_f = torch.cat([vis_encoder.unsqueeze(-2),
                              vis_third_encoder.unsqueeze(-2),
                              ray_encoder.unsqueeze(-2)], dim=-2)
        attn_x, _ = self.attn(x.unsqueeze(-2), sensor_f, sensor_f)
        x = x + attn_x[..., 0, :]

        x = torch.cat([llm_state, x], dim=-1)

        return x, self._get_empty_seq_hidden_state(x)

    def get_state_from_encoders(self,
                                encoders: torch.Tensor | tuple[torch.Tensor],
                                obs_list: list[torch.Tensor],
                                pre_action: torch.Tensor,
                                pre_seq_hidden_state: torch.Tensor | None,
                                padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        llm_obs, vis, vis_third, ray, vec = obs_list

        """ PREPROCESSING """
        llm_state = llm_obs.to(torch.long)
        llm_state = self.option_eye[llm_state.squeeze(-1)]

        vec = vec[..., 2:]

        vis_encoder, vis_third_encoder, ray_encoder = encoders

        x = self.dense(vec)

        sensor_f = torch.cat([vis_encoder.unsqueeze(-2),
                              vis_third_encoder.unsqueeze(-2),
                              ray_encoder.unsqueeze(-2)], dim=-2)
        attn_x, _ = self.attn(x.unsqueeze(-2), sensor_f, sensor_f)
        x = x + attn_x[..., 0, :]

        x = torch.cat([llm_state, x], dim=-1)

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
        ray_random = torch.rand((ray.shape[0], ray.shape[1], RAY_SIZE, 1), device=ray.device)
        ray_random = ray_random < AUG_RAY_RANDOM_PROB
        ray = ray * (~ray_random) + 1. * ray_random

        """ ENCODE """
        vis_encoder = self.conv_cam_seg(vis_aug)
        vis_third_encoder = self.conv_third_cam_seg(vis_third_aug)

        ray_encoder = self.ray_conv(ray)

        return vis_encoder, vis_third_encoder, ray_encoder


class ModelVOverOptions(m.ModelVOverOptions):
    def _build_model(self):
        super()._build_model(dense_n=64, dense_depth=2)


class ModelOptionSelectorRND(m.ModelOptionSelectorRND):
    def _build_model(self):
        super()._build_model(dense_n=64, dense_depth=2, output_size=64)


class ModelRep(m.ModelBaseRep):
    def _build_model(self, ray_random):
        self.ray_random = ray_random

        self.conv_cam_seg = m.ConvLayers(84, 84, NUM_CLASSES, 'simple',
                                         out_dense_n=64, out_dense_depth=2)

        self.conv_third_cam_seg = m.ConvLayers(84, 84, NUM_CLASSES, 'simple',
                                               out_dense_n=64, out_dense_depth=2)

        self.ray_conv = m.Conv1dLayers(RAY_SIZE, 2, 'default',
                                       out_dense_n=64, out_dense_depth=2)

        self.dense = m.LinearLayers(6, output_size=64)

        self.attn = m.MultiheadAttention(64, 8, pe=POSITIONAL_ENCODING.ROPE)

        self._vis_random_transformers = T.RandomChoice([
            m.Transform(T.RandomResizedCrop(size=(84, 84), scale=(0.8, 0.9), interpolation=InterpolationMode.NEAREST)),
            m.Transform(T.ElasticTransform(alpha=100, sigma=5, interpolation=InterpolationMode.NEAREST))
        ])

    def forward(self,
                obs_list: list[torch.Tensor],
                pre_action: torch.Tensor,
                pre_seq_hidden_state: torch.Tensor | None,
                padding_mask: torch.Tensor | None = None):
        high_state, llm_obs, vis, vis_third, ray, vec = obs_list
        high_state = high_state[..., NUM_OPTIONS:]

        """ PREPROCESSING """
        # remove the center ray
        ray = torch.cat([ray[..., :RAY_SIZE], ray[..., RAY_SIZE + 2:]], dim=-1)
        ray = ray.view(*ray.shape[:-1], RAY_SIZE, 2)

        """ DOMAIN RANDOMIZATION """
        ray_random = torch.rand((ray.shape[0], ray.shape[1], RAY_SIZE, 1), device=ray.device)
        ray_random = ray_random < self.ray_random
        ray = ray * (~ray_random) + 1. * ray_random

        """ ENCODE """
        vis_encoder = self.conv_cam_seg(vis)
        vis_third_encoder = self.conv_third_cam_seg(vis_third)

        # self._ray_visual(ray, max_batch=3)
        ray_encoder = self.ray_conv(ray)

        x = self.dense(vec)

        sensor_f = torch.cat([vis_encoder.unsqueeze(-2),
                              vis_third_encoder.unsqueeze(-2),
                              ray_encoder.unsqueeze(-2)], dim=-2)
        attn_x, _ = self.attn(x.unsqueeze(-2), sensor_f, sensor_f)
        x = x + attn_x[..., 0, :]

        x = torch.cat([high_state, x], dim=-1)

        return x, self._get_empty_seq_hidden_state(x)

    def get_state_from_encoders(self,
                                encoders: torch.Tensor | tuple[torch.Tensor],
                                obs_list: list[torch.Tensor],
                                pre_action: torch.Tensor,
                                pre_seq_hidden_state: torch.Tensor | None,
                                padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        high_state, llm_obs, vis, vis_third, ray, vec = obs_list
        high_state = high_state[..., NUM_OPTIONS:]

        vis_encoder, vis_third_encoder, ray_encoder = encoders

        x = self.dense(vec)

        sensor_f = torch.cat([vis_encoder.unsqueeze(-2),
                              vis_third_encoder.unsqueeze(-2),
                              ray_encoder.unsqueeze(-2)], dim=-2)
        attn_x, _ = self.attn(x.unsqueeze(-2), sensor_f, sensor_f)
        x = x + attn_x[..., 0, :]

        x = torch.cat([high_state, x], dim=-1)

        return x

    def get_augmented_encoders(self,
                               obs_list: list[torch.Tensor]) -> torch.Tensor | tuple[torch.Tensor]:
        high_state, llm_obs, vis, vis_third, ray, vec = obs_list

        """ PREPROCESSING """
        # remove the center ray
        ray = torch.cat([ray[..., :RAY_SIZE], ray[..., RAY_SIZE + 2:]], dim=-1)
        ray = ray.view(*ray.shape[:-1], RAY_SIZE, 2)

        """ AUGMENTATION """
        vis_aug = self._vis_random_transformers(vis)
        vis_third_aug = self._vis_random_transformers(vis_third)
        ray_random = torch.rand((ray.shape[0], ray.shape[1], RAY_SIZE, 1), device=ray.device)
        ray_random = ray_random < AUG_RAY_RANDOM_PROB
        ray = ray * (~ray_random) + 1. * ray_random

        """ ENCODE """
        vis_aug = self._map_color(vis_aug)
        vis_third_aug = self._map_color(vis_third_aug)

        vis_encoder = self.conv_cam_seg(vis_aug)
        vis_third_encoder = self.conv_third_cam_seg(vis_third_aug)

        ray_encoder = self.ray_conv(ray)

        return vis_encoder, vis_third_encoder, ray_encoder


HIGH_STATE_SIZE = 64


class ModelQ(m.ModelQ):
    def _build_model(self):
        return super()._build_model(c_dense_n=128, c_dense_depth=2)


class ModelPolicy(m.ModelPolicy):
    def _build_model(self):
        self.state_size -= HIGH_STATE_SIZE
        return super()._build_model(c_dense_n=128, c_dense_depth=2)

    def forward(self, state, obs_list):
        state = state[..., HIGH_STATE_SIZE:]
        return super().forward(state, obs_list)


class ModelRND(m.ModelRND):
    def _build_model(self):
        super()._build_model(dense_n=128, dense_depth=2, output_size=128)


class ModelTermination(m.ModelTermination):
    def _build_model(self):
        super()._build_model(dense_n=128, dense_depth=2, dropout=0.05)


ModelRepProjection = m.ModelRepProjection
ModelRepPrediction = m.ModelRepPrediction
