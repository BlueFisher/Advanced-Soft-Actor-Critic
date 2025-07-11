import torch
from torch.nn import functional
from torchvision.transforms import v2 as T
from torchvision.transforms import InterpolationMode

import algorithm.nn_models as m
from algorithm.nn_models.layers.seq_layers import POSITIONAL_ENCODING

from .nn_low import OBS_SHAPES, RAY_SIZE, AUG_RAY_RANDOM_SIZE, COLOR_MAP


NUM_OPTIONS = 5


class ModelOptionSelectorRep(m.ModelBaseOptionSelectorRep):
    def _build_model(self, ray_random):
        for u_s, s in zip(self.obs_shapes, OBS_SHAPES):
            assert u_s == s, f'{u_s} {s}'

        option_eye = torch.eye(NUM_OPTIONS, dtype=torch.float32)
        self.register_buffer('option_eye', option_eye)

        self.seg_num_classes = COLOR_MAP.shape[0]
        self.register_buffer('color_map', COLOR_MAP.permute(1, 0)[None, None, :, :, None, None])

        self.ray_random = ray_random

        self.conv_cam_seg = m.ConvLayers(84, 84, self.seg_num_classes, 'simple',
                                         out_dense_n=64, out_dense_depth=2)

        self.conv_third_cam_seg = m.ConvLayers(84, 84, self.seg_num_classes, 'simple',
                                               out_dense_n=64, out_dense_depth=2)

        self.ray_conv = m.Conv1dLayers(RAY_SIZE, 2, 'default',
                                       out_dense_n=64, out_dense_depth=2)

        self.attn = m.MultiheadAttention(64, 8, pe=POSITIONAL_ENCODING.ROPE)

        self.rnn = m.GRU(64 + 4 + self.c_action_size, 64, 1)

        self._vis_random_transformers = T.RandomChoice([
            m.Transform(T.RandomResizedCrop(size=(84, 84), scale=(0.8, 0.9), interpolation=InterpolationMode.NEAREST)),
            m.Transform(T.ElasticTransform(alpha=100, sigma=5, interpolation=InterpolationMode.NEAREST))
        ])

    @torch.no_grad()
    def _map_color(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-3)
        x = torch.argmin(torch.sum((x - self.color_map).pow(2), -4), -3)
        x = functional.one_hot(x, num_classes=self.seg_num_classes)
        x = x.permute(0, 1, 4, 2, 3).float()
        return x

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

        vec = vec[..., 2:]

        # remove the center ray
        ray = torch.cat([ray[..., :RAY_SIZE], ray[..., RAY_SIZE + 2:]], dim=-1)
        ray = ray.view(*ray.shape[:-1], RAY_SIZE, 2)

        """ DOMAIN RANDOMIZATION """
        random_index = torch.randperm(RAY_SIZE)[:self.ray_random]
        ray[..., random_index, 0] = 1.
        ray[..., random_index, 1] = 1.

        """ ENCODE """
        vis = self._map_color(vis)
        vis_third = self._map_color(vis_third)

        vis_encoder = self.conv_cam_seg(vis)
        vis_third_encoder = self.conv_third_cam_seg(vis_third)

        ray_encoder = self.ray_conv(ray)

        x = torch.cat([vis_encoder.unsqueeze(-2),
                       vis_third_encoder.unsqueeze(-2),
                       ray_encoder.unsqueeze(-2)], dim=-2)
        x, _ = self.attn(x, x, x)
        x = x.sum(dim=-2)  # [batch, seq_len, f]

        x = torch.cat([x, vec, pre_action], dim=-1)

        if pre_seq_hidden_state is not None:
            pre_seq_hidden_state = pre_seq_hidden_state[:, 0]
        x, hn = self.rnn(x, pre_seq_hidden_state)

        x = torch.cat([llm_state, x], dim=-1)

        return x, hn

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

        x = torch.cat([vis_encoder.unsqueeze(-2),
                       vis_third_encoder.unsqueeze(-2),
                       ray_encoder.unsqueeze(-2)], dim=-2)
        x, _ = self.attn(x, x, x)
        x = x.sum(dim=-2)  # [batch, seq_len, f]

        x = torch.cat([x, vec, pre_action], dim=-1)

        if pre_seq_hidden_state is not None:
            pre_seq_hidden_state = pre_seq_hidden_state[:, 0]
        x, hn = self.rnn(x, pre_seq_hidden_state)

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
        ray_random = (torch.rand(1) * AUG_RAY_RANDOM_SIZE).int()
        random_index = torch.randperm(RAY_SIZE)[:ray_random]
        ray[..., random_index, 0] = 1.
        ray[..., random_index, 1] = 1.

        """ ENCODE """
        vis_aug = self._map_color(vis_aug)
        vis_third_aug = self._map_color(vis_third_aug)

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

        self.seg_num_classes = COLOR_MAP.shape[0]
        self.register_buffer('color_map', COLOR_MAP.permute(1, 0)[None, None, :, :, None, None])

        self.ray_random = ray_random

        self.conv_cam_seg = m.ConvLayers(84, 84, self.seg_num_classes, 'simple',
                                         out_dense_n=64, out_dense_depth=2)

        self.conv_third_cam_seg = m.ConvLayers(84, 84, self.seg_num_classes, 'simple',
                                               out_dense_n=64, out_dense_depth=2)

        self.ray_conv = m.Conv1dLayers(RAY_SIZE, 2, 'default',
                                       out_dense_n=64, out_dense_depth=2)

        self.attn = m.MultiheadAttention(64, 8, pe=POSITIONAL_ENCODING.ROPE)

        self.rnn = m.GRU(64 + 4 + self.c_action_size, 64, 1)

        self._vis_random_transformers = T.RandomChoice([
            m.Transform(T.RandomResizedCrop(size=(84, 84), scale=(0.8, 0.9), interpolation=InterpolationMode.NEAREST)),
            m.Transform(T.ElasticTransform(alpha=100, sigma=5, interpolation=InterpolationMode.NEAREST))
        ])

    @torch.no_grad()
    def _map_color(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-3)
        x = torch.argmin(torch.sum((x - self.color_map).pow(2), -4), -3)
        x = functional.one_hot(x, num_classes=self.seg_num_classes)
        x = x.permute(0, 1, 4, 2, 3).float()
        return x

    def forward(self,
                obs_list: list[torch.Tensor],
                pre_action: torch.Tensor,
                pre_seq_hidden_state: torch.Tensor | None,
                padding_mask: torch.Tensor | None = None):
        high_state, llm_obs, vis, vis_third, ray, vec = obs_list
        high_state = high_state[..., NUM_OPTIONS:]

        """ PREPROCESSING """
        vec = vec[..., 2:]

        # remove the center ray
        ray = torch.cat([ray[..., :RAY_SIZE], ray[..., RAY_SIZE + 2:]], dim=-1)
        ray = ray.view(*ray.shape[:-1], RAY_SIZE, 2)

        """ DOMAIN RANDOMIZATION """
        random_index = torch.randperm(RAY_SIZE)[:self.ray_random]
        ray[..., random_index, 0] = 1.
        ray[..., random_index, 1] = 1.

        """ ENCODE """
        vis = self._map_color(vis)
        vis_third = self._map_color(vis_third)

        vis_encoder = self.conv_cam_seg(vis)
        vis_third_encoder = self.conv_third_cam_seg(vis_third)

        # self._ray_visual(ray, max_batch=3)
        ray_encoder = self.ray_conv(ray)

        x = torch.cat([vis_encoder.unsqueeze(-2),
                       vis_third_encoder.unsqueeze(-2),
                       ray_encoder.unsqueeze(-2)], dim=-2)
        x, _ = self.attn(x, x, x)
        x = x.sum(dim=-2)  # [batch, seq_len, f]

        x = torch.cat([x, vec, pre_action], dim=-1)

        if pre_seq_hidden_state is not None:
            pre_seq_hidden_state = pre_seq_hidden_state[:, 0]
        x, hn = self.rnn(x, pre_seq_hidden_state)

        x = torch.cat([high_state, x], dim=-1)

        return x, hn

    def get_state_from_encoders(self,
                                encoders: torch.Tensor | tuple[torch.Tensor],
                                obs_list: list[torch.Tensor],
                                pre_action: torch.Tensor,
                                pre_seq_hidden_state: torch.Tensor | None,
                                padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        high_state, llm_obs, vis, vis_third, ray, vec = obs_list
        high_state = high_state[..., NUM_OPTIONS:]

        vec = vec[..., 2:]

        vis_encoder, vis_third_encoder, ray_encoder = encoders

        x = torch.cat([vis_encoder.unsqueeze(-2),
                       vis_third_encoder.unsqueeze(-2),
                       ray_encoder.unsqueeze(-2)], dim=-2)
        x, _ = self.attn(x, x, x)
        x = x.sum(dim=-2)  # [batch, seq_len, f]

        x = torch.cat([x, vec, pre_action], dim=-1)

        if pre_seq_hidden_state is not None:
            pre_seq_hidden_state = pre_seq_hidden_state[:, 0]
        x, hn = self.rnn(x, pre_seq_hidden_state)

        x = torch.cat([high_state, x, vec], dim=-1)

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
        ray_random = (torch.rand(1) * AUG_RAY_RANDOM_SIZE).int()
        random_index = torch.randperm(RAY_SIZE)[:ray_random]
        ray[..., random_index, 0] = 1.
        ray[..., random_index, 1] = 1.

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
