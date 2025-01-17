import torch
from torchvision import transforms as T

import algorithm.nn_models as m
from algorithm.utils.transform import GaussianNoise, SaltAndPepperNoise

from .nn_low import OBS_SHAPES, RAY_SIZE, AUG_RAY_RANDOM_SIZE


NUM_OPTIONS = 4


class ModelOptionSelectorRep(m.ModelBaseOptionSelectorRep):
    def _build_model(self, ray_random):
        for u_s, s in zip(self.obs_shapes, OBS_SHAPES):
            assert u_s == s, f'{u_s} {s}'

        self.ray_random = ray_random

        self.option_eye = torch.eye(NUM_OPTIONS, dtype=torch.float32)

        self.conv_cam_seg = m.ConvLayers(84, 84, 3, 'simple',
                                         out_dense_n=64, out_dense_depth=2)

        self.conv_third_cam_seg = m.ConvLayers(84, 84, 3, 'simple',
                                               out_dense_n=64, out_dense_depth=2)

        self.ray_conv = m.Conv1dLayers(RAY_SIZE, 2, 'default',
                                       out_dense_n=64, out_dense_depth=2)

        self.dense = m.LinearLayers(64 * 3, dense_n=128, dense_depth=1)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.option_eye = self.option_eye.to(*args, **kwargs).detach()
        return self

    def forward(self,
                obs_list: list[torch.Tensor],
                pre_action: torch.Tensor,
                pre_seq_hidden_state: torch.Tensor | None,
                pre_termination_mask: torch.Tensor | None = None,
                padding_mask: torch.Tensor | None = None):
        llm_obs, vis, vis_third, ray, vec = obs_list
        # parking
        # clockwise_race
        # anticlockwise_race
        # reaching_park

        """ PREPROCESSING """
        llm_state = llm_obs.to(torch.long)
        llm_state = self.option_eye[llm_state.squeeze(-1)]

        # remove the center ray
        ray = torch.cat([ray[..., :RAY_SIZE], ray[..., RAY_SIZE + 2:]], dim=-1)
        ray = ray.view(*ray.shape[:-1], RAY_SIZE, 2)

        """ DOMAIN RANDOMIZATION """
        random_index = torch.randperm(RAY_SIZE)[:self.ray_random]
        ray[..., random_index, 0] = 1.
        ray[..., random_index, 1] = 1.

        """ ENCODE """
        vis_encoder = self.conv_cam_seg(vis)
        vis_third_encoder = self.conv_third_cam_seg(vis_third)

        ray_encoder = self.ray_conv(ray)

        x = self.dense(torch.cat([vis_encoder, vis_third_encoder, ray_encoder], dim=-1))
        x = torch.cat([llm_state, x, vec], dim=-1)

        return x, self._get_empty_seq_hidden_state(x)


class ModelVOverOptions(m.ModelVOverOptions):
    def _build_model(self):
        super()._build_model(dense_n=64, dense_depth=2)


class ModelOptionSelectorRND(m.ModelOptionSelectorRND):
    def _build_model(self):
        super()._build_model(dense_n=8, dense_depth=2, output_size=8)


class ModelRep(m.ModelBaseRep):
    def _build_model(self, ray_random):
        self.ray_random = ray_random

        self.conv_cam_seg = m.ConvLayers(84, 84, 3, 'simple',
                                         out_dense_n=64, out_dense_depth=2)

        self.conv_third_cam_seg = m.ConvLayers(84, 84, 3, 'simple',
                                               out_dense_n=64, out_dense_depth=2)

        self.ray_conv = m.Conv1dLayers(RAY_SIZE, 2, 'default',
                                       out_dense_n=64, out_dense_depth=2)

        self.dense = m.LinearLayers(64 * 3, dense_n=128, dense_depth=1)

    def forward(self,
                obs_list: list[torch.Tensor],
                pre_action: torch.Tensor,
                pre_seq_hidden_state: torch.Tensor | None,
                padding_mask: torch.Tensor | None = None):
        high_state, llm_state, vis, vis_third, ray, vec = obs_list
        high_state = high_state[..., NUM_OPTIONS:]

        """ PREPROCESSING """
        # remove the center ray
        ray = torch.cat([ray[..., :RAY_SIZE], ray[..., RAY_SIZE + 2:]], dim=-1)
        ray = ray.view(*ray.shape[:-1], RAY_SIZE, 2)

        """ DOMAIN RANDOMIZATION """
        random_index = torch.randperm(RAY_SIZE)[:self.ray_random]
        ray[..., random_index, 0] = 1.
        ray[..., random_index, 1] = 1.

        """ ENCODE """
        vis_encoder = self.conv_cam_seg(vis)
        vis_third_encoder = self.conv_third_cam_seg(vis_third)

        ray_encoder = self.ray_conv(ray)

        x = self.dense(torch.cat([vis_encoder, vis_third_encoder, ray_encoder], dim=-1))
        x = torch.cat([high_state, x, vec], dim=-1)

        return x, self._get_empty_seq_hidden_state(x)


HIGH_STATE_SIZE = 128 + 6


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
        super()._build_model(dense_n=128, dense_depth=1)
