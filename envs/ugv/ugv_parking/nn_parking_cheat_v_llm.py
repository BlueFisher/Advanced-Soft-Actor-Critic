from turtle import forward
import torch
from torchvision import transforms as T

import algorithm.nn_models as m
from algorithm.utils.image_visual import ImageVisual
from algorithm.utils.ray import RayVisual
from algorithm.utils.transform import GaussianNoise, SaltAndPepperNoise

# OBS_NAMES = ['CameraSensor', 'RayPerceptionSensor', 'SegmentationSensor',
#              'ThirdPersonCameraSensor', 'ThirdPersonSegmentationSensor',
#              'VectorSensor_size6']
# OBS_SHAPES = [(84, 84, 3), (802,), (84, 84, 3), (84, 84, 3), (84, 84, 3), (6,)]

OBS_NAMES = ['LLMStateSensor',
             'RayPerceptionSensor',
             'SegmentationSensor',
             'ThirdPersonSegmentationSensor',
             'VectorSensor_size6']
OBS_SHAPES = [(1,), (802,), (84, 84, 3), (84, 84, 3), (6,)]

RAY_SIZE = 400
AUG_RAY_RANDOM_SIZE = 250

NUM_OPTIONS = 4


class ModelOptionSelectorRep(m.ModelBaseOptionSelectorRep):
    def _build_model(self):
        for u_s, s in zip(self.obs_shapes, OBS_SHAPES):
            assert u_s == s, f'{u_s} {s}'

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
        llm_obs, ray, vis_seg, vis_third_seg, vec = obs_list
        # parking
        # clockwise_race
        # anticlockwise_race
        # reaching_park

        ray = torch.cat([ray[..., :RAY_SIZE], ray[..., RAY_SIZE + 2:]], dim=-1)

        llm_state = llm_obs.to(torch.long)
        llm_state = self.option_eye[llm_state.squeeze(-1)]

        vis_seg = self.conv_cam_seg(vis_seg)
        vis_third_seg = self.conv_third_cam_seg(vis_third_seg)

        ray = ray.view(*ray.shape[:-1], RAY_SIZE, 2)
        ray = self.ray_conv(ray)

        x = self.dense(torch.cat([vis_seg, vis_third_seg, ray], dim=-1))
        x = torch.cat([x, vec], dim=-1)

        state = torch.concat([llm_state, x], dim=-1)

        return state, self._get_empty_seq_hidden_state(state)


class ModelVOverOptions(m.ModelVOverOptions):
    def _build_model(self):
        pass

    def forward(self, state):
        state = state[..., :NUM_OPTIONS]
        state = state * 2. - 1.
        return state


class ModelOptionSelectorRND(m.ModelOptionSelectorRND):
    def _build_model(self):
        super()._build_model(dense_n=8, dense_depth=2, output_size=8)


class ModelRep(m.ModelBaseRep):
    def _build_model(self):
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
        high_state, llm_obs, ray, vis_seg, vis_third_seg, vec = obs_list
        high_state = high_state[..., NUM_OPTIONS:]

        ray = torch.cat([ray[..., :RAY_SIZE], ray[..., RAY_SIZE + 2:]], dim=-1)

        vis_seg = self.conv_cam_seg(vis_seg)
        vis_third_seg = self.conv_third_cam_seg(vis_third_seg)

        ray = ray.view(*ray.shape[:-1], RAY_SIZE, 2)
        ray = self.ray_conv(ray)

        x = self.dense(torch.cat([vis_seg, vis_third_seg, ray], dim=-1))
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
