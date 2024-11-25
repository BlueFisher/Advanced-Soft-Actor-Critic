import base64
import requests
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

    def forward(self,
                obs_list: list[torch.Tensor],
                pre_action: torch.Tensor,
                pre_seq_hidden_state: torch.Tensor | None,
                pre_termination_mask: torch.Tensor | None = None,
                padding_mask: torch.Tensor | None = None):
        llm_state, ray, vis_seg, vis_third_seg, vec = obs_list

        if llm_state.shape[0] > 1:  # _build_model
            return (torch.zeros((*llm_state.shape[:-1], NUM_OPTIONS),
                                device=llm_state.device),
                    torch.zeros((*llm_state.shape[:-1], NUM_OPTIONS),
                                device=llm_state.device))

        if llm_state.device != self.option_eye.device:
            self.option_eye = self.option_eye.to(llm_state.device)

        state = pre_seq_hidden_state.clone()

        if pre_termination_mask is not None:
            assert llm_state.shape[0] == 1 and llm_state.shape[1] == 1

            vis_seg = vis_seg[0, 0]  # [H, W, C]

            vis_seg = vis_seg.permute([2, 0, 1])  # [C, H, W]
            vis_seg_pil = T.ToPILImage()(vis_seg)
            vis_seg_image_bytes = vis_seg_pil.tobytes()
            vis_seg_base64_string = base64.b64encode(vis_seg_image_bytes).decode()
            print(vis_seg_base64_string)

            r = requests.post('https://ollama.n705.work/api/generate',
                              json={
                                  'model': 'llava-llama3',
                                  'prompt': 'What\'s in this image',
                                  'images': [vis_seg_base64_string],
                                  'stream': False
                              },
                              auth=('n705', 'TamWccLw2GVyHbEr'))
            print(r.text)

            new_llm_state = llm_state.to(torch.long)
            new_state = self.option_eye[new_llm_state.squeeze(-1)]

            state[pre_termination_mask, 0] = new_state[pre_termination_mask, 0]

        return state, state


class ModelVOverOptions(m.ModelVOverOptions):
    def _build_model(self):
        super()._build_model(dense_n=8, dense_depth=2)


class ModelOptionSelectorRND(m.ModelOptionSelectorRND):
    def _build_model(self):
        super()._build_model(dense_n=8, dense_depth=2, output_size=8)


class ModelRep(m.ModelBaseRep):
    def _build_model(self):
        self._ray_visual = RayVisual()

        self._image_visual = ImageVisual()

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
        high_state, llm_state, ray, vis_seg, vis_third_seg, vec = obs_list
        ray = torch.cat([ray[..., :RAY_SIZE], ray[..., RAY_SIZE + 2:]], dim=-1)

        # self._image_visual(vis_cam, vis_seg, vis_third_cam, vis_third_seg, max_batch=3)
        vis_seg = self.conv_cam_seg(vis_seg)
        vis_third_seg = self.conv_third_cam_seg(vis_third_seg)

        ray = ray.view(*ray.shape[:-1], RAY_SIZE, 2)
        # self._ray_visual(ray, max_batch=3)
        ray = self.ray_conv(ray)

        x = self.dense(torch.cat([vis_seg, vis_third_seg, ray], dim=-1))
        x = torch.cat([x, vec, high_state], dim=-1)

        return x, self._get_empty_seq_hidden_state(x)


class ModelQ(m.ModelQ):
    def _build_model(self):
        return super()._build_model(c_dense_n=128, c_dense_depth=2)


class ModelPolicy(m.ModelPolicy):
    def _build_model(self):
        self.state_size -= NUM_OPTIONS
        return super()._build_model(c_dense_n=128, c_dense_depth=2)

    def forward(self, state, obs_list):
        state = state[..., :-NUM_OPTIONS]
        state = self.dense(state)

        if self.c_action_size:
            l = self.c_dense(state)
            mean = self.mean_dense(l)
            logstd = self.logstd_dense(l)
            c_policy = torch.distributions.Normal(torch.tanh(mean / 5.) * 5., torch.exp(torch.clamp(logstd, -20, 0.5)))
        else:
            c_policy = None

        return None, c_policy


class ModelRND(m.ModelRND):
    def _build_model(self):
        super()._build_model(dense_n=128, dense_depth=2, output_size=128)


class ModelTermination(m.ModelTermination):
    def _build_model(self):
        super()._build_model(dense_n=128, dense_depth=2)
