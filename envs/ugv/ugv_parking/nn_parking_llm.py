import base64
from io import BytesIO
import time

import numpy as np
import requests
import torch
from PIL import Image

import algorithm.nn_models as m

from .nn_parking import *
from .obs_preprocessor import COLOR_MAP

PROMPT_MAP = """
This is a top-down map that has been processed using a semantic segmentation algorithm. You are currently guiding an autonomous vehicle to park in a parking space.
Green represents randomly generated obstacles, black represents non-passable areas on the ground, white represents passable inner areas on the ground, yellow represents passable outer area on the ground, blue represents the parking space, which is the only destination of the autonomous vehicle.
The map consists of an outer circular track that is passable and an inner square area with streets. There are some randomly placed green obstacles on the outer circular track. The outer circular track is separated from the inner area, but there is a random gap connecting the outer circular track to the inner area, where the inner white area connects the yellow outer area (in this example, it's in the bottom right corner), allowing access from the outer area to the inner area.
The autonomous vehicle has learned five subtasks and can only execute one subtask at a time. Once a subtask is completed, it will automatically stop. The subtasks are numbered and described as follows:
1: Drive clockwise along the outer circular track, avoiding obstacles, to reach near the gap.
2: Drive counterclockwise along the outer circular track, avoiding obstacles, to reach near the gap.
3: Drive from the gap between the outer and inner areas to the area in front of the parking space.
4: Park in the parking space from the southern area of the parking lot.
5. Turn around in the outer circular track.
"""

PROMPT_VIS_THIRD = """
This is a partial top-down view captured by a drone above the autonomous vehicle, which has been processed using a semantic segmentation algorithm.
Red represents the autonomous vehicle, which is currently facing {direction}.
Based on this semantic segmentation map, choose which subtask should be executed next.
Return a JSON object with a required key "subtask" and its value should be an integer from 1 to 5, corresponding to the subtask number.
The JSON object can have a optional key "reason" to explain why this subtask is chosen.
"""


def np2base64(image_array: np.ndarray):
    image_array = (image_array.transpose(1, 2, 0) * 255.).astype(np.uint8)
    image = Image.fromarray(image_array)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    image_bytes = buffered.getvalue()
    return base64.b64encode(image_bytes).decode('utf-8')


def get_direction(angle):
    if 337.5 <= angle or angle < 22.5:
        return "north"
    elif 22.5 <= angle < 67.5:
        return "northeast"
    elif 67.5 <= angle < 112.5:
        return "east"
    elif 112.5 <= angle < 157.5:
        return "southeast"
    elif 157.5 <= angle < 202.5:
        return "south"
    elif 202.5 <= angle < 247.5:
        return "southwest"
    elif 247.5 <= angle < 292.5:
        return "west"
    elif 292.5 <= angle < 337.5:
        return "northwest"
    else:
        return "unknown direction"


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

        image_map_array = np.load(self.model_abs_dir / 'map.npy')
        self.image_map_base64 = np2base64(image_map_array)
        image_example_1_array = np.load(self.model_abs_dir / 'example_1.npy')
        self.image_example_1_base64 = np2base64(image_example_1_array)
        image_example_2_array = np.load(self.model_abs_dir / 'example_2.npy')
        self.image_example_2_base64 = np2base64(image_example_2_array)

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
        if pre_seq_hidden_state is None:
            llm_state = torch.zeros((*llm_obs.shape[:-1], NUM_OPTIONS),
                                    device=llm_obs.device)
        else:
            llm_state = pre_seq_hidden_state.clone()

        # print(pre_termination_mask)
        if pre_termination_mask is not None:
            assert llm_obs.shape[1] == 1

            angle = (torch.atan2(vec[:, 0, 2], vec[:, 0, 3]) * 180 / torch.pi + 360) % 360  # [B, ]
            _llm_obs = llm_obs.clone()

            m_angle = angle[pre_termination_mask]  # [mB, ]
            m_llm_obs = _llm_obs[pre_termination_mask]  # [mB, 1, 1]
            m_vis_third = vis_third[pre_termination_mask]  # [mB, 1, 84, 84, 3]

            # for i in range(m_angle.shape[0]):
            #     t = time.time()
            #     r = requests.post(
            #         'http://127.0.0.1:11434/api/chat',
            #         json={
            #             'model': 'minicpm-v',
            #             "messages": [
            #                 {
            #                     "role": "system",
            #                     "content": PROMPT_MAP,
            #                     'images': [self.image_map_base64],
            #                 },
            #                 # {
            #                 #     "role": "user",
            #                 #     "content": PROMPT_VIS_THIRD.replace('{direction}', 'west'),
            #                 #     'images': [self.image_example_1_base64],
            #                 # },
            #                 # {
            #                 #     "role": "assistant",
            #                 #     "content": "1"
            #                 # },
            #                 # {
            #                 #     "role": "user",
            #                 #     "content": PROMPT_VIS_THIRD.replace('{direction}', 'north'),
            #                 #     'images': [self.image_example_2_base64],
            #                 # },
            #                 # {
            #                 #     "role": "assistant",
            #                 #     "content": "3"
            #                 # },
            #                 {
            #                     "role": "user",
            #                     "content": PROMPT_VIS_THIRD.replace('{direction}', get_direction(m_angle[i])),
            #                     'images': [np2base64(COLOR_MAP[m_vis_third[i, 0].cpu().numpy().argmax(0)].transpose(2, 0, 1))],
            #                 }
            #             ],
            #             'format': {
            #                 'type': 'object',
            #                 'properties': {
            #                     'reason': {
            #                         'type': 'string'
            #                     },
            #                     'subtask': {
            #                         'type': 'integer',
            #                     }
            #                 },
            #                 "required": ["subtask"]
            #             },
            #             "keep_alive": -1,
            #             'stream': False
            #         },
            #         auth=('n705', 'TamWccLw2GVyHbEr')
            #     )

            #     print('=' * 10)
            #     print(r.json()['message']['content'], time.time() - t)
            #     # if len(r.json()['message']['content']) == 1:
            #     #     m_llm_obs[i, 0] = int(r.json()['message']['content']) - 1

            m_llm_state = m_llm_obs.to(torch.long)
            m_llm_state = self.option_eye[m_llm_state.squeeze(-1)]
            llm_state[pre_termination_mask] = m_llm_state

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

        return x, llm_state
