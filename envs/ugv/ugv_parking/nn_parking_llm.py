import base64
from io import BytesIO

import numpy as np
import requests
import torch
from PIL import Image

import algorithm.nn_models as m

from .nn_parking import *

PROMPT_MAP = """
This is a top-down map that has been processed using a semantic segmentation algorithm. You are currently guiding an autonomous vehicle to park in a parking space.
Green represents randomly generated obstacles, white represents walls, yellow represents non-passable areas on the ground, pink represents passable areas on the ground, blue represents the parking space, which is the only final destination of the autonomous vehicle, and black represents unrecognized areas.
The map consists of an outer circular track that is passable and an inner square area with streets. There are some randomly placed green obstacles on the outer circular track. The outer circular track is separated from the inner area by walls, but there is a random gap connecting the outer circular track to the inner area, where a small section of the white wall is missing (in this example, it's in the upper right corner), allowing access from the outer area to the inner area.
The autonomous vehicle has learned four subtasks, and can only execute one subtask at a time. Once a subtask is completed, it will automatically stop. The subtasks are numbered and described as follows:
1: Park in the parking space from the southern area of the parking lot.
2: Drive clockwise along the outer circular track, avoiding obstacles, to reach near the gap.
3: Drive counterclockwise along the outer circular track, avoiding obstacles, to reach near the gap.
4: Drive from the gap between the outer and inner areas to the area in front of the parking space.
"""

PROMPT_VIS_THIRD = """
This is a partial top-down view captured by a drone above the autonomous vehicle, which has been processed using a semantic segmentation algorithm.
Red represents the autonomous vehicle, which is currently facing {direction}.
Based on this semantic segmentation map, choose which subtask should be executed next.
Output the number of subtask only.
"""


def np2base64(image_array: np.ndarray):
    image_array = (image_array * 255.).astype(np.uint8)
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

        image_map_array = np.load(self.model_abs_dir / 'map.npy')
        self.image_map_base64 = np2base64(image_map_array)
        image_example_array = np.load(self.model_abs_dir / 'example.npy')
        self.image_example_base64 = np2base64(image_example_array)

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

        if pre_seq_hidden_state is None:
            llm_state = torch.zeros((*llm_obs.shape[:-1], NUM_OPTIONS),
                                    device=llm_obs.device)
        else:
            llm_state = pre_seq_hidden_state.clone()

        if pre_termination_mask is not None:
            assert llm_obs.shape[1] == 1

            angle = (torch.atan2(vec[:, 0, 2], vec[:, 0, 3]) * 180 / torch.pi + 360) % 360  # [B, ]
            llm_obs = torch.zeros_like(llm_obs)  # [B, 1, 1]

            m_angle = angle[pre_termination_mask]  # [mB, ]
            m_llm_obs = llm_obs[pre_termination_mask]  # [mB, 1, 1]
            m_vis_third_seg = vis_third_seg[pre_termination_mask]  # [mB, 1, 84, 84, 3]

            for i in range(m_angle.shape[0]):
                r = requests.post('http://127.0.0.1:11434/api/chat',
                                  json={
                                      'model': 'llama3.2-vision',
                                      "messages": [
                                          {
                                              "role": "user",
                                              "content": PROMPT_MAP,
                                              'images': [self.image_map_base64],
                                          },
                                          {
                                              "role": "user",
                                              "content": PROMPT_VIS_THIRD.replace('{direction}', 'east'),
                                              'images': [self.image_example_base64],
                                          }, {
                                              "role": "assistant",
                                              "content": "2"
                                          }, {
                                              "role": "user",
                                              "content": PROMPT_VIS_THIRD.replace('{direction}', get_direction(m_angle[i])),
                                              'images': [np2base64(m_vis_third_seg[i, 0].cpu().numpy())],
                                          }
                                      ],
                                      'stream': False
                                  },
                                  auth=('n705', 'TamWccLw2GVyHbEr'))

                print(r.json()['message']['content'])
                if len(r.json()['message']['content']) == 1:
                    m_llm_obs[i, 0] = int(r.json()['message']['content']) - 1

            llm_obs[pre_termination_mask, 0] = m_llm_obs[:, 0]

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
