from typing import Dict, List, Optional, Tuple
import base64
from io import BytesIO
import numpy as np
from PIL import Image
import requests

from algorithm.env_wrapper.env_wrapper import DecisionStep, TerminalStep
from algorithm.env_wrapper.obs_preprocessor_wrapper import ObsPreprocessorWrapper


PROMPT_MAP = """
This is a top-down map that has been processed using a semantic segmentation algorithm. You are currently guiding an autonomous vehicle to park in a parking space.
Green represents randomly generated obstacles, white represents walls, yellow represents non-passable areas on the ground, pink represents passable areas on the ground, blue represents the parking space, which is the only final destination of the autonomous vehicle, and black represents unrecognized areas.
The map consists of an outer circular track that is passable and an inner square area with streets. There are some randomly placed green obstacles on the outer circular track. The outer circular track is separated from the inner area by walls, but there is a random gap connecting the outer circular track to the inner area, where a small section of the white wall is missing (in this example, it's in the upper right corner), allowing access from the outer area to the inner area.
The autonomous vehicle has learned four subtasks, and can only execute one subtask at a time. Once a subtask is completed, it will automatically stop. The subtasks are numbered and described as follows:
1: Park in the parking space from the southern area of the parking lot.
2: Drive clockwise along the outer circular track, avoiding obstacles, to reach near the gap.
3: Drive counterclockwise along the outer circular track, avoiding obstacles, to reach near the gap.
4: Drive from the gap between the outer and inner areas to the area in front of the parking space.
The sequence of subtasks to complete the parking task can only be 2-4-1 or 3-4-1.
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


class ObsPreprocessor(ObsPreprocessorWrapper):
    def __init__(self, env):
        super().__init__(env)

        if env.model_abs_dir:
            image_map_array = np.load(env.model_abs_dir / 'map.npy')
            self.image_map_base64 = np2base64(image_map_array)
            image_example_array = np.load(env.model_abs_dir / 'example.npy')
            self.image_example_base64 = np2base64(image_example_array)

    def _preprocess_obs(self, obs_list: List[np.ndarray]):
        llm_obs, ray, vis_seg, vis_third_seg, vec = obs_list

        angles = (np.atan2(vec[:, 2], vec[:, 3]) * 180 / np.pi + 360) % 360
        llm_result = np.zeros_like(llm_obs)
        for i, angle in enumerate(angles):
            r = requests.post('http://127.0.0.1:11434/api/chat',
                              json={
                                  'model': 'llama3.2-vision',
                                  "messages": [
                                      {
                                          "role": "user",
                                          "content": PROMPT_MAP,
                                          'images': [self.image_map_base64],
                                      }, {
                                          "role": "user",
                                          "content": PROMPT_VIS_THIRD.replace('{direction}', 'east'),
                                          'images': [self.image_example_base64],
                                      }, {
                                          "role": "assistant",
                                          "content": "4"
                                      }, {
                                          "role": "user",
                                          "content": PROMPT_VIS_THIRD.replace('{direction}', get_direction(angle)),
                                          'images': [np2base64(vis_seg[i])],
                                      }
                                  ],
                                  'stream': False
                              },
                              auth=('n705', 'TamWccLw2GVyHbEr'))
            if len(r.json()['message']['content']) == 1:
                llm_result[i] = int(r.json()['message']['content']) - 1

        return [llm_result, ray, vis_seg, vis_third_seg, vec]

    def reset(self, reset_config: Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray],
                                                                  Dict[str, List[np.ndarray]]]:

        ma_agent_ids, ma_obs_list = self._env.reset(reset_config)
        for n in ma_obs_list:
            ma_obs_list[n] = self._preprocess_obs(ma_obs_list[n])

        return ma_agent_ids, ma_obs_list

    def step(self,
             ma_d_action: Dict[str, np.ndarray],
             ma_c_action: Dict[str, np.ndarray]) -> Tuple[DecisionStep, TerminalStep]:

        decision_step, terminal_step, all_envs_done = self._env.step(ma_d_action, ma_c_action)

        for n in decision_step.ma_obs_list:
            decision_step.ma_obs_list[n] = self._preprocess_obs(decision_step.ma_obs_list[n])

        for n in terminal_step.ma_obs_list:
            terminal_step.ma_obs_list[n] = self._preprocess_obs(terminal_step.ma_obs_list[n])

        return decision_step, terminal_step, all_envs_done
