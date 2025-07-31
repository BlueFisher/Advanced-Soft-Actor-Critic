import logging

import numpy as np

from algorithm.env_wrapper.env_wrapper import DecisionStep, TerminalStep
from algorithm.env_wrapper.obs_preprocessor_wrapper import \
    ObsPreprocessorWrapper

logger = logging.getLogger('only_cam_preprocessor')

ONLY_CAM_OBS_NAMES = ['CameraSensor',
                      'LLMStateSensor',
                      'RayPerceptionSensor',
                      'ThirdPersonCameraSensor',
                      'VectorSensor_size6']

ONLY_CAM_OBS_SHAPES = [(3, 84, 84),
                       (1,),
                       (802,),
                       (3, 84, 84),
                       (6,)]


ONLY_SEG_OBS_NAMES = ['LLMStateSensor',
                      'RayPerceptionSensor',
                      'SegmentationSensor',
                      'ThirdPersonSegmentationSensor',
                      'VectorSensor_size6']

ONLY_SEG_OBS_SHAPES = [(1,),
                       (802,),
                       (3, 84, 84),
                       (3, 84, 84),
                       (6,)]


COLOR_MAP = np.array([
    [0, 0, 1],  # target
    [0, 1, 0],  # obstacle
    [1, 0, 0],  # ugv
    [1, 1, 1],  # inner_road
    [1, 1, 0],  # outer_road
    [0, 0, 0],  # block
], dtype=np.float32)

NUM_CLASSES = len(COLOR_MAP)
COLOR_MAP_RESHAPED = COLOR_MAP.reshape(1, NUM_CLASSES, 3, 1, 1)


class ObsPreprocessor(ObsPreprocessorWrapper):
    def init(self):
        (ma_obs_names,
         ma_obs_shapes,
         ma_obs_dtypes,
         ma_d_action_sizes,
         ma_c_action_size) = self._env.init()

        self._only_cam = 'CameraSensor' in ma_obs_names['UGVParkingAgent?team=0']

        if self._only_cam:
            for i, n in enumerate(ma_obs_names['UGVParkingAgent?team=0']):
                assert n == ONLY_CAM_OBS_NAMES[i]
            for i, n in enumerate(ma_obs_shapes['UGVParkingAgent?team=0']):
                assert n == ONLY_CAM_OBS_SHAPES[i]
        else:
            assert 'SegmentationSensor' in ma_obs_names['UGVParkingAgent?team=0']

            for i, n in enumerate(ma_obs_names['UGVParkingAgent?team=0']):
                assert n == ONLY_SEG_OBS_NAMES[i]
            for i, n in enumerate(ma_obs_shapes['UGVParkingAgent?team=0']):
                assert n == ONLY_SEG_OBS_SHAPES[i]

        logger.info(f'Original obs: {ma_obs_names["UGVParkingAgent?team=0"]}')

        if self._only_cam:
            ma_obs_names['UGVParkingAgent?team=0'] = [ONLY_CAM_OBS_NAMES[1],
                                                      ONLY_CAM_OBS_NAMES[0],
                                                      ONLY_CAM_OBS_NAMES[3],
                                                      ONLY_CAM_OBS_NAMES[2],
                                                      ONLY_CAM_OBS_NAMES[4]]
            ma_obs_shapes['UGVParkingAgent?team=0'] = [ONLY_CAM_OBS_SHAPES[1],
                                                       ONLY_CAM_OBS_SHAPES[0],
                                                       ONLY_CAM_OBS_SHAPES[3],
                                                       ONLY_CAM_OBS_SHAPES[2],
                                                       ONLY_CAM_OBS_SHAPES[4]]
        else:
            ma_obs_names['UGVParkingAgent?team=0'] = [ONLY_SEG_OBS_NAMES[0],
                                                      ONLY_SEG_OBS_NAMES[2],
                                                      ONLY_SEG_OBS_NAMES[3],
                                                      ONLY_SEG_OBS_NAMES[1],
                                                      ONLY_SEG_OBS_NAMES[4]]
            ma_obs_shapes['UGVParkingAgent?team=0'] = [ONLY_SEG_OBS_SHAPES[0],
                                                       (NUM_CLASSES, 84, 84),
                                                       (NUM_CLASSES, 84, 84),
                                                       ONLY_SEG_OBS_SHAPES[1],
                                                       ONLY_SEG_OBS_SHAPES[4]]
            ma_obs_dtypes['UGVParkingAgent?team=0'][1] = np.bool
            ma_obs_dtypes['UGVParkingAgent?team=0'][2] = np.bool

        logger.info(f'Processed obs: {ma_obs_names["UGVParkingAgent?team=0"]}')
        logger.info(f'Processed obs: {ma_obs_dtypes["UGVParkingAgent?team=0"]}')

        return ma_obs_names, ma_obs_shapes, ma_obs_dtypes, ma_d_action_sizes, ma_c_action_size

    def _map_color(self, x: np.ndarray) -> np.ndarray:
        """
        Convert semantic segmentation images with dimensions [batch, 3, height, width]
        to one-hot NumPy arrays based on COLOR_MAP.

        Args:
            x (np): Input image array with dimensions [B, 3, H, W].

        Returns:
            np: One-hot encoded array with dimensions [B, num_classes, H, W].
        """

        x_expanded = np.expand_dims(x, axis=1)
        distances = np.sum(np.abs((x_expanded - COLOR_MAP_RESHAPED)), axis=2)
        indices = np.argmin(distances, axis=1)
        one_hot = np.eye(NUM_CLASSES, dtype=np.bool)[indices]

        return one_hot.transpose(0, 3, 1, 2)

    def _preprocess_obs(self, obs_list: list[np.ndarray]):
        if self._only_cam:
            vis, llm_obs, ray, vis_third, vec = obs_list
            return [llm_obs, vis, vis_third, ray, vec]
        else:
            llm_obs, ray, vis_seg, vis_third_seg, vec = obs_list
            return [llm_obs, self._map_color(vis_seg), self._map_color(vis_third_seg), ray, vec]

    def reset(self, reset_config: dict | None = None) -> tuple[dict[str, np.ndarray],
                                                               dict[str, list[np.ndarray]]]:

        ma_agent_ids, ma_obs_list = self._env.reset(reset_config)
        ma_obs_list['UGVParkingAgent?team=0'] = self._preprocess_obs(ma_obs_list['UGVParkingAgent?team=0'])

        return ma_agent_ids, ma_obs_list

    def step(self,
             ma_d_action: dict[str, np.ndarray],
             ma_c_action: dict[str, np.ndarray]) -> tuple[DecisionStep, TerminalStep]:

        decision_step, terminal_step, all_envs_done = self._env.step(ma_d_action, ma_c_action)

        decision_step.ma_obs_list['UGVParkingAgent?team=0'] = self._preprocess_obs(decision_step.ma_obs_list['UGVParkingAgent?team=0'])
        terminal_step.ma_obs_list['UGVParkingAgent?team=0'] = self._preprocess_obs(terminal_step.ma_obs_list['UGVParkingAgent?team=0'])

        return decision_step, terminal_step, all_envs_done
