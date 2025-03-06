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

ONLY_CAM_OBS_SHAPES = [(84, 84, 3),
                       (1,),
                       (802,),
                       (84, 84, 3),
                       (6,)]


ONLY_SEG_OBS_NAMES = ['LLMStateSensor',
                      'RayPerceptionSensor',
                      'SegmentationSensor',
                      'ThirdPersonSegmentationSensor',
                      'VectorSensor_size6']

ONLY_SEG_OBS_SHAPES = [(1,),
                       (802,),
                       (84, 84, 3),
                       (84, 84, 3),
                       (6,)]


class ObsPreprocessor(ObsPreprocessorWrapper):
    def init(self):
        (ma_obs_names,
         ma_obs_shapes,
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
                                                       ONLY_SEG_OBS_SHAPES[2],
                                                       ONLY_SEG_OBS_SHAPES[3],
                                                       ONLY_SEG_OBS_SHAPES[1],
                                                       ONLY_SEG_OBS_SHAPES[4]]

        logger.info(f'Processed obs: {ma_obs_names["UGVParkingAgent?team=0"]}')

        return ma_obs_names, ma_obs_shapes, ma_d_action_sizes, ma_c_action_size

    def _preprocess_obs(self, obs_list: list[np.ndarray]):
        if self._only_cam:
            vis, llm_obs, ray, vis_third, vec = obs_list
            return [llm_obs, vis, vis_third, ray, vec]
        else:
            llm_obs, ray, vis_seg, vis_third_seg, vec = obs_list
            return [llm_obs, vis_seg, vis_third_seg, ray, vec]

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
