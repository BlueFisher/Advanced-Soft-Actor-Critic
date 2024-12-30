import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from algorithm.env_wrapper.env_wrapper import DecisionStep, TerminalStep
from algorithm.env_wrapper.obs_preprocessor_wrapper import \
    ObsPreprocessorWrapper

logger = logging.getLogger('only_cam_preprocessor')

OBS_NAMES = ['CameraSensor',
             'LLMStateSensor',
             'RayPerceptionSensor',
             'SegmentationSensor',
             'ThirdPersonCameraSensor',
             'ThirdPersonSegmentationSensor',
             'VectorSensor_size6']

OBS_SHAPES = [(84, 84, 3),
              (1,),
              (802,),
              (84, 84, 3),
              (84, 84, 3),
              (84, 84, 3),
              (6,)]


class ObsPreprocessor(ObsPreprocessorWrapper):
    def init(self):
        (ma_obs_names,
         ma_obs_shapes,
         ma_d_action_sizes,
         ma_c_action_size) = self._env.init()

        for i, n in enumerate(ma_obs_names['UGVParkingAgent?team=0']):
            assert n == OBS_NAMES[i]
        for i, n in enumerate(ma_obs_shapes['UGVParkingAgent?team=0']):
            assert n == OBS_SHAPES[i]

        logger.info(f'Original obs: {ma_obs_names["UGVParkingAgent?team=0"]}')

        ma_obs_names['UGVParkingAgent?team=0'] = [OBS_NAMES[1], OBS_NAMES[0], OBS_NAMES[4], OBS_NAMES[2], OBS_NAMES[6]]
        ma_obs_shapes['UGVParkingAgent?team=0'] = [OBS_SHAPES[1], OBS_SHAPES[0], OBS_SHAPES[4], OBS_SHAPES[2], OBS_SHAPES[6]]

        logger.info(f'Processed obs: {ma_obs_names["UGVParkingAgent?team=0"]}')

        return ma_obs_names, ma_obs_shapes, ma_d_action_sizes, ma_c_action_size

    def _preprocess_obs(self, obs_list: List[np.ndarray]):
        vis, llm_obs, ray, vis_seg, vis_third, vis_third_seg, vec = obs_list

        return [llm_obs, vis, vis_third, ray, vec]

    def reset(self, reset_config: Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray],
                                                                  Dict[str, List[np.ndarray]]]:

        ma_agent_ids, ma_obs_list = self._env.reset(reset_config)
        ma_obs_list['UGVParkingAgent?team=0'] = self._preprocess_obs(ma_obs_list['UGVParkingAgent?team=0'])

        return ma_agent_ids, ma_obs_list

    def step(self,
             ma_d_action: Dict[str, np.ndarray],
             ma_c_action: Dict[str, np.ndarray]) -> Tuple[DecisionStep, TerminalStep]:

        decision_step, terminal_step, all_envs_done = self._env.step(ma_d_action, ma_c_action)

        decision_step.ma_obs_list['UGVParkingAgent?team=0'] = self._preprocess_obs(decision_step.ma_obs_list['UGVParkingAgent?team=0'])
        terminal_step.ma_obs_list['UGVParkingAgent?team=0'] = self._preprocess_obs(terminal_step.ma_obs_list['UGVParkingAgent?team=0'])

        return decision_step, terminal_step, all_envs_done
