import logging
from typing import Dict, List, Tuple

import numpy as np

from .only_cam_preprocessor import OBS_NAMES, OBS_SHAPES
from .only_cam_preprocessor import ObsPreprocessor as OnlyCamObsPreprocessor

logger = logging.getLogger('only_seg_preprocessor')


class ObsPreprocessor(OnlyCamObsPreprocessor):
    def init(self):
        (ma_obs_names,
         ma_obs_shapes,
         ma_d_action_sizes,
         ma_c_action_size) = self._env.init()

        for i, n in enumerate(ma_obs_names['UGVParkingAgent?team=0']):
            assert n == OBS_NAMES[i], (n, OBS_NAMES[i])

        logger.info(f'Original obs: {ma_obs_names["UGVParkingAgent?team=0"]}')

        ma_obs_names['UGVParkingAgent?team=0'] = [OBS_NAMES[1], OBS_NAMES[3], OBS_NAMES[5], OBS_NAMES[2], OBS_NAMES[6]]
        ma_obs_shapes['UGVParkingAgent?team=0'] = [OBS_SHAPES[1], OBS_SHAPES[3], OBS_SHAPES[5], OBS_SHAPES[2], OBS_SHAPES[6]]

        logger.info(f'Processed obs: {ma_obs_names["UGVParkingAgent?team=0"]}')

        return ma_obs_names, ma_obs_shapes, ma_d_action_sizes, ma_c_action_size

    def _preprocess_obs(self, obs_list: List[np.ndarray]):
        vis, llm_obs, ray, vis_seg, vis_third, vis_third_seg, vec = obs_list

        return [llm_obs, vis_seg, vis_third_seg, ray, vec]
