import logging
import numpy as np


from algorithm.env_wrapper.env_wrapper import DecisionStep, TerminalStep
from algorithm.env_wrapper.obs_preprocessor_wrapper import \
    ObsPreprocessorWrapper

logger = logging.getLogger('obs_preprocessor')

ORI_OBS_NAMES = ['BoundingBoxSensor',
                 'CameraSensor',
                 'SegmentationSensor',
                 'VectorSensor_size7']

ORI_OBS_SHAPES = [(6, 6),
                  (3, 84, 84),
                  (3, 84, 84),
                  (7,)]


COLOR_MAP = np.array([
    [0, 0, 1],  # target
    [0, 1, 0],  # obstacle
    [1, 0, 0],  # wall
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

        for i, n in enumerate(ma_obs_names['USV?team=0']):
            assert n == ORI_OBS_NAMES[i]
        for i, n in enumerate(ma_obs_shapes['USV?team=0']):
            assert n == ORI_OBS_SHAPES[i]

        logger.info(f'Original obs: {ma_obs_names["USV?team=0"]}')
        logger.info(f'Processed obs: {ma_obs_dtypes["USV?team=0"]}')

        ma_obs_dtypes['USV?team=0'][1] = np.uint8

        ma_obs_shapes['USV?team=0'][2] = (NUM_CLASSES, 84, 84)
        ma_obs_dtypes['USV?team=0'][2] = np.bool

        logger.info(f'Processed obs: {ma_obs_names["USV?team=0"]}')
        logger.info(f'Processed obs: {ma_obs_dtypes["USV?team=0"]}')

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
        bbox, vis_camera, vis_segmentation, vec = obs_list

        vis_camera = (vis_camera * 255.).astype(np.uint8)
        vis_segmentation = self._map_color(vis_segmentation)

        return [bbox, vis_camera, vis_segmentation, vec]

    def reset(self, reset_config: dict | None = None) -> tuple[dict[str, np.ndarray],
                                                               dict[str, list[np.ndarray]]]:

        ma_agent_ids, ma_obs_list = self._env.reset(reset_config)
        ma_obs_list['USV?team=0'] = self._preprocess_obs(ma_obs_list['USV?team=0'])

        return ma_agent_ids, ma_obs_list

    def step(self,
             ma_d_action: dict[str, np.ndarray],
             ma_c_action: dict[str, np.ndarray]) -> tuple[DecisionStep, TerminalStep]:

        decision_step, terminal_step, all_envs_done = self._env.step(ma_d_action, ma_c_action)

        decision_step.ma_obs_list['USV?team=0'] = self._preprocess_obs(decision_step.ma_obs_list['USV?team=0'])
        terminal_step.ma_obs_list['USV?team=0'] = self._preprocess_obs(terminal_step.ma_obs_list['USV?team=0'])

        return decision_step, terminal_step, all_envs_done
