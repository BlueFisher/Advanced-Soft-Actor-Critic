import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

if __name__ in ('__main__', '__mp_main__'):
    from env_wrapper import EnvWrapper
else:
    from .env_wrapper import EnvWrapper


class OfflineWrapper(EnvWrapper):
    # TODO
    def __init__(self,
                 env_name: str,
                 env_args: Optional[str | Dict] = None,
                 n_envs: int = 1):
        super().__init__(True, env_name, env_args, n_envs)

        self._logger = logging.getLogger('OfflineWrapper')

        dataset_dir = Path(__file__).resolve().parent
        if 'dataset_path' in self.env_args:
            dataset_dir = Path(self.env_args['dataset_path'])

        self._dataset_dir = dataset_dir / env_name

        self._logger.info(f'Dataset directory: {self._dataset_dir}')

    def init(self):
        # Load dataset info
        with open(self._dataset_dir / 'info.json') as f:
            dataset_info = json.load(f)

        ma_obs_names: Dict[str, List[str]] = dataset_info['ma_obs_names']
        ma_obs_shapes: Dict[str, List[Tuple]] = {n: [tuple(s) for s in obs_shapes] for n, obs_shapes in dataset_info['ma_obs_shapes'].items()}
        ma_d_action_sizes: Dict[str, List[int]] = dataset_info['ma_d_action_sizes']
        ma_c_action_size: Dict[str, int] = dataset_info['ma_c_action_size']
        ep_count: int = dataset_info['ep_count']
        ep_max_len: int = dataset_info['ep_max_len']

        for n, obs_names in ma_obs_names.items():
            obs_names.append('_OFFLINE_ACTION')
        for n, obs_shapes in ma_obs_shapes.items():
            d_action_summed_size = sum(ma_d_action_sizes[n])
            c_action_size = ma_c_action_size[n]
            obs_shapes.append((d_action_summed_size + c_action_size, ))

        self.ma_names = list(ma_obs_names.keys())
        self.ma_obs_names = ma_obs_names
        self.ma_obs_shapes = ma_obs_shapes
        self.ma_d_action_sizes = ma_d_action_sizes
        self.ma_c_action_size = ma_c_action_size
        self.ep_count = ep_count
        self.ep_max_len = ep_max_len  # NOT including next_obs
        #                             # The length of obs_list is `ep_max_len + 1`
        self._logger.info(f'Dataset episode count: {ep_count}')
        self._logger.info(f'Dataset episode max length: {ep_max_len}')

        # Load dataset
        with np.load(self._dataset_dir / 'dataset.npz') as f:
            self._dataset = {k: v for k, v in f.items()}
        for n, obs_names in self.ma_obs_names.items():
            for obs_name in obs_names:
                if obs_name == '_OFFLINE_ACTION':
                    continue
                assert self.ep_count == self._dataset[f'obs_{n}_{obs_name}'].shape[0]
                assert self.ep_max_len + 1 == self._dataset[f'obs_{n}_{obs_name}'].shape[1]

            assert self.ep_max_len == self._dataset[f'action_{n}'].shape[1]

        self._ma_next_ep_index = {n: 0 for n in self.ma_names}

        return ma_obs_names, ma_obs_shapes, ma_d_action_sizes, ma_c_action_size

    def reset(self, reset_config=None):
        self._ma_ep_index = {}
        self._ma_step_index = {}

        for n in self.ma_names:
            self._ma_ep_index[n] = np.arange(self._ma_next_ep_index[n], self._ma_next_ep_index[n] + self.n_envs,
                                             dtype=np.int32)
            self._ma_ep_index[n] = self._ma_ep_index[n] % self.ep_count

            self._ma_next_ep_index[n] += self.n_envs
            self._ma_next_ep_index[n] = self._ma_next_ep_index[n] % self.ep_count

            self._ma_step_index[n] = np.zeros(self.n_envs, dtype=np.int32)

        ma_obs_list = {}
        for n, obs_names in self.ma_obs_names.items():
            ep_index, step_index = self._ma_ep_index[n], self._ma_step_index[n]

            obs_list = []
            for obs_name in obs_names:
                if obs_name == '_OFFLINE_ACTION':
                    obs = self._dataset[f'action_{n}'][ep_index, step_index]
                else:
                    obs = self._dataset[f'obs_{n}_{obs_name}'][ep_index, step_index]
                obs_list.append(obs)
            ma_obs_list[n] = obs_list

        return ma_obs_list

    def step(self, ma_d_action=None, ma_c_action=None):
        ma_obs_list = {}
        ma_reward = {}
        ma_done = {}
        ma_max_step = {}
        ma_padding_mask = {}

        for n, obs_names in self.ma_obs_names.items():
            ep_index, step_index = self._ma_ep_index[n], self._ma_step_index[n]

            obs_list = []
            for obs_name in obs_names:
                if obs_name == '_OFFLINE_ACTION':
                    mask = step_index == self.ep_max_len - 1
                    _step_index = step_index + 1
                    _step_index[mask] = 0
                    obs = self._dataset[f'action_{n}'][ep_index, _step_index]
                else:
                    obs = self._dataset[f'obs_{n}_{obs_name}'][ep_index, step_index + 1]
                obs_list.append(obs)
            ma_obs_list[n] = obs_list

            ma_reward[n] = self._dataset[f'reward_{n}'][ep_index, step_index]
            ma_done[n] = self._dataset[f'done_{n}'][ep_index, step_index]
            ma_max_step[n] = self._dataset[f'max_step_{n}'][ep_index, step_index]
            ma_padding_mask[n] = self._dataset[f'padding_mask_{n}'][ep_index, step_index]

        for n in self.ma_names:
            self._ma_step_index[n] += 1

            mask = self._ma_step_index[n] == self.ep_max_len
            mask = np.logical_or(mask, ma_done[n])

            self._ma_step_index[n][mask] = 0

            terminated_count: int = np.sum(mask)

            new_ep_index = np.arange(self._ma_next_ep_index[n], self._ma_next_ep_index[n] + terminated_count,
                                     dtype=np.int32)
            new_ep_index = new_ep_index % self.ep_count
            self._ma_ep_index[n][mask] = new_ep_index
            self._ma_next_ep_index[n] += terminated_count
            self._ma_next_ep_index[n] = self._ma_next_ep_index[n] % self.ep_count

        return ma_obs_list, ma_reward, ma_done, ma_max_step, ma_padding_mask

    def close(self):
        pass
