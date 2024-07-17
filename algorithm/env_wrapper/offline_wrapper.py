import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from algorithm.env_wrapper.env_wrapper import DecisionStep, TerminalStep

if __name__ in ('__main__', '__mp_main__'):
    from env_wrapper import EnvWrapper
else:
    from .env_wrapper import EnvWrapper


class OfflineWrapper(EnvWrapper):
    def __init__(self,
                 env_name: str,
                 env_args: Optional[str | Dict] = None,
                 n_envs: int = 1):
        super().__init__(True, env_name, env_args, n_envs)

        self._logger = logging.getLogger('OfflineWrapper')

        dataset_dir = Path(__file__).resolve().parent / env_name
        if 'dataset_path' in self.env_args:
            dataset_dir = Path(self.env_args['dataset_path'])

        self._dataset_dir = dataset_dir

        self._logger.info(f'Dataset directory: {self._dataset_dir}')

    def init(self) -> Tuple[Dict[str, List[str]],
                            Dict[str, List[Tuple[int]]],
                            Dict[str, List[int]],
                            Dict[str, int]]:
        # Load dataset info
        with open(self._dataset_dir / 'info.json') as f:
            dataset_info = json.load(f)

        ma_names: List[str] = dataset_info['ma_names']
        ma_obs_names: Dict[str, List[str]] = dataset_info['ma_obs_names']
        ma_obs_shapes: Dict[str, List[Tuple]] = {n: [tuple(s) for s in obs_shapes] for n, obs_shapes in dataset_info['ma_obs_shapes'].items()}
        ma_d_action_sizes: Dict[str, List[int]] = dataset_info['ma_d_action_sizes']
        ma_c_action_size: Dict[str, int] = dataset_info['ma_c_action_size']
        ma_ep_count: Dict[str, int] = dataset_info['ma_ep_count']
        ma_max_step: Dict[str, int] = dataset_info['ma_max_step']

        self.ma_names = ma_names
        self.ma_obs_names = ma_obs_names
        self.ma_obs_shapes = ma_obs_shapes
        self.ma_d_action_sizes = ma_d_action_sizes
        self.ma_c_action_size = ma_c_action_size
        self.ma_ep_count = ma_ep_count
        self.ma_max_step = ma_max_step  # not include the last next_obs
        for n in ma_names:
            self._logger.info(f'Dataset {n} episode count: {ma_ep_count[n]}, max step (not include the last next_obs): {ma_max_step[n]}')

        # Load dataset
        with np.load(self._dataset_dir / 'dataset.npz') as f:
            self._dataset = {k: v for k, v in f.items()}
        for n, obs_names in self.ma_obs_names.items():
            for obs_name in obs_names:
                assert self.ma_ep_count[n] == self._dataset[f'obs-{n}-{obs_name}'].shape[0]
                assert self.ma_max_step[n] + 1 == self._dataset[f'obs-{n}-{obs_name}'].shape[1]
                self._dataset[f'obs-{n}-{obs_name}'] = self._dataset[f'obs-{n}-{obs_name}'].astype(np.float32)

            assert self.ma_max_step[n] + 1 == self._dataset[f'action-{n}'].shape[1]
            self._dataset[f'action-{n}'] = self._dataset[f'action-{n}'].astype(np.float32)
            self._dataset[f'reward-{n}'] = self._dataset[f'reward-{n}'].astype(np.float32)

        self._ma_next_ep_index: Dict[str, int] = {n: 0 for n in self.ma_names}

        return ma_obs_names, ma_obs_shapes, ma_d_action_sizes, ma_c_action_size

    def reset(self, reset_config=None):
        self._ma_ep_index = {}  # {ma_name: (n_envs, )}
        self._ma_step_index = {}  # {ma_name: (n_envs, )}

        for n in self.ma_names:
            self._ma_ep_index[n] = np.arange(self._ma_next_ep_index[n], self._ma_next_ep_index[n] + self.n_envs, dtype=np.int32)
            self._ma_ep_index[n] = self._ma_ep_index[n] % self.ma_ep_count[n]

            self._ma_next_ep_index[n] += self.n_envs
            self._ma_next_ep_index[n] = self._ma_next_ep_index[n] % self.ma_ep_count[n]

            self._ma_step_index[n] = np.zeros(self.n_envs, dtype=np.int32)

        ma_agent_ids = {n: np.arange(self.n_envs) for n in self.ma_names}
        ma_obs_list = {}
        ma_offline_action = {}
        for n, obs_names in self.ma_obs_names.items():
            ep_index, step_index = self._ma_ep_index[n], self._ma_step_index[n]

            ma_obs_list[n] = [self._dataset[f'obs-{n}-{obs_name}'][ep_index, step_index] for obs_name in obs_names]
            ma_offline_action[n] = self._dataset[f'action-{n}'][ep_index, step_index]

        return ma_agent_ids, ma_obs_list, ma_offline_action

    def step(self, ma_d_action=None, ma_c_action=None):
        ma_agent_ids = {n: np.arange(self.n_envs) for n in self.ma_names}

        ma_last_reward = {}
        ma_done = {}
        ma_max_reached = {}
        ma_padding_mask = {}

        ma_switch_ep_mask = {}

        # Get previous reward and done
        for n in self.ma_names:
            ep_index, step_index = self._ma_ep_index[n], self._ma_step_index[n]
            if n == 'USVGuard':
                print(ep_index, step_index)

            ma_last_reward[n] = self._dataset[f'reward-{n}'][ep_index, step_index]
            ma_done[n] = self._dataset[f'done-{n}'][ep_index, step_index]
            ma_max_reached[n] = self._dataset[f'max_reached-{n}'][ep_index, step_index]
            ma_padding_mask[n] = self._dataset[f'padding_mask-{n}'][ep_index, step_index]

        # Step forward
        for n in self.ma_names:
            self._ma_step_index[n] += 1

            mask = self._ma_step_index[n] == self.ma_max_step[n]
            mask = np.logical_or(mask, ma_done[n])
            mask = np.logical_or(mask, ma_max_reached[n])
            mask = np.logical_or(mask, ma_padding_mask[n])
            ma_switch_ep_mask[n] = mask

            self._ma_step_index[n][mask] = 0

            switch_ep_count: int = np.sum(mask)

            new_ep_index = np.arange(self._ma_next_ep_index[n], self._ma_next_ep_index[n] + switch_ep_count,
                                     dtype=np.int32)
            new_ep_index = new_ep_index % self.ma_ep_count[n]
            self._ma_ep_index[n][mask] = new_ep_index
            self._ma_next_ep_index[n] += switch_ep_count
            self._ma_next_ep_index[n] = self._ma_next_ep_index[n] % self.ma_ep_count[n]

        ma_obs_list = {}
        ma_offline_action = {}

        # Get current observations
        for n in self.ma_names:
            ep_index, step_index = self._ma_ep_index[n], self._ma_step_index[n]

            obs_names = self.ma_obs_names[n]
            mask = ma_switch_ep_mask[n]

            ma_obs_list[n] = [self._dataset[f'obs-{n}-{obs_name}'][ep_index, step_index] for obs_name in obs_names]
            ma_offline_action[n] = self._dataset[f'action-{n}'][ep_index, step_index]

        decisionStep = DecisionStep(
            {n: agent_ids[~ma_switch_ep_mask[n]] for n, agent_ids in ma_agent_ids.items()},
            {n: [obs[~ma_switch_ep_mask[n]] for obs in obs_list] for n, obs_list in ma_obs_list.items()},
            {n: reward[~ma_switch_ep_mask[n]] for n, reward in ma_last_reward.items()},
            ma_offline_action={n: offline_action[~ma_switch_ep_mask[n]] for n, offline_action in ma_offline_action.items()}
        )

        terminalStep = TerminalStep(
            {n: agent_ids[ma_switch_ep_mask[n]] for n, agent_ids in ma_agent_ids.items()},
            {n: [obs[ma_switch_ep_mask[n]] for obs in obs_list] for n, obs_list in ma_obs_list.items()},
            {n: reward[ma_switch_ep_mask[n]] for n, reward in ma_last_reward.items()},
            {n: max_reached[ma_switch_ep_mask[n]] for n, max_reached in ma_max_reached.items()},
            ma_offline_action={n: offline_action[ma_switch_ep_mask[n]] for n, offline_action in ma_offline_action.items()}
        )

        return decisionStep, terminalStep, False

    def close(self):
        pass
