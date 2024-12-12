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
                 n_envs: int = 1,
                 model_abs_dir: Path | None = None):
        super().__init__(True, env_name, env_args, n_envs, model_abs_dir)

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
            ma_dataset_info = json.load(f)

        ma_names: List[str] = list(ma_dataset_info.keys())
        ma_obs_names: Dict[str, List[str]] = {n: ma_dataset_info[n]['obs_names'] for n in ma_names}
        ma_obs_shapes: Dict[str, List[Tuple]] = {n: [tuple(s) for s in ma_dataset_info[n]['obs_shapes']] for n in ma_names}
        ma_d_action_sizes: Dict[str, List[int]] = {n: ma_dataset_info[n]['d_action_sizes'] for n in ma_names}
        ma_c_action_size: Dict[str, int] = {n: ma_dataset_info[n]['c_action_size'] for n in ma_names}
        ma_ep_count: Dict[str, int] = {n: ma_dataset_info[n]['ep_count'] for n in ma_names}
        ma_max_step: Dict[str, int] = {n: ma_dataset_info[n]['max_step'] for n in ma_names}

        self.ma_names = ma_names
        self.ma_obs_names = ma_obs_names
        self.ma_obs_shapes = ma_obs_shapes
        self.ma_d_action_sizes = ma_d_action_sizes
        self.ma_c_action_size = ma_c_action_size
        self.ma_ep_count = ma_ep_count
        self.ma_max_step = ma_max_step  # not include the last next_obs

        self._ma_dataset = {}
        for n in ma_names:
            # Load dataset
            with np.load(self._dataset_dir / f'dataset-{ma_dataset_info[n]["path_name"]}.npz') as f:
                self._logger.info(f'Loading dataset {n}...')
                self._ma_dataset[n] = {k: v for k, v in f.items()}

            self._logger.info(f'Dataset {n} episode count: {ma_ep_count[n]}, max step (except the last next_obs): {ma_max_step[n]}')

        for n, obs_names in self.ma_obs_names.items():
            dataset = self._ma_dataset[n]
            for obs_name in obs_names:
                assert self.ma_ep_count[n] == dataset[f'obs-{obs_name}'].shape[0]
                assert self.ma_max_step[n] + 1 == dataset[f'obs-{obs_name}'].shape[1]
                dataset[f'obs-{obs_name}'] = dataset[f'obs-{obs_name}'].astype(np.float32)

            assert self.ma_max_step[n] + 1 == dataset[f'action'].shape[1]
            dataset[f'action'] = dataset[f'action'].astype(np.float32)
            dataset[f'reward'] = dataset[f'reward'].astype(np.float32)

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
            dataset = self._ma_dataset[n]
            ep_index, step_index = self._ma_ep_index[n], self._ma_step_index[n]

            ma_obs_list[n] = [dataset[f'obs-{obs_name}'][ep_index, step_index] for obs_name in obs_names]
            ma_offline_action[n] = dataset[f'action'][ep_index, step_index]

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
            dataset = self._ma_dataset[n]
            ep_index, step_index = self._ma_ep_index[n], self._ma_step_index[n]

            ma_last_reward[n] = dataset[f'reward'][ep_index, step_index]
            ma_done[n] = dataset[f'done'][ep_index, step_index]
            ma_max_reached[n] = dataset[f'max_reached'][ep_index, step_index]
            ma_padding_mask[n] = dataset[f'padding_mask'][ep_index, step_index]

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
            dataset = self._ma_dataset[n]
            ep_index, step_index = self._ma_ep_index[n], self._ma_step_index[n]

            obs_names = self.ma_obs_names[n]
            mask = ma_switch_ep_mask[n]

            ma_obs_list[n] = [dataset[f'obs-{obs_name}'][ep_index, step_index] for obs_name in obs_names]
            ma_offline_action[n] = dataset[f'action'][ep_index, step_index]

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
