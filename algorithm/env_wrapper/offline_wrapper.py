import json
import logging
from pathlib import Path

import numpy as np

if __name__ in ('__main__', '__mp_main__'):
    from env_wrapper import EnvWrapper, DecisionStep, TerminalStep
else:
    from .env_wrapper import EnvWrapper, DecisionStep, TerminalStep


class OfflineWrapper(EnvWrapper):
    def __init__(self,
                 env_name: str,
                 env_args: str | dict | None = None,
                 n_envs: int = 1,
                 model_abs_dir: Path | None = None):
        super().__init__(True, env_name, env_args, n_envs, model_abs_dir)

        self._logger = logging.getLogger('OfflineWrapper')

        dataset_dir = Path(__file__).resolve().parent / env_name
        if 'dataset_path' in self.env_args:
            dataset_dir = Path(self.env_args['dataset_path'])

        self._dataset_dir = dataset_dir

        self._logger.info(f'Dataset directory: {self._dataset_dir}')

    def init(self) -> tuple[dict[str, list[str]],
                            dict[str, list[tuple[int]]],
                            dict[str, list[int]],
                            dict[str, int]]:
        # Load dataset info
        with open(self._dataset_dir / 'info.json') as f:
            ma_dataset_info = json.load(f)

        ma_names: list[str] = list(ma_dataset_info.keys())
        ma_obs_names: dict[str, list[str]] = {n: ma_dataset_info[n]['obs_names'] for n in ma_names}
        ma_obs_shapes: dict[str, list[tuple]] = {n: [tuple(s) for s in ma_dataset_info[n]['obs_shapes']] for n in ma_names}
        ma_d_action_sizes: dict[str, list[int]] = {n: ma_dataset_info[n]['d_action_sizes'] for n in ma_names}
        ma_c_action_size: dict[str, int] = {n: ma_dataset_info[n]['c_action_size'] for n in ma_names}
        ma_ep_count: dict[str, int] = {n: ma_dataset_info[n]['ep_count'] for n in ma_names}
        ma_max_step: dict[str, int] = {n: ma_dataset_info[n]['max_step'] for n in ma_names}

        self.ma_names = ma_names
        self.ma_obs_names = ma_obs_names
        self.ma_obs_shapes = ma_obs_shapes
        self.ma_d_action_sizes = ma_d_action_sizes
        self.ma_c_action_size = ma_c_action_size
        self.ma_ep_count = ma_ep_count
        self.ma_max_step = ma_max_step  # not include the last next_obs

        self._ma_agent_ids = {n: np.arange(self.n_envs) for n in self.ma_names}  # Fixed

        self._ma_dataset_eps = {n: {
            **{f'obs-{obs_name}': [] for obs_name in ma_obs_names[n]},
            'action': [],
            'reward': [],
            'done': [],
            'max_reached': [],
            'padding_mask': []
        } for n in ma_names}

        for n in ma_names:
            self._logger.info(f'Loading dataset {n} ...')
            dataset_eps = self._ma_dataset_eps[n]

            for k in dataset_eps:
                npz = np.load(self._dataset_dir / f'dataset-{ma_dataset_info[n]["path_name"]}-{k}.npz')
                dataset_eps[k] = [npz[f'arr_{i}'] for i in range(ma_ep_count[n])]

            self._logger.info(f'Dataset {n} episode count: {ma_ep_count[n]}, max step (except the last next_obs): {ma_max_step[n]}')

        self._ma_next_ep_index: dict[str, int] = {n: 0 for n in self.ma_names}

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

        ma_obs_list = {}
        ma_offline_action = {}
        for n, obs_names in self.ma_obs_names.items():
            dataset_eps = self._ma_dataset_eps[n]
            ep_indexes, step_indexes = self._ma_ep_index[n], self._ma_step_index[n]

            obs_list = [[dataset_eps[f'obs-{obs_name}'][ep_i][step_i] for ep_i, step_i in zip(ep_indexes, step_indexes)]
                        for obs_name in obs_names]
            ma_obs_list[n] = [np.stack(obs, axis=0) for obs in obs_list]

            offline_action = [dataset_eps['action'][ep_i][step_i] for ep_i, step_i in zip(ep_indexes, step_indexes)]
            ma_offline_action[n] = np.stack(offline_action, axis=0)

        return self._ma_agent_ids, ma_obs_list, ma_offline_action

    def step(self, ma_d_action=None, ma_c_action=None):
        ma_last_reward = {}
        ma_done = {}
        ma_max_reached = {}
        ma_padding_mask = {}

        ma_switch_ep_mask = {}

        # Get previous reward and done
        for n in self.ma_names:
            dataset_eps = self._ma_dataset_eps[n]
            ep_indexes, step_indexes = self._ma_ep_index[n], self._ma_step_index[n]

            last_reward = [dataset_eps['reward'][ep_i][step_i] for ep_i, step_i in zip(ep_indexes, step_indexes)]
            ma_last_reward[n] = np.stack(last_reward, axis=0)

            done = [dataset_eps['done'][ep_i][step_i] for ep_i, step_i in zip(ep_indexes, step_indexes)]
            ma_done[n] = np.stack(done, axis=0)

            max_reached = [dataset_eps['max_reached'][ep_i][step_i] for ep_i, step_i in zip(ep_indexes, step_indexes)]
            ma_max_reached[n] = np.stack(max_reached, axis=0)

            padding_mask = [dataset_eps['padding_mask'][ep_i][step_i] for ep_i, step_i in zip(ep_indexes, step_indexes)]
            ma_padding_mask[n] = np.stack(padding_mask, axis=0)

        # Step forward
        for n in self.ma_names:
            self._ma_step_index[n] += 1

            # Check if the episode is done
            mask = self._ma_step_index[n] == self.ma_max_step[n]
            mask = np.logical_or(mask, ma_done[n])
            mask = np.logical_or(mask, ma_max_reached[n])
            mask = np.logical_or(mask, ma_padding_mask[n])
            ma_switch_ep_mask[n] = mask

            # The episode is done, switch to the next episode
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
        for n, obs_names in self.ma_obs_names.items():
            dataset_eps = self._ma_dataset_eps[n]
            ep_indexes, step_indexes = self._ma_ep_index[n], self._ma_step_index[n]

            obs_list = [[dataset_eps[f'obs-{obs_name}'][ep_i][step_i] for ep_i, step_i in zip(ep_indexes, step_indexes)]
                        for obs_name in obs_names]
            ma_obs_list[n] = [np.stack(obs, axis=0) for obs in obs_list]

            offline_action = [dataset_eps['action'][ep_i][step_i] for ep_i, step_i in zip(ep_indexes, step_indexes)]
            ma_offline_action[n] = np.stack(offline_action, axis=0)

        decisionStep = DecisionStep(
            {n: agent_ids[~ma_switch_ep_mask[n]] for n, agent_ids in self._ma_agent_ids.items()},
            {n: [obs[~ma_switch_ep_mask[n]] for obs in obs_list] for n, obs_list in ma_obs_list.items()},
            {n: reward[~ma_switch_ep_mask[n]] for n, reward in ma_last_reward.items()},
            ma_offline_action={n: offline_action[~ma_switch_ep_mask[n]] for n, offline_action in ma_offline_action.items()}
        )

        terminalStep = TerminalStep(
            {n: agent_ids[ma_switch_ep_mask[n]] for n, agent_ids in self._ma_agent_ids.items()},
            {n: [obs[ma_switch_ep_mask[n]] for obs in obs_list] for n, obs_list in ma_obs_list.items()},
            {n: reward[ma_switch_ep_mask[n]] for n, reward in ma_last_reward.items()},
            {n: max_reached[ma_switch_ep_mask[n]] for n, max_reached in ma_max_reached.items()},
            ma_offline_action={n: offline_action[ma_switch_ep_mask[n]] for n, offline_action in ma_offline_action.items()}
        )

        return decisionStep, terminalStep, False

    def get_episode(self):
        ma_ep_obs_list = {}
        ma_ep_action = {}
        ma_reward = {}
        ma_done = {}

        for n in self.ma_names:
            dataset_eps = self._ma_dataset_eps[n]
            ep_i = self._ma_next_ep_index[n]

            ma_ep_obs_list[n] = [np.expand_dims(dataset_eps[f'obs-{obs_name}'][ep_i], 0)
                                 for obs_name in self.ma_obs_names[n]]
            ma_ep_action[n] = np.expand_dims(dataset_eps['action'][ep_i], 0)
            ma_reward[n] = np.expand_dims(dataset_eps['reward'][ep_i], 0)
            ma_done[n] = np.expand_dims(dataset_eps['done'][ep_i], 0)

            self._ma_next_ep_index[n] = (self._ma_next_ep_index[n] + 1) % self.ma_ep_count[n]

        return (ma_ep_obs_list,
                ma_ep_action,
                ma_reward,
                ma_done)

    def close(self):
        pass


if __name__ == '__main__':
    env = OfflineWrapper('USVEscort', env_args={
        'dataset_path': r'C:\Users\fisher\Documents\Unity\Demonstrations\USVEscort'
    })
    env.init()
    env.reset()
    for i in range(100):
        print(env.step())
