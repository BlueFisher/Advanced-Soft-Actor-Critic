from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

from algorithm.sac_base import SAC_Base
from algorithm.utils.enums import *
from algorithm.utils.operators import gen_pre_n_actions


class Agent:
    reward = 0  # The reward of the *first* completed episode (done == True)
    _last_reward = 0  # The reward of the last episode
    steps = 0  # The step count of the *first* completed episode
    _last_steps = 0  # The step count of the last episode
    done = False  # If has one completed episode
    max_reached = False

    def __init__(self,
                 agent_id: int,
                 obs_shapes: List[Tuple[int, ...]],
                 d_action_sizes: List[int],
                 c_action_size: int,
                 seq_hidden_state_shape: Optional[Tuple[int, ...]] = None,
                 max_return_episode_trans=-1):
        self.agent_id = agent_id
        self.obs_shapes = obs_shapes
        self.d_action_sizes = d_action_sizes
        self.d_action_summed_size = sum(d_action_sizes)
        self.c_action_size = c_action_size
        self.seq_hidden_state_shape = seq_hidden_state_shape
        self.max_return_episode_trans = max_return_episode_trans

        self._tmp_episode_trans = self._generate_empty_episode_trans()

    def _generate_empty_episode_trans(self, episode_length: int = 0) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
        return {
            'index': -np.ones((episode_length, ), dtype=np.int32),
            'obs_list': [np.zeros((episode_length, *s), dtype=np.float32) for s in self.obs_shapes],
            'action': np.zeros((episode_length, self.d_action_summed_size + self.c_action_size), dtype=np.float32),
            'reward': np.zeros((episode_length, ), dtype=np.float32),
            'local_done': np.zeros((episode_length, ), dtype=bool),
            'max_reached': np.zeros((episode_length, ), dtype=bool),
            'next_obs_list': [np.zeros(s, dtype=np.float32) for s in self.obs_shapes],
            'prob': np.zeros((episode_length, self.d_action_summed_size + self.c_action_size), dtype=np.float32),
            'seq_hidden_state': np.zeros((episode_length, *self.seq_hidden_state_shape), dtype=np.float32) if self.seq_hidden_state_shape is not None else None,
        }

    def add_transition(self,
                       obs_list: List[np.ndarray],
                       action: np.ndarray,
                       reward: float,
                       local_done: bool,
                       max_reached: bool,
                       next_obs_list: List[np.ndarray],
                       prob: np.ndarray,
                       seq_hidden_state: Optional[np.ndarray] = None) -> Optional[Dict[str,
                                                                                       Union[np.ndarray, List[np.ndarray]]]]:
        """
        Args:
            obs_list: List([*obs_shapes_i], ...)
            action: [action_size, ]
            reward: float
            local_done: bool
            max_reached: bool
            next_obs_list: List([*obs_shapes_i], ...)
            prob: [action_size, ]
            seq_hidden_state: [*seq_hidden_state_shape]

        Returns:
            ep_indexes (np.int32): [1, episode_len], int
            ep_obses_list: List([1, episode_len, *obs_shapes_i], ...)
            ep_actions: [1, episode_len, action_size]
            ep_rewards: [1, episode_len]
            next_obs_list: List([1, *obs_shapes_i], ...)
            ep_dones (np.bool): [1, episode_len], bool
            ep_probs: [1, episode_len, action_size]
            ep_seq_hidden_states: [1, episode_len, *seq_hidden_state_shape]
        """
        expaned_transition = {
            'index': np.expand_dims(self._last_steps, 0).astype(np.int32),

            'obs_list': [np.expand_dims(o, 0).astype(np.float32) for o in obs_list],
            'action': np.expand_dims(action, 0).astype(np.float32),
            'reward': np.expand_dims(reward, 0).astype(np.float32),
            'local_done': np.expand_dims(local_done, 0).astype(bool),
            'max_reached': np.expand_dims(max_reached, 0).astype(bool),
            'next_obs_list': [o.astype(np.float32) for o in next_obs_list],
            'prob': np.expand_dims(prob, 0).astype(np.float32),
            'seq_hidden_state': np.expand_dims(seq_hidden_state, 0).astype(np.float32) if seq_hidden_state is not None else None,
        }

        for k in self._tmp_episode_trans:
            if k == 'obs_list':
                self._tmp_episode_trans[k] = [
                    np.concatenate([o, t_o]) for o, t_o in zip(self._tmp_episode_trans[k],
                                                               expaned_transition[k])
                ]
            elif k == 'next_obs_list':
                self._tmp_episode_trans[k] = expaned_transition[k]
            elif self._tmp_episode_trans[k] is not None:
                self._tmp_episode_trans[k] = np.concatenate([self._tmp_episode_trans[k],
                                                             expaned_transition[k]])

        if not self.done:
            self.reward += reward
            self.steps += 1

        self._last_reward += reward
        self._last_steps += 1

        self._extra_log(obs_list,
                        action,
                        reward,
                        local_done,
                        max_reached,
                        next_obs_list,
                        prob)

        if local_done:
            self.done = True
            self.max_reached = max_reached
            self._last_reward = 0
            self._last_steps = 0

        if local_done or self.episode_length == self.max_return_episode_trans:
            episode_trans = self.get_episode_trans()
            self._tmp_episode_trans = self._generate_empty_episode_trans()

            return episode_trans

    @property
    def episode_length(self) -> int:
        return len(self._tmp_episode_trans['index'])

    def _extra_log(self,
                   obs_list,
                   action,
                   reward,
                   local_done,
                   max_reached,
                   next_obs_list,
                   prob) -> None:
        pass

    def get_episode_trans(self, force_length: int = None) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
        """
        Returns:
            ep_indexes (np.int32): [1, episode_len]
            ep_obses_list: List([1, episode_len, *obs_shapes_i], ...)
            ep_actions: [1, episode_len, action_size]
            ep_rewards: [1, episode_len]
            next_obs_list: List([1, *obs_shapes_i], ...)
            ep_dones (np.bool): [1, episode_len]
            ep_probs: [1, episode_len, action_size]
            ep_seq_hidden_states: [1, episode_len, *seq_hidden_state_shape]
        """
        tmp = self._tmp_episode_trans.copy()

        if force_length is not None:
            delta = force_length - self.episode_length
            if delta <= 0:
                for k in tmp:
                    if k == 'obs_list':
                        tmp[k] = [o[-force_length:] for o in tmp[k]]
                    elif k == 'next_obs_list':
                        pass
                    elif tmp[k] is not None:
                        tmp[k] = tmp[k][-force_length:]
            else:
                tmp_empty = self._generate_empty_episode_trans(delta)
                for k in tmp:
                    if k == 'obs_list':
                        tmp[k] = [np.concatenate([t_o, o]) for t_o, o in zip(tmp_empty[k], tmp[k])]
                    elif k == 'next_obs_list':
                        pass
                    elif tmp[k] is not None:
                        tmp[k] = np.concatenate([tmp_empty[k], tmp[k]])

        ep_indexes = np.expand_dims(tmp['index'], 0)
        # [1, episode_len]
        ep_obses_list = [np.expand_dims(o, 0) for o in tmp['obs_list']]
        # List([1, episode_len, *obs_shape_si], ...)
        ep_actions = np.expand_dims(tmp['action'], 0)  # [1, episode_len, action_size]
        ep_rewards = np.expand_dims(tmp['reward'], 0)  # [1, episode_len]
        next_obs_list = [np.expand_dims(o, 0) for o in tmp['next_obs_list']]
        # List([1, *obs_shapes_i], ...)
        ep_dones = np.expand_dims(np.logical_and(tmp['local_done'],
                                                 ~tmp['max_reached']),
                                  0)  # [1, episode_len]
        ep_probs = np.expand_dims(tmp['prob'], 0)  # [1, episode_len, action_size]
        ep_seq_hidden_states = np.expand_dims(tmp['seq_hidden_state'], 0) if tmp['seq_hidden_state'] is not None else None
        # [1, episode_len, *seq_hidden_state_shape]

        return {
            'l_indexes': ep_indexes,
            'l_obses_list': ep_obses_list,
            'l_actions': ep_actions,
            'l_rewards': ep_rewards,
            'next_obs_list': next_obs_list,
            'l_dones': ep_dones,
            'l_probs': ep_probs,
            'l_seq_hidden_states': ep_seq_hidden_states
        }

    def is_empty(self) -> bool:
        return self.episode_length == 0

    def clear(self) -> None:
        self.reward = 0
        self.steps = 0
        self.done = False
        self.max_reached = False
        self._last_reward = 0
        self._last_steps = 0
        self._tmp_episode_trans = self._generate_empty_episode_trans()

    def reset(self) -> None:
        """
        The agent may continue in a new iteration but save its last status
        """
        self.reward = self._last_reward
        self.steps = self._last_steps
        self.done = False
        self.max_reached = False


class AgentManager:
    def __init__(self,
                 name: str,
                 obs_names: List[str],
                 obs_shapes: List[Tuple[int]],
                 d_action_sizes: List[int],
                 c_action_size: int):
        self.name = name
        self.obs_names = obs_names
        self.obs_shapes = obs_shapes
        self.d_action_sizes = d_action_sizes
        self.d_action_summed_size = sum(d_action_sizes)
        self.c_action_size = c_action_size
        self.action_size = self.d_action_summed_size + self.c_action_size

        self.rl = None
        self.seq_encoder = None

        self._data = {}

    def __getitem__(self, k: str):
        return self._data[k]

    def __setitem__(self, k: str, v):
        self._data[k] = v

    def set_config(self, config) -> None:
        self.config = deepcopy(config)

    def set_model_abs_dir(self, model_abs_dir: Path) -> None:
        model_abs_dir.mkdir(parents=True, exist_ok=True)
        self.model_abs_dir = model_abs_dir

    def set_rl(self, rl: SAC_Base) -> None:
        self.rl = rl
        self.seq_encoder = rl.seq_encoder

    def pre_run(self, num_agents: int) -> None:
        if self.rl is not None:
            self['initial_pre_action'] = self.rl.get_initial_action(num_agents)  # [n_envs, action_size]
            self['pre_action'] = self['initial_pre_action']
            if self.seq_encoder is not None:
                self['initial_seq_hidden_state'] = self.rl.get_initial_seq_hidden_state(num_agents)  # [n_envs, *seq_hidden_state_shape]
                self['seq_hidden_state'] = self['initial_seq_hidden_state']

        self.agents: List[Agent] = [
            Agent(i,
                  self.obs_shapes,
                  self.d_action_sizes,
                  self.c_action_size,
                  seq_hidden_state_shape=self.rl.seq_hidden_state_shape
                  if self.seq_encoder is not None else None)
            for i in range(num_agents)
        ]

    # Reset
    def set_obs_list(self, obs_list):
        self['obs_list'] = obs_list
        self['padding_mask'] = np.zeros(len(self.agents), dtype=bool)

    def get_action(self,
                   disable_sample: bool = False,
                   force_rnd_if_available: bool = False) -> None:
        if self.seq_encoder == SEQ_ENCODER.RNN:
            action, prob, next_seq_hidden_state = self.rl.choose_rnn_action(
                obs_list=self['obs_list'],
                pre_action=self['pre_action'],
                rnn_state=self['seq_hidden_state'],

                disable_sample=disable_sample,
                force_rnd_if_available=force_rnd_if_available
            )

        elif self.seq_encoder == SEQ_ENCODER.ATTN:
            ep_length = min(512, max([a.episode_length for a in self.agents]))

            all_episode_trans = [a.get_episode_trans(ep_length).values() for a in self.agents]
            (all_ep_indexes,
             all_ep_obses_list,
             all_ep_actions,
             all_all_ep_rewards,
             all_next_obs_list,
             all_ep_dones,
             all_ep_probs,
             all_ep_attn_states) = zip(*all_episode_trans)

            ep_indexes = np.concatenate(all_ep_indexes)
            ep_obses_list = [np.concatenate(o) for o in zip(*all_ep_obses_list)]
            ep_actions = np.concatenate(all_ep_actions)
            ep_attn_states = np.concatenate(all_ep_attn_states)

            if ep_indexes.shape[1] == 0:
                ep_indexes = np.zeros((ep_indexes.shape[0], 1), dtype=ep_indexes.dtype)
            else:
                ep_indexes = np.concatenate([ep_indexes, ep_indexes[:, -1:] + 1], axis=1)
            ep_padding_masks = ep_indexes == -1
            ep_obses_list = [np.concatenate([o, np.expand_dims(t_o, 1)], axis=1)
                             for o, t_o in zip(ep_obses_list, self['obs_list'])]
            ep_pre_actions = gen_pre_n_actions(ep_actions, True)
            ep_attn_states = np.concatenate([ep_attn_states,
                                             np.expand_dims(self['seq_hidden_state'], 1)], axis=1)

            action, prob, next_seq_hidden_state = self.rl.choose_attn_action(
                ep_indexes=ep_indexes,
                ep_padding_masks=ep_padding_masks,
                ep_obses_list=ep_obses_list,
                ep_pre_actions=ep_pre_actions,
                ep_attn_states=ep_attn_states,

                disable_sample=disable_sample,
                force_rnd_if_available=force_rnd_if_available
            )

        else:
            action, prob = self.rl.choose_action(self['obs_list'],

                                                 disable_sample=disable_sample,
                                                 force_rnd_if_available=force_rnd_if_available)
            next_seq_hidden_state = None

        self['action'] = action
        self['d_action'] = action[..., :self.d_action_summed_size]
        self['c_action'] = action[..., -self.c_action_size:]
        self['prob'] = prob
        self['next_seq_hidden_state'] = next_seq_hidden_state

    def get_test_action(self) -> None:
        action = np.zeros((len(self.agents), self.action_size), dtype=np.float32)
        d_action = c_action = None

        if self.d_action_sizes:
            d_action_list = [np.random.randint(0, d_action_size, size=len(self.agents))
                             for d_action_size in self.d_action_sizes]
            d_action_list = [np.eye(d_action_size, dtype=np.int32)[d_action]
                             for d_action, d_action_size in zip(d_action_list, self.d_action_sizes)]
            d_action = np.concatenate(d_action_list, axis=-1)
            action[:, :self.d_action_summed_size] = d_action

        if self.c_action_size:
            c_action = np.random.randn(len(self.agents), self.c_action_size)
            # c_action = np.ones((len(self.agents), self.c_action_size), dtype=np.float32)
            action[:, -self.c_action_size:] = c_action

        self['action'] = action
        self['d_action'] = d_action
        self['c_action'] = c_action
        self['prob'] = np.random.rand(len(self.agents), self.action_size)

    def set_env_step(self,
                     next_obs_list: List[np.ndarray],
                     reward: np.ndarray,
                     local_done: np.ndarray,
                     max_reached: np.ndarray) -> None:
        episode_trans_list = []

        for i, a in enumerate(self.agents):
            if not self['padding_mask'][i]:
                episode_trans = a.add_transition(
                    obs_list=[o[i] for o in self['obs_list']],
                    action=self['action'][i],
                    reward=reward[i],
                    local_done=local_done[i],
                    max_reached=max_reached[i],
                    next_obs_list=[o[i] for o in next_obs_list],
                    prob=self['prob'][i],
                    seq_hidden_state=self['seq_hidden_state'][i] if self.seq_encoder is not None else None,
                )
                if episode_trans is not None:
                    episode_trans_list.append(episode_trans)

        self['episode_trans_list'] = episode_trans_list

    def train(self) -> int:
        if len(self['episode_trans_list']) != 0:
            # ep_indexes,
            # ep_obses_list, ep_actions, ep_rewards, next_obs_list, ep_dones, ep_probs,
            # ep_seq_hidden_states
            for episode_trans in self['episode_trans_list']:
                self.rl.put_episode(**episode_trans)

        trained_steps = self.rl.train()

        return trained_steps

    def post_step(self, next_obs_list, local_done, next_padding_mask) -> None:
        self['obs_list'] = next_obs_list
        self['padding_mask'] = next_padding_mask
        self['pre_action'] = self['action']
        self['pre_action'][local_done] = self['initial_pre_action'][local_done]
        if self.seq_encoder is not None:
            self['seq_hidden_state'] = self['next_seq_hidden_state']
            self['seq_hidden_state'][local_done] = self['initial_seq_hidden_state'][local_done]


class MultiAgentsManager:
    def __init__(self,
                 ma_obs_names: dict,
                 ma_obs_shapes: dict,
                 ma_d_action_sizes: dict,
                 ma_c_action_size: dict,
                 model_abs_dir: Path):
        self._ma_manager: Dict[str, AgentManager] = {}
        for n in ma_obs_shapes:
            self._ma_manager[n] = AgentManager(n,
                                               ma_obs_names[n],
                                               ma_obs_shapes[n],
                                               ma_d_action_sizes[n],
                                               ma_c_action_size[n])

            if len(ma_obs_shapes) == 1:
                self._ma_manager[n].set_model_abs_dir(model_abs_dir)
            else:
                self._ma_manager[n].set_model_abs_dir(model_abs_dir / n.replace('?', '-'))

    def __iter__(self) -> Iterator[Tuple[str, AgentManager]]:
        return iter(self._ma_manager.items())

    def __getitem__(self, k) -> AgentManager:
        return self._ma_manager[k]

    def __len__(self) -> int:
        return len(self._ma_manager)

    def pre_run(self, ma_num_agents) -> None:
        for n, mgr in self:
            mgr.pre_run(ma_num_agents[n])

    def is_max_reached(self) -> bool:
        return any([any([a.max_reached for a in mgr.agents]) for n, mgr in self])

    def is_done(self) -> bool:
        return all([all([a.done for a in mgr.agents]) for n, mgr in self])

    def set_obs_list(self, ma_obs_list) -> None:
        for n, mgr in self:
            mgr.set_obs_list(ma_obs_list[n])

    def clear(self) -> None:
        for n, mgr in self:
            for a in mgr.agents:
                a.clear()

    def reset(self) -> None:
        for n, mgr in self:
            for a in mgr.agents:
                a.reset()

    def set_train_mode(self, train_mode: bool = True):
        for n, mgr in self:
            mgr.rl.train_mode = train_mode

    def get_ma_action(self,
                      disable_sample: bool = False,
                      force_rnd_if_available: bool = False) -> Tuple[Dict[str, np.ndarray],
                                                                     Dict[str, np.ndarray]]:
        for n, mgr in self:
            mgr.get_action(disable_sample, force_rnd_if_available)

        ma_d_action = {n: mgr['d_action'] for n, mgr in self}
        ma_c_action = {n: mgr['c_action'] for n, mgr in self}

        return ma_d_action, ma_c_action

    def get_test_ma_action(self) -> Tuple[Dict[str, np.ndarray],
                                          Dict[str, np.ndarray]]:
        for n, mgr in self:
            mgr.get_test_action()

        ma_d_action = {n: mgr['d_action'] for n, mgr in self}
        ma_c_action = {n: mgr['c_action'] for n, mgr in self}

        return ma_d_action, ma_c_action

    def set_ma_env_step(self,
                        ma_next_obs_list,
                        ma_reward,
                        ma_local_done,
                        ma_max_reached) -> None:
        for n, mgr in self:
            mgr.set_env_step(ma_next_obs_list[n],
                             ma_reward[n],
                             ma_local_done[n],
                             ma_max_reached[n])

    def put_episode(self) -> None:
        for n, mgr in self:
            mgr.put_episode()

    def train(self, trained_steps: int) -> int:
        for n, mgr in self:
            trained_steps = max(mgr.train(), trained_steps)

        return trained_steps

    def post_step(self, ma_next_obs_list, ma_local_done, ma_next_padding_mask) -> None:
        for n, mgr in self:
            mgr.post_step(ma_next_obs_list[n], ma_local_done[n], ma_next_padding_mask[n])

    def save_model(self, save_replay_buffer=False) -> None:
        for n, mgr in self:
            mgr.rl.save_model(save_replay_buffer)
