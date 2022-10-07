from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from algorithm.utils.enums import *
from algorithm.utils.operators import gen_pre_n_actions

from .. import agent
from ..agent import Agent, AgentManager, MultiAgentsManager
from .oc_base import OC_Base


class OC_Agent(Agent):
    def __init__(self, agent_id: int,
                 obs_shapes: List[Tuple],
                 action_size: int,
                 seq_hidden_state_shape=None,
                 max_return_episode_trans=-1):
        self.option_index_count = defaultdict(int)
        super().__init__(agent_id,
                         obs_shapes,
                         action_size,
                         seq_hidden_state_shape,
                         max_return_episode_trans)

    def _generate_empty_episode_trans(self, episode_length: int = 0):
        return {
            'index': -np.ones((episode_length, ), dtype=int),
            'padding_mask': np.ones((episode_length, ), dtype=bool),
            'obs_list': [np.zeros((episode_length, *s), dtype=np.float32) for s in self.obs_shapes],
            'option_index': np.full((episode_length, ), -1, dtype=np.int8),
            'action': np.zeros((episode_length, self.action_size), dtype=np.float32),
            'reward': np.zeros((episode_length, ), dtype=np.float32),
            'local_done': np.zeros((episode_length, ), dtype=bool),
            'max_reached': np.zeros((episode_length, ), dtype=bool),
            'next_obs_list': [np.zeros(s, dtype=np.float32) for s in self.obs_shapes],
            'prob': np.zeros((episode_length, ), dtype=np.float32),
            'seq_hidden_state': np.zeros((episode_length, *self.seq_hidden_state_shape), dtype=np.float32) if self.seq_hidden_state_shape is not None else None,
        }

    def add_transition(self,
                       obs_list: List[np.ndarray],
                       option_index: np.ndarray,
                       action: np.ndarray,
                       reward: float,
                       local_done: bool,
                       max_reached: bool,
                       next_obs_list: List[np.ndarray],
                       prob: float,
                       is_padding: bool = False,
                       seq_hidden_state: Optional[np.ndarray] = None):
        """
        Args:
            obs_list: List([*obs_shapes_i], ...)
            action: [action_size, ]
            reward: float
            local_done: bool
            max_reached: bool
            next_obs_list: List([*obs_shapes_i], ...)
            prob: float
            seq_hidden_state: [*seq_hidden_state_shape]

        Returns:
            ep_indexes: [1, episode_len], int
            ep_padding_masks: [1, episode_len], bool
            ep_obses_list: List([1, episode_len, *obs_shapes_i], ...), np.float32
            ep_actions: [1, episode_len, action_size], np.float32
            ep_rewards: [1, episode_len], np.float32
            next_obs_list: List([1, *obs_shapes_i], ...), np.float32
            ep_dones: [1, episode_len], bool
            ep_probs: [1, episode_len], np.float32
            ep_seq_hidden_states: [1, episode_len, *seq_hidden_state_shape], np.float32
        """
        expaned_transition = {
            'index': np.expand_dims(self._last_steps if not is_padding else -1, 0),
            'padding_mask': np.expand_dims(is_padding, 0).astype(bool),

            'obs_list': [np.expand_dims(o, 0).astype(np.float32) for o in obs_list],
            'option_index': np.expand_dims(option_index, 0).astype(np.int8),
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
            self.option_index_count[int(option_index)] += 1
            self.reward += reward
            self.steps += 1
        self._last_reward += reward

        if not is_padding:
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

    def get_episode_trans(self, force_length: int = None):
        """
        Returns:
            ep_indexes: [1, episode_len], int
            ep_padding_masks: [1, episode_len], bool
            ep_obses_list: List([1, episode_len, *obs_shapes_i], ...), np.float32
            ep_actions: [1, episode_len, action_size], np.float32
            ep_rewards: [1, episode_len], np.float32
            next_obs_list: List([1, *obs_shapes_i], ...), np.float32
            ep_dones: [1, episode_len], bool
            ep_probs: [1, episode_len], np.float32
            ep_seq_hidden_states: [1, episode_len, *seq_hidden_state_shape], np.float32
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
        ep_padding_masks = np.expand_dims(tmp['padding_mask'], 0)
        # [1, episode_len]
        ep_obses_list = [np.expand_dims(o, 0) for o in tmp['obs_list']]
        # List([1, episode_len, *obs_shape_si], ...)
        ep_option_index = np.expand_dims(tmp['option_index'], 0)  # [1, episode_len]
        ep_actions = np.expand_dims(tmp['action'], 0)  # [1, episode_len, action_size]
        ep_rewards = np.expand_dims(tmp['reward'], 0)  # [1, episode_len]
        next_obs_list = [np.expand_dims(o, 0) for o in tmp['next_obs_list']]
        # List([1, *obs_shapes_i], ...)
        ep_dones = np.expand_dims(np.logical_and(tmp['local_done'],
                                                 ~tmp['max_reached']),
                                  0)  # [1, episode_len]
        ep_probs = np.expand_dims(tmp['prob'], 0)  # [1, episode_len]
        ep_seq_hidden_states = np.expand_dims(tmp['seq_hidden_state'], 0) if tmp['seq_hidden_state'] is not None else None
        # [1, episode_len, *seq_hidden_state_shape]

        return (ep_indexes,
                ep_padding_masks,
                ep_obses_list,
                ep_option_index,
                ep_actions,
                ep_rewards,
                next_obs_list,
                ep_dones,
                ep_probs,
                ep_seq_hidden_states)

    def clear(self):
        for k in self.option_index_count:
            self.option_index_count[k] = 0
        return super().clear()

    def reset(self):
        for k in self.option_index_count:
            self.option_index_count[k] = 0
        return super().reset()


class OC_AgentManager(AgentManager):
    def set_rl(self, rl: OC_Base):
        self.rl = rl
        self.seq_encoder = rl.seq_encoder

    def pre_run(self, num_agents: int):
        self['initial_option_index'] = self.rl.get_initial_option_index(num_agents)  # [n_agents, action_size]
        self['pre_option_index'] = self['initial_option_index']
        self['initial_pre_action'] = self.rl.get_initial_action(num_agents)  # [n_agents, action_size]
        self['pre_action'] = self['initial_pre_action']
        if self.seq_encoder is not None:
            self['initial_seq_hidden_state'] = self.rl.get_initial_seq_hidden_state(num_agents)  # [n_agents, *seq_hidden_state_shape]
            self['seq_hidden_state'] = self['initial_seq_hidden_state']

        self.agents: List[OC_Agent] = [
            OC_Agent(i, self.obs_shapes, self.action_size,
                     seq_hidden_state_shape=self.rl.seq_hidden_state_shape
                     if self.seq_encoder is not None else None)
            for i in range(num_agents)
        ]

    def get_action(self,
                   disable_sample: bool = False,
                   force_rnd_if_available: bool = False):
        if self.seq_encoder == SEQ_ENCODER.RNN:
            (option_index,
             action,
             prob,
             next_seq_hidden_state) = self.rl.choose_rnn_action(self['obs_list'],
                                                                pre_option_index=self['pre_option_index'],
                                                                pre_action=self['pre_action'],
                                                                rnn_state=self['seq_hidden_state'])

        else:
            option_index, action, prob = self.rl.choose_action(self['obs_list'],
                                                               pre_option_index=self['pre_option_index'])
            next_seq_hidden_state = None

        self['option_index'] = option_index
        self['action'] = action
        self['d_action'] = action[..., :self.d_action_size]
        self['c_action'] = action[..., self.d_action_size:]
        self['prob'] = prob
        self['next_seq_hidden_state'] = next_seq_hidden_state

    def set_env_step(self,
                     next_obs_list,
                     reward,
                     local_done,
                     max_reached):

        episode_trans_list = [
            a.add_transition(
                obs_list=[o[i] for o in self['obs_list']],
                option_index=self['option_index'][i],
                action=self['action'][i],
                reward=reward[i],
                local_done=local_done[i],
                max_reached=max_reached[i],
                next_obs_list=[o[i] for o in next_obs_list],
                prob=self['prob'][i],
                is_padding=False,
                seq_hidden_state=self['seq_hidden_state'][i] if self.seq_encoder is not None else None,
            ) for i, a in enumerate(self.agents)
        ]

        self['episode_trans_list'] = [t for t in episode_trans_list if t is not None]

    def post_step(self, next_obs_list, local_done):
        super().post_step(next_obs_list, local_done)

        self['pre_option_index'] = self['option_index']


class OC_MultiAgentsManager(MultiAgentsManager):
    def burn_in_padding(self):
        for n, mgr in self:
            for a in [a for a in mgr.agents if a.is_empty()]:
                for _ in range(mgr.rl.burn_in_step):
                    a.add_transition(
                        obs_list=[np.zeros(t, dtype=np.float32) for t in mgr.obs_shapes],
                        option_index=-1,
                        action=mgr['initial_pre_action'][0],
                        reward=0.,
                        local_done=False,
                        max_reached=False,
                        next_obs_list=[np.zeros(t, dtype=np.float32) for t in mgr.obs_shapes],
                        prob=0.,
                        is_padding=True,
                        seq_hidden_state=mgr['initial_seq_hidden_state'][0]
                    )

    def get_option(self):
        return {n: mgr['option_index'] for n, mgr in self}

agent.Agent = OC_Agent
agent.AgentManager = OC_AgentManager
