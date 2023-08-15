from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np

from algorithm.oc.option_selector_base import OptionSelectorBase
from algorithm.utils.enums import *
from algorithm.utils.operators import gen_pre_n_actions

from .. import agent
from ..agent import Agent, AgentManager, MultiAgentsManager


class OC_Agent(Agent):
    _last_option_changed_index = -1  # The latest option start index
    _last_option_index = -1  # The latest option index

    def __init__(self,
                 agent_id: int,
                 obs_shapes: List[Tuple],
                 d_action_sizes: List[int],
                 c_action_size: int,
                 seq_hidden_state_shape=None,
                 low_seq_hidden_state_shape=None,
                 max_return_episode_trans=-1):

        self.low_seq_hidden_state_shape = low_seq_hidden_state_shape

        self._tmp_option_changed_indexes = []

        super().__init__(agent_id,
                         obs_shapes,
                         d_action_sizes,
                         c_action_size,
                         seq_hidden_state_shape,
                         max_return_episode_trans)

    def _generate_empty_episode_trans(self, episode_length: int = 0):
        return {
            'index': -np.ones((episode_length, ), dtype=np.int32),
            'obs_list': [np.zeros((episode_length, *s), dtype=np.float32) for s in self.obs_shapes],
            'option_index': np.full((episode_length, ), -1, dtype=np.int8),
            'option_changed_index': np.full((episode_length, ), -1, dtype=np.int32),
            'action': np.zeros((episode_length, self.d_action_summed_size + self.c_action_size), dtype=np.float32),
            'reward': np.zeros((episode_length, ), dtype=np.float32),
            'local_done': np.zeros((episode_length, ), dtype=bool),
            'max_reached': np.zeros((episode_length, ), dtype=bool),
            'next_obs_list': [np.zeros(s, dtype=np.float32) for s in self.obs_shapes],
            'prob': np.zeros((episode_length, self.d_action_summed_size + self.c_action_size), dtype=np.float32),
            'seq_hidden_state': np.zeros((episode_length, *self.seq_hidden_state_shape), dtype=np.float32) if self.seq_hidden_state_shape is not None else None,
            'low_seq_hidden_state': np.zeros((episode_length, *self.low_seq_hidden_state_shape), dtype=np.float32) if self.low_seq_hidden_state_shape is not None else None,
        }

    def add_transition(self,
                       obs_list: List[np.ndarray],
                       option_index: int,
                       action: np.ndarray,
                       reward: float,
                       local_done: bool,
                       max_reached: bool,
                       next_obs_list: List[np.ndarray],
                       prob: np.ndarray,
                       seq_hidden_state: Optional[np.ndarray] = None,
                       low_seq_hidden_state: Optional[np.ndarray] = None):
        """
        Args:
            obs_list: List([*obs_shapes_i], ...)
            option_index: int
            action: [action_size, ]
            reward: float
            local_done: bool
            max_reached: bool
            next_obs_list: List([*obs_shapes_i], ...)
            prob: [action_size, ]
            seq_hidden_state: [*seq_hidden_state_shape]
            low_seq_hidden_state: [*low_seq_hidden_state_shape]

        Returns:
            ep_indexes (np.int32): [1, episode_len], int
            ep_obses_list: List([1, episode_len, *obs_shapes_i], ...)
            ep_option_indexes (np.int8): [1, episode_len]
            ep_actions: [1, episode_len, action_size]
            ep_rewards: [1, episode_len]
            next_obs_list: List([1, *obs_shapes_i], ...)
            ep_dones (np.bool): [1, episode_len], bool
            ep_probs: [1, episode_len, action_size]
            ep_seq_hidden_states: [1, episode_len, *seq_hidden_state_shape]
            ep_low_seq_hidden_states: [1, episode_len, *low_seq_hidden_state_shape]
        """
        if option_index == self._last_option_index:
            option_changed_index = self._last_option_changed_index
        else:
            option_changed_index = self._last_steps
            self._tmp_option_changed_indexes.append(self._last_steps)

        expaned_transition = {
            'index': np.expand_dims(self._last_steps, 0).astype(np.int32),

            'obs_list': [np.expand_dims(o, 0).astype(np.float32) for o in obs_list],
            'option_index': np.expand_dims(option_index, 0).astype(np.int8),
            'option_changed_index': np.expand_dims(option_changed_index, 0).astype(np.int32),
            'action': np.expand_dims(action, 0).astype(np.float32),
            'reward': np.expand_dims(reward, 0).astype(np.float32),
            'local_done': np.expand_dims(local_done, 0).astype(bool),
            'max_reached': np.expand_dims(max_reached, 0).astype(bool),
            'next_obs_list': [o.astype(np.float32) for o in next_obs_list],
            'prob': np.expand_dims(prob, 0).astype(np.float32),
            'seq_hidden_state': np.expand_dims(seq_hidden_state, 0).astype(np.float32) if seq_hidden_state is not None else None,
            'low_seq_hidden_state': np.expand_dims(low_seq_hidden_state, 0).astype(np.float32) if low_seq_hidden_state is not None else None,
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

        self._last_option_index = option_index
        self._last_option_changed_index = option_changed_index

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
            self._last_option_index = -1
            self._last_option_changed_index = -1

        if local_done or self.episode_length == self.max_return_episode_trans:
            episode_trans = self.get_episode_trans()
            self._tmp_episode_trans = self._generate_empty_episode_trans()
            self._tmp_option_changed_indexes = []

            return episode_trans

    def get_episode_trans(self,
                          force_length: int = None):
        """
        Returns:
            ep_indexes (np.int32): [1, episode_len]
            ep_obses_list: List([1, episode_len, *obs_shapes_i], ...)
            ep_option_indexes (np.int8): [1, episode_len]
            ep_option_changed_indexes (np.int32): [1, episode_len]
            ep_actions: [1, episode_len, action_size]
            ep_rewards: [1, episode_len]
            next_obs_list: List([1, *obs_shapes_i], ...)
            ep_dones (np.bool): [1, episode_len]
            ep_probs: [1, episode_len]
            ep_seq_hidden_states: [1, episode_len, *seq_hidden_state_shape]
            ep_low_seq_hidden_states: [1, episode_len, *low_seq_hidden_state_shape]
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
        ep_option_indexes = np.expand_dims(tmp['option_index'], 0)  # [1, episode_len]
        ep_option_changed_indexes = np.expand_dims(tmp['option_changed_index'], 0)  # [1, episode_len]
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
        ep_low_seq_hidden_states = np.expand_dims(tmp['low_seq_hidden_state'], 0) if tmp['low_seq_hidden_state'] is not None else None
        # [1, episode_len, *seq_hidden_state_shape]

        return {
            'ep_indexes': ep_indexes,
            'ep_obses_list': ep_obses_list,
            'ep_option_indexes': ep_option_indexes,
            'ep_option_changed_indexes': ep_option_changed_indexes,
            'ep_actions': ep_actions,
            'ep_rewards': ep_rewards,
            'next_obs_list': next_obs_list,
            'ep_dones': ep_dones,
            'ep_probs': ep_probs,
            'ep_seq_hidden_states': ep_seq_hidden_states,
            'ep_low_seq_hidden_states': ep_low_seq_hidden_states
        }

    @property
    def key_trans_length(self):
        return len(self._tmp_option_changed_indexes)

    def get_key_trans(self, force_length: int):
        """
        Returns:
            key_indexes (np.int32): [1, key_len]
            key_padding_masks (np.bool): [1, key_len]
            key_obses_list: List([1, key_len, *obs_shapes_i], ...)
            key_seq_hidden_states: [1, key_len, *seq_hidden_state_shape]
        """
        tmp = self._tmp_episode_trans
        option_changed_indexes = self._tmp_option_changed_indexes

        ep_indexes = np.expand_dims(tmp['index'], 0)
        # [1, episode_len]
        ep_obses_list = [np.expand_dims(o, 0) for o in tmp['obs_list']]
        # List([1, episode_len, *obs_shape_si], ...)
        ep_seq_hidden_states = np.expand_dims(tmp['seq_hidden_state'], 0)
        # [1, episode_len, *seq_hidden_state_shape]

        key_indexes = ep_indexes[:, option_changed_indexes]
        key_padding_masks = np.zeros_like(key_indexes, dtype=bool)
        key_obses_list = [o[:, option_changed_indexes] for o in ep_obses_list]
        key_seq_hidden_states = ep_seq_hidden_states[:, option_changed_indexes]

        assert force_length >= self.key_trans_length

        delta = force_length - self.key_trans_length
        if delta > 0:
            delta_key_indexes = -np.ones((1, delta), dtype=np.int32)
            delta_padding_masks = np.ones((1, delta), dtype=bool)  # `True` indicates ignored
            delta_obses_list = [np.zeros((1, delta, *s), dtype=np.float32) for s in self.obs_shapes]
            delta_seq_hidden_states = np.zeros((1, delta, *self.seq_hidden_state_shape), dtype=np.float32)

            key_indexes = np.concatenate([delta_key_indexes, key_indexes], axis=1)
            key_padding_masks = np.concatenate([delta_padding_masks, key_padding_masks], axis=1)
            key_obses_list = [np.concatenate([d_o, o], axis=1) for d_o, o in zip(delta_obses_list, key_obses_list)]
            key_seq_hidden_states = np.concatenate([delta_seq_hidden_states, key_seq_hidden_states], axis=1)

        return (
            key_indexes,
            key_padding_masks,
            key_obses_list,
            key_seq_hidden_states,
        )

    def get_last_index(self) -> np.ndarray:
        index = self._tmp_episode_trans['index'][-1:]
        if len(index) == 0:
            index = np.full((1, ), -1, dtype=np.int32)
        index = np.expand_dims(index, 0)

        return index

    def clear(self):
        self._last_option_changed_index = -1
        self._last_option_index = -1
        self._tmp_option_changed_indexes = []
        return super().clear()


class OC_AgentManager(AgentManager):
    def set_rl(self, rl: OptionSelectorBase):
        self.rl = rl
        self.seq_encoder = rl.seq_encoder
        self.use_dilation = rl.use_dilation

    def pre_run(self, num_agents: int):
        self['option_index'] = np.full(num_agents, -1, dtype=np.int8)
        self['initial_option_index'] = self.rl.get_initial_option_index(num_agents)  # [n_envs, action_size]
        self['pre_option_index'] = self['initial_option_index']
        self['initial_pre_action'] = self.rl.get_initial_action(num_agents)  # [n_envs, action_size]
        self['pre_action'] = self['initial_pre_action']
        if self.seq_encoder is not None:
            self['initial_seq_hidden_state'] = self.rl.get_initial_seq_hidden_state(num_agents)  # [n_envs, *seq_hidden_state_shape]
            self['seq_hidden_state'] = self['initial_seq_hidden_state']
            if self.use_dilation:
                self['key_seq_hidden_state'] = self['initial_seq_hidden_state'].copy()

            self['initial_low_seq_hidden_state'] = self.rl.get_initial_low_seq_hidden_state(num_agents)  # [n_envs, *los_seq_hidden_state_shape]
            self['low_seq_hidden_state'] = self['initial_low_seq_hidden_state']

        self.agents: List[OC_Agent] = [
            OC_Agent(i,
                     self.obs_shapes,
                     self.d_action_sizes,
                     self.c_action_size,
                     seq_hidden_state_shape=self.rl.seq_hidden_state_shape
                     if self.seq_encoder is not None else None,
                     low_seq_hidden_state_shape=self.rl.low_seq_hidden_state_shape
                     if self.seq_encoder is not None else None)
            for i in range(num_agents)
        ]

    def get_action(self,
                   disable_sample: bool = False,
                   force_rnd_if_available: bool = False):
        if self.seq_encoder == SEQ_ENCODER.RNN and not self.use_dilation:
            (option_index,
             action,
             prob,
             next_seq_hidden_state,
             next_low_seq_hidden_state) = self.rl.choose_rnn_action(
                obs_list=self['obs_list'],
                pre_option_index=self['pre_option_index'],
                pre_action=self['pre_action'],
                rnn_state=self['seq_hidden_state'],
                low_rnn_state=self['low_seq_hidden_state'],

                disable_sample=disable_sample,
                force_rnd_if_available=force_rnd_if_available
            )

        elif self.seq_encoder == SEQ_ENCODER.RNN and self.use_dilation:
            (option_index,
             action,
             prob,
             next_seq_hidden_state,
             next_low_seq_hidden_state) = self.rl.choose_rnn_action(
                obs_list=self['obs_list'],
                pre_option_index=self['pre_option_index'],
                pre_action=self['pre_action'],
                rnn_state=self['key_seq_hidden_state'],  # The previous key rnn_state
                low_rnn_state=self['low_seq_hidden_state'],

                disable_sample=disable_sample,
                force_rnd_if_available=force_rnd_if_available
            )

        elif self.seq_encoder == SEQ_ENCODER.ATTN and not self.use_dilation:
            ep_length = min(512, max([a.episode_length for a in self.agents]))

            all_episode_trans = [a.get_episode_trans(ep_length).values() for a in self.agents]
            (all_ep_indexes,
             all_ep_obses_list,
             all_option_indexes,
             all_option_changed_indexes,
             all_ep_actions,
             all_all_ep_rewards,
             all_next_obs_list,
             all_ep_dones,
             all_ep_probs,
             all_ep_attn_states,
             all_ep_low_rnn_states) = zip(*all_episode_trans)

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

            (option_index,
             action,
             prob,
             next_seq_hidden_state,
             next_low_seq_hidden_state) = self.rl.choose_attn_action(
                ep_indexes=ep_indexes,
                ep_padding_masks=ep_padding_masks,
                ep_obses_list=ep_obses_list,
                ep_pre_actions=ep_pre_actions,
                ep_attn_states=ep_attn_states,

                pre_option_index=self['pre_option_index'],
                low_rnn_state=self['low_seq_hidden_state'],

                disable_sample=disable_sample,
                force_rnd_if_available=force_rnd_if_available
            )

        elif self.seq_encoder == SEQ_ENCODER.ATTN and self.use_dilation:
            key_trans_length = max([a.key_trans_length for a in self.agents])

            all_key_trans = [a.get_key_trans(key_trans_length) for a in self.agents]
            (all_key_indexes,
             all_key_padding_masks,
             all_key_obses_list,
             all_key_attn_states) = zip(*all_key_trans)

            key_indexes = np.concatenate(all_key_indexes)
            key_padding_masks = np.concatenate(all_key_padding_masks)
            key_obses_list = [np.concatenate(o) for o in zip(*all_key_obses_list)]
            key_attn_states = np.concatenate(all_key_attn_states)

            all_last_indexes = [a.get_last_index() for a in self.agents]
            last_indexes = np.concatenate(all_last_indexes)

            if key_indexes.shape[1] == 0:
                key_indexes = np.zeros((key_indexes.shape[0], 1), dtype=key_indexes.dtype)
            else:
                key_indexes = np.concatenate([key_indexes, last_indexes + 1], axis=1)
            key_padding_masks = np.concatenate([key_padding_masks,
                                                np.zeros((key_padding_masks.shape[0], 1), dtype=bool)], axis=1)
            key_obses_list = [np.concatenate([o, np.expand_dims(t_o, 1)], axis=1)
                              for o, t_o in zip(key_obses_list, self['obs_list'])]
            key_attn_states = np.concatenate([key_attn_states,
                                              np.expand_dims(self['key_seq_hidden_state'], 1)], axis=1)

            (option_index,
             action,
             prob,
             next_seq_hidden_state,
             next_low_seq_hidden_state) = self.rl.choose_dilated_attn_action(
                key_indexes=key_indexes,
                key_padding_masks=key_padding_masks,
                key_obses_list=key_obses_list,
                key_attn_states=key_attn_states,

                pre_option_index=self['pre_option_index'],
                pre_action=self['pre_action'],
                low_rnn_state=self['low_seq_hidden_state'],

                disable_sample=disable_sample,
                force_rnd_if_available=force_rnd_if_available
            )

        else:
            option_index, action, prob = self.rl.choose_action(
                obs_list=self['obs_list'],
                pre_option_index=self['pre_option_index']
            )
            next_seq_hidden_state = None
            next_low_seq_hidden_state = None

        if self.use_dilation:
            key_mask = self['option_index'] != option_index
            self['key_seq_hidden_state'][key_mask] = next_seq_hidden_state[key_mask]

        self['option_index'] = option_index
        self['action'] = action
        self['d_action'] = action[..., :self.d_action_summed_size]
        self['c_action'] = action[..., -self.c_action_size:]
        self['prob'] = prob
        self['next_seq_hidden_state'] = next_seq_hidden_state
        self['next_low_seq_hidden_state'] = next_low_seq_hidden_state

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
                    option_index=self['option_index'][i],
                    action=self['action'][i],
                    reward=reward[i],
                    local_done=local_done[i],
                    max_reached=max_reached[i],
                    next_obs_list=[o[i] for o in next_obs_list],
                    prob=self['prob'][i],
                    seq_hidden_state=self['seq_hidden_state'][i] if self.seq_encoder is not None else None,
                    low_seq_hidden_state=self['low_seq_hidden_state'][i] if self.seq_encoder is not None else None,
                )
                if episode_trans is not None:
                    episode_trans_list.append(episode_trans)

        self['episode_trans_list'] = episode_trans_list

    def post_step(self, next_obs_list, local_done, next_padding_mask) -> None:
        super().post_step(next_obs_list, local_done, next_padding_mask)

        self['pre_option_index'] = self['option_index']
        if self.seq_encoder is not None:
            self['low_seq_hidden_state'] = self['next_low_seq_hidden_state']


class OC_MultiAgentsManager(MultiAgentsManager):
    def get_option(self):
        return {n: mgr['option_index'] for n, mgr in self}


agent.Agent = OC_Agent
agent.AgentManager = OC_AgentManager
