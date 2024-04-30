from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import numpy as np

from algorithm.oc.option_selector_base import OptionSelectorBase
from algorithm.utils.enums import *
from algorithm.utils.operators import gen_pre_n_actions

from .. import agent
from ..agent import AGENT_MAX_LIVENESS, Agent, AgentManager, MultiAgentsManager


class OC_Agent(Agent):
    # tmp data, waiting for reward and done to `end_transition`
    _tmp_option_changed_index: int = -1  # The latest option start index
    _tmp_option_index: int = -1  # The latest option index
    _tmp_key_seq_hidden_state: Optional[np.ndarray] = None
    _tmp_low_seq_hidden_state: Optional[np.ndarray] = None

    def __init__(self,
                 agent_id: int,
                 obs_shapes: List[Tuple],
                 d_action_sizes: List[int],
                 c_action_size: int,
                 seq_hidden_state_shape: Optional[Tuple[int, ...]] = None,
                 low_seq_hidden_state_shape: Optional[Tuple[int, ...]] = None,
                 max_return_episode_trans=-1):

        self.low_seq_hidden_state_shape = low_seq_hidden_state_shape

        if self.low_seq_hidden_state_shape is not None:
            self._padding_low_seq_hidden_state = np.zeros(self.low_seq_hidden_state_shape, dtype=np.float32)

        self._tmp_option_changed_indexes = []

        super().__init__(agent_id,
                         obs_shapes,
                         d_action_sizes,
                         c_action_size,
                         seq_hidden_state_shape,
                         max_return_episode_trans)

    def _generate_empty_episode_trans(self, episode_length: int = 0) -> Dict[str, np.ndarray | List[np.ndarray]]:
        empty_episode_trans = super()._generate_empty_episode_trans(episode_length)
        empty_episode_trans['option_index'] = np.full((episode_length, ), -1, dtype=np.int8)
        empty_episode_trans['option_changed_index'] = np.full((episode_length, ), -1, dtype=np.int32)
        empty_episode_trans['low_seq_hidden_state'] = np.zeros((episode_length, *self.low_seq_hidden_state_shape),
                                                               dtype=np.float32) if self.low_seq_hidden_state_shape is not None else None

        return empty_episode_trans

    def set_tmp_obs_action(self,
                           obs_list: List[np.ndarray],
                           option_index: int,
                           action: np.ndarray,
                           prob: np.ndarray,
                           seq_hidden_state: Optional[np.ndarray] = None,
                           low_seq_hidden_state: Optional[np.ndarray] = None):

        super().set_tmp_obs_action(obs_list, action, prob, seq_hidden_state)

        if option_index != self._tmp_option_index:
            option_changed_index = self._tmp_index
            self._tmp_option_changed_indexes.append(option_changed_index)
            self._tmp_option_changed_index = option_changed_index
            self._tmp_key_seq_hidden_state = seq_hidden_state

        self._tmp_option_index = option_index
        self._tmp_low_seq_hidden_state = low_seq_hidden_state

    def get_tmp_option_index(self) -> int:
        return self._tmp_option_index

    def get_key_seq_hidden_state(self) -> Optional[np.ndarray]:
        if self.seq_hidden_state_shape is None:
            return None
        return self._tmp_key_seq_hidden_state if self._tmp_key_seq_hidden_state is not None else self._padding_seq_hidden_state

    def get_low_seq_hidden_state(self) -> Optional[np.ndarray]:
        if self.low_seq_hidden_state_shape is None:
            return None
        return self._tmp_low_seq_hidden_state if self._tmp_low_seq_hidden_state is not None else self._padding_low_seq_hidden_state

    def end_transition(self,
                       reward: float,
                       done: bool = False,
                       max_reached: bool = False,
                       next_obs_list: Optional[List[np.ndarray]] = None) -> Optional[Dict[str, np.ndarray | List[np.ndarray]]]:
        if self._tmp_obs_list is None:
            return

        self._add_transition(
            index=self._tmp_index,
            obs_list=self._tmp_obs_list,
            option_index=self._tmp_option_index,
            option_changed_index=self._tmp_option_changed_index,
            action=self._tmp_action,
            reward=reward,
            done=done,
            max_reached=max_reached,
            prob=self._tmp_prob,
            seq_hidden_state=self._tmp_seq_hidden_state,
            low_seq_hidden_state=self._tmp_low_seq_hidden_state
        )

        self.current_reward += reward

        if not self.done:
            self.steps += 1
            self.reward += reward

        if done:
            if not self.done:
                self.done = True
                self.max_reached = max_reached

            return self._end_episode(next_obs_list if next_obs_list is not None else self._padding_obs_list)

    def _end_episode(self,
                     next_obs_list: List[np.ndarray]) \
            -> Optional[Dict[str, np.ndarray | List[np.ndarray]]]:

        self._add_transition(
            index=self._tmp_index + 1,
            obs_list=next_obs_list,
            option_index=-1,
            option_changed_index=-1,
            action=self._padding_action,
            reward=0,
            done=True,
            max_reached=True,
            prob=np.ones_like(self._padding_action),
            seq_hidden_state=self._padding_seq_hidden_state if self.seq_hidden_state_shape is not None else None,
            low_seq_hidden_state=self._padding_low_seq_hidden_state if self.low_seq_hidden_state_shape is not None else None
        )

        episode_trans = self.get_episode_trans()

        self.current_reward = 0
        self.current_step = 0

        self._tmp_index: int = -1
        self._tmp_obs_list = None
        self._tmp_option_index = -1
        self._tmp_option_changed_index = -1
        self._tmp_action = None
        self._tmp_prob = None
        self._tmp_seq_hidden_state = None
        self._tmp_low_seq_hidden_state = None
        self._tmp_episode_trans = self._generate_empty_episode_trans()
        self._tmp_option_changed_indexes = []

        return episode_trans

    def _add_transition(self,
                        index: int,
                        obs_list: List[np.ndarray],
                        option_index: int,
                        option_changed_index: int,
                        action: np.ndarray,
                        reward: float,
                        done: bool,
                        max_reached: bool,
                        prob: np.ndarray,
                        seq_hidden_state: Optional[np.ndarray] = None,
                        low_seq_hidden_state: Optional[np.ndarray] = None):
        """
        Args:
            index: int
            obs_list: List([*obs_shapes_i], ...)
            option_index: int
            option_changed_index: int
            action: [action_size, ]
            reward: float
            done: bool
            max_reached: bool
            prob: [action_size, ]
            seq_hidden_state: [*seq_hidden_state_shape]
            low_seq_hidden_state: [*low_seq_hidden_state_shape]

        Returns:
            ep_indexes (np.int32): [1, episode_len], int
            ep_obses_list: List([1, episode_len, *obs_shapes_i], ...)
            ep_option_indexes (np.int8): [1, episode_len]
            ep_actions: [1, episode_len, action_size]
            ep_rewards: [1, episode_len]
            ep_dones (np.bool): [1, episode_len], bool
            ep_probs: [1, episode_len, action_size]
            ep_seq_hidden_states: [1, episode_len, *seq_hidden_state_shape]
            ep_low_seq_hidden_states: [1, episode_len, *low_seq_hidden_state_shape]
        """
        expaned_transition = {
            'index': np.expand_dims(index, 0).astype(np.int32),
            'obs_list': [np.expand_dims(o, 0).astype(np.float32) for o in obs_list],
            'option_index': np.expand_dims(option_index, 0).astype(np.int8),
            'option_changed_index': np.expand_dims(option_changed_index, 0).astype(np.int32),
            'action': np.expand_dims(action, 0).astype(np.float32),
            'reward': np.expand_dims(reward, 0).astype(np.float32),
            'done': np.expand_dims(done, 0).astype(bool),
            'max_reached': np.expand_dims(max_reached, 0).astype(bool),
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
            elif self._tmp_episode_trans[k] is not None:
                self._tmp_episode_trans[k] = np.concatenate([self._tmp_episode_trans[k],
                                                             expaned_transition[k]])

        self._extra_log(obs_list,
                        action,
                        reward,
                        done,
                        max_reached,
                        prob)

    def get_episode_trans(self,
                          force_length: int = None) -> Optional[Dict[str, np.ndarray | List[np.ndarray]]]:
        """
        Returns:
            ep_indexes (np.int32): [1, episode_len]
            ep_obses_list: List([1, episode_len, *obs_shapes_i], ...)
            ep_option_indexes (np.int8): [1, episode_len]
            ep_option_changed_indexes (np.int32): [1, episode_len]
            ep_actions: [1, episode_len, action_size]
            ep_rewards: [1, episode_len]
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
                    elif tmp[k] is not None:
                        tmp[k] = tmp[k][-force_length:]
            else:
                tmp_empty = self._generate_empty_episode_trans(delta)
                for k in tmp:
                    if k == 'obs_list':
                        tmp[k] = [np.concatenate([t_o, o]) for t_o, o in zip(tmp_empty[k], tmp[k])]
                    elif tmp[k] is not None:
                        tmp[k] = np.concatenate([tmp_empty[k], tmp[k]])
        else:
            if tmp['index'].shape[0] <= 1:
                return None

        ep_indexes = np.expand_dims(tmp['index'], 0)
        # [1, episode_len]
        ep_obses_list = [np.expand_dims(o, 0) for o in tmp['obs_list']]
        # List([1, episode_len, *obs_shape_si], ...)
        ep_option_indexes = np.expand_dims(tmp['option_index'], 0)  # [1, episode_len]
        ep_option_changed_indexes = np.expand_dims(tmp['option_changed_index'], 0)  # [1, episode_len]
        ep_actions = np.expand_dims(tmp['action'], 0)  # [1, episode_len, action_size]
        ep_rewards = np.expand_dims(tmp['reward'], 0)  # [1, episode_len]
        ep_dones = np.expand_dims(np.logical_and(tmp['done'],
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
            'ep_dones': ep_dones,
            'ep_probs': ep_probs,
            'ep_seq_hidden_states': ep_seq_hidden_states,
            'ep_low_seq_hidden_states': ep_low_seq_hidden_states
        }

    @property
    def key_trans_length(self):
        return len(self._tmp_option_changed_indexes)

    def get_key_trans(self,
                      force_length: int) -> Dict[str, np.ndarray | List[np.ndarray]]:
        """
        Returns:
            key_indexes (np.int32): [1, key_len]
            key_padding_masks (np.bool): [1, key_len]
            key_obses_list: List([1, key_len, *obs_shapes_i], ...)
            key_option_indexes (np.int32): [1, key_len]
            key_seq_hidden_states: [1, key_len, *seq_hidden_state_shape]
        """
        tmp = self._tmp_episode_trans
        option_changed_indexes = self._tmp_option_changed_indexes

        ep_indexes = np.expand_dims(tmp['index'], 0)
        # [1, episode_len]
        ep_obses_list = [np.expand_dims(o, 0) for o in tmp['obs_list']]
        # List([1, episode_len, *obs_shape_si], ...)
        ep_option_indexes = np.expand_dims(tmp['option_index'], 0)
        # [1, episode_len]
        ep_seq_hidden_states = np.expand_dims(tmp['seq_hidden_state'], 0)
        # [1, episode_len, *seq_hidden_state_shape]

        key_indexes = ep_indexes[:, option_changed_indexes]
        key_padding_masks = np.zeros_like(key_indexes, dtype=bool)
        key_obses_list = [o[:, option_changed_indexes] for o in ep_obses_list]
        key_option_indexes = ep_option_indexes[:, option_changed_indexes]
        key_seq_hidden_states = ep_seq_hidden_states[:, option_changed_indexes]

        assert force_length >= self.key_trans_length

        delta = force_length - self.key_trans_length
        if delta > 0:
            delta_key_indexes = -np.ones((1, delta), dtype=np.int32)
            delta_padding_masks = np.ones((1, delta), dtype=bool)  # `True` indicates ignored
            delta_obses_list = [np.zeros((1, delta, *s), dtype=np.float32) for s in self.obs_shapes]
            delta_option_indexes = -np.ones((1, delta), dtype=np.int32)
            delta_seq_hidden_states = np.zeros((1, delta, *self.seq_hidden_state_shape), dtype=np.float32)

            key_indexes = np.concatenate([delta_key_indexes, key_indexes], axis=1)
            key_padding_masks = np.concatenate([delta_padding_masks, key_padding_masks], axis=1)
            key_obses_list = [np.concatenate([d_o, o], axis=1) for d_o, o in zip(delta_obses_list, key_obses_list)]
            key_option_indexes = np.concatenate([delta_option_indexes, key_option_indexes], axis=1)
            key_seq_hidden_states = np.concatenate([delta_seq_hidden_states, key_seq_hidden_states], axis=1)

        return (
            key_indexes,
            key_padding_masks,
            key_obses_list,
            key_option_indexes,
            key_seq_hidden_states,
        )

    def reset(self) -> None:
        """
        The agent may continue in a new iteration but save its last status
        """
        self._tmp_option_index = -1
        self._tmp_option_changed_index = -1
        self._tmp_key_seq_hidden_state = None
        self._tmp_low_seq_hidden_state = None
        self._tmp_option_changed_indexes = []
        return super().reset()


class OC_AgentManager(AgentManager):
    def __init__(self,
                 name: str,
                 obs_names: List[str],
                 obs_shapes: List[Tuple[int]],
                 d_action_sizes: List[int],
                 c_action_size: int):
        super().__init__(name, obs_names, obs_shapes, d_action_sizes, c_action_size)
        self.agents_dict: Dict[int, OC_Agent] = {}
        self.rl: Optional[OptionSelectorBase] = None

    def set_rl(self, rl: OptionSelectorBase) -> None:
        super().set_rl(rl)

        self.option_seq_encoder = rl.option_seq_encoder
        self.use_dilation = rl.use_dilation

    def _verify_agents(self,
                       agent_ids: np.ndarray):

        for agent_id in self.agents_liveness:
            self.agents_liveness[agent_id] -= 1

        for agent_id in agent_ids:
            if agent_id not in self.agents_dict:
                self.agents_dict[agent_id] = OC_Agent(
                    agent_id,
                    self.obs_shapes,
                    self.d_action_sizes,
                    self.c_action_size,
                    seq_hidden_state_shape=self.rl.seq_hidden_state_shape
                    if self.seq_encoder is not None else None,
                    low_seq_hidden_state_shape=self.rl.low_seq_hidden_state_shape
                    if self.option_seq_encoder is not None else None
                )
            self.agents_liveness[agent_id] = AGENT_MAX_LIVENESS

        # Some agents may disabled unexpectively
        # Some agents in Unity may disabled and enabled again in a new episode,
        #   but are assigned new agent ids
        # Set done to these zombie agents
        for agent_id in self.agents_liveness:
            agent = self.agents_dict[agent_id]
            if self.agents_liveness[agent_id] <= 0 and not agent.done:
                agent.force_done()

    def _get_merged_option_index(self, agent_ids: np.ndarray) -> np.ndarray:
        return np.stack([self.agents_dict[_id].get_tmp_option_index() for _id in agent_ids])

    def get_merged_option_index(self) -> np.ndarray:
        return self._get_merged_option_index(self.agents_dict.keys())

    def _get_merged_low_seq_hidden_state(self, agent_ids: np.ndarray) -> np.ndarray:
        return np.stack([self.agents_dict[_id].get_low_seq_hidden_state() for _id in agent_ids])

    def _get_merged_key_seq_hidden_state(self, agent_ids: np.ndarray) -> np.ndarray:
        return np.stack([self.agents_dict[_id].get_key_seq_hidden_state() for _id in agent_ids])

    def get_action(self,
                   agent_ids: np.ndarray,
                   obs_list: List[np.ndarray],
                   last_reward: np.ndarray,
                   disable_sample: bool = False,
                   force_rnd_if_available: bool = False) -> None:
        assert len(agent_ids) == obs_list[0].shape[0]

        if self.rl is None:
            return self.get_test_action(
                agent_ids=agent_ids,
                obs_list=obs_list,
                last_reward=last_reward
            )

        self._verify_agents(agent_ids)

        for i, agent_id in enumerate(agent_ids):
            agent = self.agents_dict[agent_id]
            agent.end_transition(
                reward=last_reward[i],
                done=False,
                max_reached=False
            )

        pre_option_index = self._get_merged_option_index(agent_ids)
        if self.seq_encoder == SEQ_ENCODER.RNN and not self.use_dilation:
            pre_action = self._get_merged_action(agent_ids)
            seq_hidden_state = self._get_merged_seq_hidden_state(agent_ids)
            low_seq_hidden_state = self._get_merged_low_seq_hidden_state(agent_ids) \
                if self.option_seq_encoder is not None else None

            (option_index,
             action,
             prob,
             next_seq_hidden_state,
             next_low_seq_hidden_state) = self.rl.choose_rnn_action(
                obs_list=obs_list,
                pre_option_index=pre_option_index,
                pre_action=pre_action,
                rnn_state=seq_hidden_state,
                low_rnn_state=low_seq_hidden_state,

                disable_sample=disable_sample,
                force_rnd_if_available=force_rnd_if_available
            )

        elif self.seq_encoder == SEQ_ENCODER.RNN and self.use_dilation:
            pre_action = self._get_merged_action(agent_ids)
            seq_hidden_state = self._get_merged_seq_hidden_state(agent_ids)
            low_seq_hidden_state = self._get_merged_low_seq_hidden_state(agent_ids) \
                if self.option_seq_encoder is not None else None
            key_seq_hidden_state = self._get_merged_key_seq_hidden_state(agent_ids)

            (option_index,
             action,
             prob,
             next_seq_hidden_state,
             next_low_seq_hidden_state) = self.rl.choose_rnn_action(
                obs_list=obs_list,
                pre_option_index=pre_option_index,
                pre_action=pre_action,
                rnn_state=key_seq_hidden_state,  # The previous key rnn_state
                low_rnn_state=low_seq_hidden_state,

                disable_sample=disable_sample,
                force_rnd_if_available=force_rnd_if_available
            )

        elif self.seq_encoder == SEQ_ENCODER.ATTN and not self.use_dilation:
            low_seq_hidden_state = self._get_merged_low_seq_hidden_state(agent_ids) \
                if self.option_seq_encoder is not None else None

            ep_length = min(512, max([self.agents_dict[agent_id].episode_length for agent_id in agent_ids]))

            all_episode_trans = [self.agents_dict[agent_id].get_episode_trans(ep_length).values() for agent_id in agent_ids]
            (all_ep_indexes,
             all_ep_obses_list,
             all_option_indexes,
             all_option_changed_indexes,
             all_ep_actions,
             all_all_ep_rewards,
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
                             for o, t_o in zip(ep_obses_list, obs_list)]
            ep_pre_actions = gen_pre_n_actions(ep_actions, True)

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

                pre_option_index=pre_option_index,
                low_rnn_state=low_seq_hidden_state,

                disable_sample=disable_sample,
                force_rnd_if_available=force_rnd_if_available
            )

        elif self.seq_encoder == SEQ_ENCODER.ATTN and self.use_dilation:
            index = self._get_merged_index(agent_ids)
            pre_action = self._get_merged_action(agent_ids)
            low_seq_hidden_state = self._get_merged_low_seq_hidden_state(agent_ids) \
                if self.option_seq_encoder is not None else None
            key_seq_hidden_state = self._get_merged_key_seq_hidden_state(agent_ids)

            key_trans_length = max([self.agents_dict[agent_id].key_trans_length for agent_id in agent_ids])

            all_key_trans = [self.agents_dict[agent_id].get_key_trans(key_trans_length) for agent_id in agent_ids]
            (all_key_indexes,
             all_key_padding_masks,
             all_key_obses_list,
             all_key_option_indexes,
             all_key_attn_states) = zip(*all_key_trans)

            key_indexes = np.concatenate(all_key_indexes)
            key_padding_masks = np.concatenate(all_key_padding_masks)
            key_obses_list = [np.concatenate(o) for o in zip(*all_key_obses_list)]
            key_option_indexes = np.concatenate(all_key_option_indexes)
            key_attn_states = np.concatenate(all_key_attn_states)

            if key_indexes.shape[1] == 0:
                key_indexes = np.zeros((key_indexes.shape[0], 1), dtype=key_indexes.dtype)
            else:
                key_indexes = np.concatenate([key_indexes, np.expand_dims(index + 1, 1)], axis=1)
            key_padding_masks = np.concatenate([key_padding_masks,
                                                np.zeros((key_padding_masks.shape[0], 1), dtype=bool)], axis=1)
            key_obses_list = [np.concatenate([o, np.expand_dims(t_o, 1)], axis=1)
                              for o, t_o in zip(key_obses_list, obs_list)]
            key_option_indexes = np.concatenate([key_option_indexes,
                                                 -np.ones((key_padding_masks.shape[0], 1), dtype=np.int32)], axis=1)
            key_attn_states = np.concatenate([key_attn_states,
                                              np.expand_dims(key_seq_hidden_state, 1)], axis=1)

            (option_index,
             action,
             prob,
             next_seq_hidden_state,
             next_low_seq_hidden_state) = self.rl.choose_dilated_attn_action(
                key_indexes=key_indexes,
                key_padding_masks=key_padding_masks,
                key_obses_list=key_obses_list,
                key_option_indexes=key_option_indexes,
                key_attn_states=key_attn_states,

                pre_option_index=pre_option_index,
                pre_action=pre_action,
                low_rnn_state=low_seq_hidden_state,

                disable_sample=disable_sample,
                force_rnd_if_available=force_rnd_if_available
            )

        else:
            option_index, action, prob = self.rl.choose_action(obs_list,
                                                               pre_option_index,
                                                               disable_sample=disable_sample,
                                                               force_rnd_if_available=force_rnd_if_available)
            next_seq_hidden_state = None
            next_low_seq_hidden_state = None

        for i, agent_id in enumerate(agent_ids):
            agent = self.agents_dict[agent_id]
            agent.set_tmp_obs_action(
                obs_list=[o[i] for o in obs_list],
                option_index=option_index[i],
                action=action[i],
                prob=prob[i],
                seq_hidden_state=next_seq_hidden_state[i] if self.seq_encoder is not None else None,
                low_seq_hidden_state=next_low_seq_hidden_state[i] if self.option_seq_encoder is not None else None
            )

        return action[..., :self.d_action_summed_size], action[..., self.d_action_summed_size:]


class OC_MultiAgentsManager(MultiAgentsManager):
    _ma_manager: Dict[str, OC_AgentManager] = {}

    def __init__(self,
                 ma_obs_names: Dict[str, List[str]],
                 ma_obs_shapes: Dict[str, List[Tuple[int, ...]]],
                 ma_d_action_sizes: Dict[str, List[int]],
                 ma_c_action_size: Dict[str, int],
                 inference_ma_names: Set[str],
                 model_abs_dir: Path):

        agent.Agent = OC_Agent
        agent.AgentManager = OC_AgentManager

        super().__init__(ma_obs_names, ma_obs_shapes, ma_d_action_sizes, ma_c_action_size, inference_ma_names, model_abs_dir)

    def __iter__(self) -> Iterator[Tuple[str, OC_AgentManager]]:
        return iter(self._ma_manager.items())

    def get_option(self):
        return {n: mgr.get_merged_option_index() for n, mgr in self}
