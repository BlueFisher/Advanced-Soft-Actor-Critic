import logging
from pathlib import Path
from typing import Iterator

import numpy as np

from algorithm.oc.option_selector_base import OptionSelectorBase
from algorithm.utils.elapse_timer import unified_elapsed_timer
from algorithm.utils.enums import *
from algorithm.utils.operators import gen_pre_n_actions
from algorithm.utils.visualization.option import OptionVisual

from .. import agent
from ..agent import AGENT_MAX_LIVENESS, Agent, AgentManager, MultiAgentsManager


class OC_Agent(Agent):
    # tmp data, waiting for reward and done to `end_transition`
    _tmp_option_changed_index: int = -1  # The latest option start index
    _tmp_option_index: int = -1  # The latest option index
    _tmp_pre_low_seq_hidden_state: np.ndarray | None = None
    _tmp_low_seq_hidden_state: np.ndarray | None = None
    _tmp_termination: float = 1.
    _tmp_key_seq_hidden_state: np.ndarray | None = None

    def __init__(self,
                 agent_id: int,
                 obs_shapes: list[tuple],
                 d_action_sizes: list[int],
                 c_action_size: int,
                 option_names: list[str],
                 seq_hidden_state_shape: tuple[int, ...],
                 low_seq_hidden_state_shape: tuple[int, ...],
                 max_episode_length=-1,
                 hit_reward: int | None = None):

        self.option_names = option_names

        self.low_seq_hidden_state_shape = low_seq_hidden_state_shape

        self._padding_low_seq_hidden_state = np.zeros(self.low_seq_hidden_state_shape, dtype=np.float32)

        self._tmp_option_changed_indexes = []

        super().__init__(agent_id,
                         obs_shapes,
                         d_action_sizes,
                         c_action_size,
                         seq_hidden_state_shape,
                         max_episode_length,
                         hit_reward)

        self._option_visual = OptionVisual(option_names)

    def _generate_empty_episode_trans(self, episode_length: int = 0) -> dict[str, np.ndarray | list[np.ndarray]]:
        empty_episode_trans = super()._generate_empty_episode_trans(episode_length)
        empty_episode_trans['option_index'] = np.full((episode_length, ), -1, dtype=np.int8)
        empty_episode_trans['option_changed_index'] = np.full((episode_length, ), -1, dtype=np.int32)
        empty_episode_trans['pre_low_seq_hidden_state'] = np.zeros((episode_length, *self.low_seq_hidden_state_shape),
                                                                   dtype=np.float32)
        empty_episode_trans['termination'] = np.zeros((episode_length, ), dtype=np.float32)

        return empty_episode_trans

    def set_tmp_obs_action(self,
                           obs_list: list[np.ndarray],
                           option_index: int,
                           action: np.ndarray,
                           prob: np.ndarray,
                           seq_hidden_state: np.ndarray,
                           low_seq_hidden_state: np.ndarray,
                           termination: float):

        super().set_tmp_obs_action(obs_list, action, prob, seq_hidden_state)

        if option_index != self._tmp_option_index:
            option_changed_index = self._tmp_index
            self._tmp_option_changed_indexes.append(option_changed_index)
            self._tmp_option_changed_index = option_changed_index
            self._tmp_key_seq_hidden_state = seq_hidden_state

        self._tmp_option_index = option_index
        self._tmp_pre_low_seq_hidden_state = self._tmp_low_seq_hidden_state
        self._tmp_low_seq_hidden_state = low_seq_hidden_state
        self._tmp_termination = termination

        self._option_visual.add_option_termination(option_index, termination)

    def get_tmp_option_index(self) -> int:
        return self._tmp_option_index

    def get_termination(self) -> float:
        return self._tmp_termination

    def get_low_seq_hidden_state(self) -> np.ndarray:
        if self._tmp_low_seq_hidden_state is None:
            return self._padding_low_seq_hidden_state
        return self._tmp_low_seq_hidden_state

    def get_key_seq_hidden_state(self) -> np.ndarray:
        if self._tmp_key_seq_hidden_state is None:
            return self._padding_seq_hidden_state
        return self._tmp_key_seq_hidden_state

    def end_transition(self,
                       reward: float,
                       done: bool = False,
                       max_reached: bool = False,
                       force_terminated: bool = False,
                       next_obs_list: list[np.ndarray] | None = None) -> dict[str, np.ndarray | list[np.ndarray]] | None:
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
            pre_seq_hidden_state=self._tmp_pre_seq_hidden_state,
            pre_low_seq_hidden_state=self._tmp_pre_low_seq_hidden_state
        )

        self.current_reward += reward

        if not self.done:
            self.steps += 1
            self.reward += reward
            if self.hit_reward is not None and reward >= self.hit_reward:
                self.hit += 1

        if done:
            # if the episode is done and the agent is empty, reset the agent
            if not self.done and self.is_empty:
                self.steps = 0
                self.reward = 0
                self.hit = 0
                self.current_step = 0
                self.current_reward = 0
                return

            if not self.done:
                self.done = True
                self.max_reached = max_reached
                self.force_terminated = force_terminated

            return self._end_episode(next_obs_list if next_obs_list is not None else self._padding_obs_list)

    def _end_episode(self,
                     next_obs_list: list[np.ndarray]) \
            -> dict[str, np.ndarray | list[np.ndarray]] | None:

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
            pre_seq_hidden_state=self._padding_seq_hidden_state,
            pre_low_seq_hidden_state=self._padding_low_seq_hidden_state,
        )

        if self._option_visual is not None:
            self._option_visual.reset()

        episode_trans = self.get_episode_trans()

        self.current_reward = 0
        self.current_step = 0

        self._tmp_index: int = -1
        self._tmp_obs_list = None
        self._tmp_option_index = -1
        self._tmp_option_changed_index = -1
        self._tmp_action = None
        self._tmp_prob = None
        self._tmp_pre_seq_hidden_state = None
        self._tmp_seq_hidden_state = None
        self._tmp_pre_low_seq_hidden_state = None
        self._tmp_low_seq_hidden_state = None
        self._tmp_termination = 1.
        self._tmp_episode_trans = self._generate_empty_episode_trans(self.max_episode_length)
        self._tmp_option_changed_indexes = []

        return episode_trans

    def _add_transition(self,
                        index: int,
                        obs_list: list[np.ndarray],
                        option_index: int,
                        option_changed_index: int,
                        action: np.ndarray,
                        reward: float,
                        done: bool,
                        max_reached: bool,
                        prob: np.ndarray,
                        pre_seq_hidden_state: np.ndarray | None,
                        pre_low_seq_hidden_state: np.ndarray | None):
        """
        Args:
            index: int
            obs_list: list([*obs_shapes_i], ...)
            option_index: int
            option_changed_index: int
            action: [action_size, ]
            reward: float
            done: bool
            max_reached: bool
            prob: [action_size, ]
            pre_seq_hidden_state: [*seq_hidden_state_shape]
            pre_low_seq_hidden_state: [*low_seq_hidden_state_shape]

        Returns:
            ep_indexes (np.int32): [1, episode_len], int
            ep_obses_list: list([1, episode_len, *obs_shapes_i], ...)
            ep_option_indexes (np.int8): [1, episode_len]
            ep_actions: [1, episode_len, action_size]
            ep_rewards: [1, episode_len]
            ep_dones (np.bool): [1, episode_len], bool
            ep_probs: [1, episode_len, action_size]
            ep_pre_seq_hidden_states: [1, episode_len, *seq_hidden_state_shape]
            ep_pre_low_seq_hidden_states: [1, episode_len, *low_seq_hidden_state_shape]
        """
        if self.current_step == self.max_episode_length:
            self._logger.warning(f'_tmp_episode_trans is full {self.max_episode_length}')
            for k, v in self._tmp_episode_trans.items():
                if k == 'obs_list':
                    for o in self._tmp_episode_trans[k]:
                        o[:-1] = o[1:]
                else:
                    v[:-1] = v[1:]

            self.current_step -= 1

        self._tmp_episode_trans['index'][self.current_step] = index
        for tmp_obs, obs in zip(self._tmp_episode_trans['obs_list'], obs_list):
            tmp_obs[self.current_step] = obs
        self._tmp_episode_trans['option_index'][self.current_step] = option_index
        self._tmp_episode_trans['option_changed_index'][self.current_step] = option_changed_index
        self._tmp_episode_trans['action'][self.current_step] = action
        self._tmp_episode_trans['reward'][self.current_step] = reward
        self._tmp_episode_trans['done'][self.current_step] = done
        self._tmp_episode_trans['max_reached'][self.current_step] = max_reached
        self._tmp_episode_trans['prob'][self.current_step] = prob
        if pre_seq_hidden_state is None:
            pre_seq_hidden_state = self._padding_seq_hidden_state
        self._tmp_episode_trans['pre_seq_hidden_state'][self.current_step] = pre_seq_hidden_state
        if pre_low_seq_hidden_state is None:
            pre_low_seq_hidden_state = self._padding_low_seq_hidden_state
        self._tmp_episode_trans['pre_low_seq_hidden_state'][self.current_step] = pre_low_seq_hidden_state

        self.current_step += 1

    def get_episode_trans(self,
                          force_length: int | None = None) -> dict[str, np.ndarray | list[np.ndarray]] | None:
        """
        Returns:
            ep_indexes (np.int32): [1, episode_len]
            ep_obses_list: list([1, episode_len, *obs_shapes_i], ...)
            ep_option_indexes (np.int8): [1, episode_len]
            ep_option_changed_indexes (np.int32): [1, episode_len]
            ep_actions: [1, episode_len, action_size]
            ep_rewards: [1, episode_len]
            ep_dones (np.bool): [1, episode_len]
            ep_probs: [1, episode_len]
            ep_pre_seq_hidden_states: [1, episode_len, *seq_hidden_state_shape]
            ep_pre_low_seq_hidden_states: [1, episode_len, *low_seq_hidden_state_shape]
            ep_terminations: [1, episode_len]
        """
        tmp = {}
        for k, v in self._tmp_episode_trans.items():
            if k == 'obs_list':
                tmp[k] = [o[:self.episode_length] for o in v]
            else:
                tmp[k] = v[:self.episode_length]

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
            if self.episode_length <= 1:
                return None

        ep_indexes = np.expand_dims(tmp['index'], 0)
        # [1, episode_len]
        ep_obses_list = [np.expand_dims(o, 0) for o in tmp['obs_list']]
        # list([1, episode_len, *obs_shape_si], ...)
        ep_option_indexes = np.expand_dims(tmp['option_index'], 0)  # [1, episode_len]
        ep_option_changed_indexes = np.expand_dims(tmp['option_changed_index'], 0)  # [1, episode_len]
        ep_actions = np.expand_dims(tmp['action'], 0)  # [1, episode_len, action_size]
        ep_rewards = np.expand_dims(tmp['reward'], 0)  # [1, episode_len]
        ep_dones = np.expand_dims(np.logical_and(tmp['done'],
                                                 ~tmp['max_reached']),
                                  0)  # [1, episode_len]
        ep_probs = np.expand_dims(tmp['prob'], 0)  # [1, episode_len]
        ep_pre_seq_hidden_states = np.expand_dims(tmp['pre_seq_hidden_state'], 0)
        # [1, episode_len, *seq_hidden_state_shape]
        ep_pre_low_seq_hidden_states = np.expand_dims(tmp['pre_low_seq_hidden_state'], 0)
        # [1, episode_len, *low_seq_hidden_state_shape]

        return {
            'ep_indexes': ep_indexes,
            'ep_obses_list': ep_obses_list,
            'ep_option_indexes': ep_option_indexes,
            'ep_option_changed_indexes': ep_option_changed_indexes,
            'ep_actions': ep_actions,
            'ep_rewards': ep_rewards,
            'ep_dones': ep_dones,
            'ep_probs': ep_probs,
            'ep_pre_seq_hidden_states': ep_pre_seq_hidden_states,
            'ep_pre_low_seq_hidden_states': ep_pre_low_seq_hidden_states,
        }

    @property
    def key_trans_length(self):
        return len(self._tmp_option_changed_indexes)

    def get_key_trans(self,
                      force_length: int) -> dict[str, np.ndarray | list[np.ndarray]]:
        """
        Returns:
            key_indexes (np.int32): [1, key_len]
            key_padding_masks (np.bool): [1, key_len]
            key_obses_list: list([1, key_len, *obs_shapes_i], ...)
            key_option_indexes (np.int32): [1, key_len]
            key_pre_seq_hidden_states: [1, key_len, *seq_hidden_state_shape]
        """
        tmp = self._tmp_episode_trans
        option_changed_indexes = self._tmp_option_changed_indexes

        ep_indexes = np.expand_dims(tmp['index'], 0)
        # [1, episode_len]
        ep_obses_list = [np.expand_dims(o, 0) for o in tmp['obs_list']]
        # list([1, episode_len, *obs_shape_si], ...)
        ep_option_indexes = np.expand_dims(tmp['option_index'], 0)
        # [1, episode_len]
        ep_pre_seq_hidden_states = np.expand_dims(tmp['pre_seq_hidden_state'], 0)
        # [1, episode_len, *seq_hidden_state_shape]

        key_indexes = ep_indexes[:, option_changed_indexes]
        key_padding_masks = np.zeros_like(key_indexes, dtype=bool)
        key_obses_list = [o[:, option_changed_indexes] for o in ep_obses_list]
        key_option_indexes = ep_option_indexes[:, option_changed_indexes]
        key_pre_seq_hidden_states = ep_pre_seq_hidden_states[:, option_changed_indexes]

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
            key_pre_seq_hidden_states = np.concatenate([delta_seq_hidden_states, key_pre_seq_hidden_states], axis=1)

        return (
            key_indexes,
            key_padding_masks,
            key_obses_list,
            key_option_indexes,
            key_pre_seq_hidden_states,
        )

    def reset(self) -> None:
        """
        The agent may continue in a new iteration but save its last status
        """
        self._tmp_option_index = -1
        self._tmp_option_changed_index = -1
        self._tmp_option_changed_indexes = []
        super().reset()


class OC_AgentManager(AgentManager):
    def __init__(self,
                 name: str,
                 obs_names: list[str],
                 obs_shapes: list[tuple[int]],
                 d_action_sizes: list[int],
                 c_action_size: int,
                 max_episode_length: int = -1,
                 hit_reward: int | None = None):
        super().__init__(name, obs_names, obs_shapes, d_action_sizes, c_action_size,
                         max_episode_length, hit_reward)

        self.agents_dict: dict[int, OC_Agent] = {}
        self.rl: OptionSelectorBase | None = None

    def set_rl(self, rl: OptionSelectorBase) -> None:
        super().set_rl(rl)

        self.option_seq_encoder = rl.option_seq_encoder
        self.use_dilation = rl.use_dilation
        self.option_names = rl.get_option_names()

    def _verify_agents(self,
                       agent_ids: np.ndarray):

        assert self.rl is not None

        for agent_id in self.agents_liveness:
            self.agents_liveness[agent_id] -= 1

        for agent_id in agent_ids:
            if agent_id not in self.agents_dict:
                self.agents_dict[agent_id] = OC_Agent(
                    agent_id,
                    self.obs_shapes,
                    self.d_action_sizes,
                    self.c_action_size,
                    self.option_names,
                    seq_hidden_state_shape=self.rl.seq_hidden_state_shape,
                    low_seq_hidden_state_shape=self.rl.low_seq_hidden_state_shape,
                    max_episode_length=self.max_episode_length,
                    hit_reward=self.hit_reward
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

    def _get_merged_pre_termination(self, agent_ids: np.ndarray) -> np.ndarray:
        return np.stack([self.agents_dict[_id].get_termination() for _id in agent_ids], dtype=np.float32)

    def _get_merged_key_seq_hidden_state(self, agent_ids: np.ndarray) -> np.ndarray:
        return np.stack([self.agents_dict[_id].get_key_seq_hidden_state() for _id in agent_ids])

    @unified_elapsed_timer('get_action', repeat=10)
    def get_action(self,
                   agent_ids: np.ndarray,
                   obs_list: list[np.ndarray],
                   last_reward: np.ndarray,
                   disable_sample: bool = False,
                   force_rnd_if_available: bool = False) -> None:
        assert len(agent_ids) == obs_list[0].shape[0]

        if self.rl is None:
            return self.get_test_action(
                agent_ids=agent_ids,
                obs_list=obs_list
            )

        self._verify_agents(agent_ids)

        for i, agent_id in enumerate(agent_ids):
            agent = self.agents_dict[agent_id]
            agent.end_transition(
                reward=last_reward[i],
                done=False,
                max_reached=False,
            )

        pre_option_index = self._get_merged_option_index(agent_ids)

        if self.seq_encoder in (None, SEQ_ENCODER.RNN) and not self.use_dilation:
            pre_action = self._get_merged_action(agent_ids)
            pre_seq_hidden_state = self._get_merged_seq_hidden_state(agent_ids)
            pre_low_seq_hidden_state = self._get_merged_low_seq_hidden_state(agent_ids)
            pre_termination = self._get_merged_pre_termination(agent_ids)

            (option_index,
             action,
             prob,
             seq_hidden_state,
             low_seq_hidden_state,
             termination) = self.rl.choose_action(
                obs_list=obs_list,
                pre_option_index=pre_option_index,
                pre_action=pre_action,
                pre_seq_hidden_state=pre_seq_hidden_state,
                pre_low_seq_hidden_state=pre_low_seq_hidden_state,
                pre_termination=pre_termination,

                disable_sample=disable_sample,
                force_rnd_if_available=force_rnd_if_available
            )

        elif self.seq_encoder in (None, SEQ_ENCODER.RNN) and self.use_dilation:
            pre_action = self._get_merged_action(agent_ids)
            # pre_seq_hidden_state = self._get_merged_seq_hidden_state(agent_ids)
            pre_low_seq_hidden_state = self._get_merged_low_seq_hidden_state(agent_ids)
            pre_termination = self._get_merged_pre_termination(agent_ids)

            pre_key_seq_hidden_state = self._get_merged_key_seq_hidden_state(agent_ids)

            (option_index,
             action,
             prob,
             seq_hidden_state,
             low_seq_hidden_state,
             termination) = self.rl.choose_action(
                obs_list=obs_list,
                pre_option_index=pre_option_index,
                pre_action=pre_action,
                pre_seq_hidden_state=pre_key_seq_hidden_state,  # The previous key seq_hidden_state
                pre_low_seq_hidden_state=pre_low_seq_hidden_state,
                pre_termination=pre_termination,

                disable_sample=disable_sample,
                force_rnd_if_available=force_rnd_if_available
            )

        elif self.seq_encoder == SEQ_ENCODER.ATTN and not self.use_dilation:
            pre_low_seq_hidden_state = self._get_merged_low_seq_hidden_state(agent_ids)
            pre_termination = self._get_merged_pre_termination(agent_ids)

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
             all_ep_low_seq_hidden_state) = zip(*all_episode_trans)

            ep_indexes = np.concatenate(all_ep_indexes)
            ep_obses_list = [np.concatenate(o) for o in zip(*all_ep_obses_list)]
            ep_actions = np.concatenate(all_ep_actions)
            ep_pre_attn_states = np.concatenate(all_ep_attn_states)

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
             seq_hidden_state,
             low_seq_hidden_state,
             termination) = self.rl.choose_attn_action(
                ep_indexes=ep_indexes,
                ep_padding_masks=ep_padding_masks,
                ep_obses_list=ep_obses_list,
                ep_pre_actions=ep_pre_actions,
                ep_pre_attn_states=ep_pre_attn_states,

                pre_option_index=pre_option_index,
                pre_low_seq_hidden_state=pre_low_seq_hidden_state,
                pre_termination=pre_termination,

                disable_sample=disable_sample,
                force_rnd_if_available=force_rnd_if_available
            )

        elif self.seq_encoder == SEQ_ENCODER.ATTN and self.use_dilation:
            index = self._get_merged_index(agent_ids)
            pre_action = self._get_merged_action(agent_ids)
            pre_low_seq_hidden_state = self._get_merged_low_seq_hidden_state(agent_ids)
            pre_termination = self._get_merged_pre_termination(agent_ids)

            key_pre_seq_hidden_state = self._get_merged_key_seq_hidden_state(agent_ids)

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
                                              np.expand_dims(key_pre_seq_hidden_state, 1)], axis=1)

            (option_index,
             action,
             prob,
             seq_hidden_state,
             low_seq_hidden_state,
             termination) = self.rl.choose_dilated_attn_action(
                key_indexes=key_indexes,
                key_padding_masks=key_padding_masks,
                key_obses_list=key_obses_list,
                key_option_indexes=key_option_indexes,
                key_attn_states=key_attn_states,

                pre_option_index=pre_option_index,
                pre_action=pre_action,
                pre_low_seq_hidden_state=pre_low_seq_hidden_state,
                pre_termination=pre_termination,

                disable_sample=disable_sample,
                force_rnd_if_available=force_rnd_if_available
            )

        for i, agent_id in enumerate(agent_ids):
            agent = self.agents_dict[agent_id]
            agent.set_tmp_obs_action(
                obs_list=[o[i] for o in obs_list],
                option_index=option_index[i],
                action=action[i],
                prob=prob[i],
                seq_hidden_state=seq_hidden_state[i],
                low_seq_hidden_state=low_seq_hidden_state[i],
                termination=termination[i]
            )

        return action[..., :self.d_action_summed_size], action[..., self.d_action_summed_size:]


class OC_MultiAgentsManager(MultiAgentsManager):
    _ma_manager: dict[str, OC_AgentManager]

    def __init__(self,
                 ma_obs_names: dict[str, list[str]],
                 ma_obs_shapes: dict[str, list[tuple[int, ...]]],
                 ma_d_action_sizes: dict[str, list[int]],
                 ma_c_action_size: dict[str, int],
                 inference_ma_names: set[str],
                 model_abs_dir: Path,
                 max_episode_length: int = -1,
                 hit_reward: int | None = None):

        agent.Agent = OC_Agent
        agent.AgentManager = OC_AgentManager

        super().__init__(ma_obs_names,
                         ma_obs_shapes,
                         ma_d_action_sizes,
                         ma_c_action_size,
                         inference_ma_names,
                         model_abs_dir,
                         max_episode_length=max_episode_length,
                         hit_reward=hit_reward)

    def __iter__(self) -> Iterator[tuple[str, OC_AgentManager]]:
        return iter(self._ma_manager.items())

    def get_option(self):
        return {n: mgr.get_merged_option_index() for n, mgr in self}
