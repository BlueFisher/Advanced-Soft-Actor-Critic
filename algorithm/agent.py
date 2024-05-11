import logging
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import numpy as np

from algorithm.sac_base import SAC_Base
from algorithm.utils.elapse_timer import (UnifiedElapsedTimer,
                                          unified_elapsed_timer)
from algorithm.utils.enums import *
from algorithm.utils.operators import gen_pre_n_actions

AGENT_MAX_LIVENESS = 20
NON_EMPTY_STEPS = 2


class Agent:
    reward = 0  # The reward of the *first* completed episode (done == True)
    steps = 0  # The step count of the *first* completed episode (done == True)
    done = False  # If has one completed episode
    max_reached = False  # If has one completed episode that reaches max step

    current_reward = 0  # The reward of the current episode
    current_step = 0  # The step count of the current episode

    # tmp data, waiting for reward and done to `end_transition`
    _tmp_index: int = -1
    _tmp_obs_list: Optional[List[np.ndarray]] = None
    _tmp_action: Optional[np.ndarray] = None
    _tmp_prob: Optional[np.ndarray] = None
    _tmp_seq_hidden_state: Optional[np.ndarray] = None

    _tmp_episode_trans: Dict[str, np.ndarray | List[np.ndarray]]

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

        self._padding_obs_list = [np.zeros(o).astype(np.float32) for o in self.obs_shapes]
        d_action_list = [np.eye(d_action_size, dtype=np.float32)[0]
                         for d_action_size in self.d_action_sizes]
        self._padding_action = np.concatenate(d_action_list + [np.zeros(self.c_action_size, dtype=np.float32)], axis=-1)
        if self.seq_hidden_state_shape is not None:
            self._padding_seq_hidden_state = np.zeros(self.seq_hidden_state_shape, dtype=np.float32)

        self._tmp_episode_trans = self._generate_empty_episode_trans()

    def _generate_empty_episode_trans(self, episode_length: int = 0) -> Dict[str, np.ndarray | List[np.ndarray]]:
        return {
            'index': -np.ones((episode_length, ), dtype=np.int32),
            'obs_list': [np.zeros((episode_length, *s), dtype=np.float32) for s in self.obs_shapes],
            'action': np.zeros((episode_length, self.d_action_summed_size + self.c_action_size), dtype=np.float32),
            'reward': np.zeros((episode_length, ), dtype=np.float32),
            'done': np.zeros((episode_length, ), dtype=bool),
            'max_reached': np.zeros((episode_length, ), dtype=bool),
            'prob': np.zeros((episode_length, self.d_action_summed_size + self.c_action_size), dtype=np.float32),
            'seq_hidden_state': np.zeros((episode_length, *self.seq_hidden_state_shape), dtype=np.float32) if self.seq_hidden_state_shape is not None else None,
        }

    def set_tmp_obs_action(self,
                           obs_list: List[np.ndarray],
                           action: np.ndarray,
                           prob: np.ndarray,
                           seq_hidden_state: Optional[np.ndarray] = None):
        self._tmp_index += 1
        self._tmp_obs_list = obs_list
        self._tmp_action = action
        self._tmp_prob = prob
        self._tmp_seq_hidden_state = seq_hidden_state

    def get_tmp_index(self) -> int:
        return self._tmp_index

    def get_tmp_obs_list(self) -> List[np.ndarray]:
        return self._tmp_obs_list if self._tmp_obs_list is not None else self._padding_obs_list

    def get_tmp_action(self) -> np.ndarray:
        return self._tmp_action if self._tmp_action is not None else self._padding_action

    def get_tmp_seq_hidden_state(self) -> Optional[np.ndarray]:
        if self.seq_hidden_state_shape is None:
            return None
        return self._tmp_seq_hidden_state if self._tmp_seq_hidden_state is not None else self._padding_seq_hidden_state

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
            action=self._tmp_action,
            reward=reward,
            done=done,
            max_reached=max_reached,
            prob=self._tmp_prob,
            seq_hidden_state=self._tmp_seq_hidden_state
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
            action=self._padding_action,
            reward=0.,
            done=True,
            max_reached=True,
            prob=np.ones_like(self._padding_action),
            seq_hidden_state=self._padding_seq_hidden_state if self.seq_hidden_state_shape is not None else None
        )

        episode_trans = self.get_episode_trans()

        self.current_reward = 0
        self.current_step = 0

        self._tmp_index: int = -1
        self._tmp_obs_list = None
        self._tmp_action = None
        self._tmp_prob = None
        self._tmp_seq_hidden_state = None
        self._tmp_episode_trans = self._generate_empty_episode_trans()

        return episode_trans

    def _add_transition(self,
                        index: int,
                        obs_list: List[np.ndarray],
                        action: np.ndarray,
                        reward: float,
                        done: bool,
                        max_reached: bool,
                        prob: np.ndarray,
                        seq_hidden_state: Optional[np.ndarray] = None) \
            -> Optional[Dict[str, np.ndarray | List[np.ndarray]]]:
        """
        Args:
            index: int
            obs_list: List([*obs_shapes_i], ...)
            action: [action_size, ]
            reward: float
            done: bool
            max_reached: bool
            prob: [action_size, ]
            seq_hidden_state: [*seq_hidden_state_shape]

        Returns:
            ep_indexes (np.int32): [1, episode_len], int
            ep_obses_list: List([1, episode_len, *obs_shapes_i], ...)
            ep_actions: [1, episode_len, action_size]
            ep_rewards: [1, episode_len]
            ep_dones (np.bool): [1, episode_len], bool
            ep_probs: [1, episode_len, action_size]
            ep_seq_hidden_states: [1, episode_len, *seq_hidden_state_shape]
        """
        expaned_transition = {
            'index': np.expand_dims(index, 0).astype(np.int32),
            'obs_list': [np.expand_dims(o, 0).astype(np.float32) for o in obs_list],
            'action': np.expand_dims(action, 0).astype(np.float32),
            'reward': np.expand_dims(reward, 0).astype(np.float32),
            'done': np.expand_dims(done, 0).astype(bool),
            'max_reached': np.expand_dims(max_reached, 0).astype(bool),
            'prob': np.expand_dims(prob, 0).astype(np.float32),
            'seq_hidden_state': np.expand_dims(seq_hidden_state, 0).astype(np.float32) if seq_hidden_state is not None else None,
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

    @property
    def episode_length(self) -> int:
        return len(self._tmp_episode_trans['index'])

    def _extra_log(self,
                   obs_list,
                   action,
                   reward,
                   done,
                   max_reached,
                   prob) -> None:
        pass

    def get_episode_trans(self,
                          force_length: int = None) -> Optional[Dict[str, np.ndarray | List[np.ndarray]]]:
        """
        Returns:
            ep_indexes (np.int32): [1, episode_len]
            ep_obses_list: List([1, episode_len, *obs_shapes_i], ...)
            ep_actions: [1, episode_len, action_size]
            ep_rewards: [1, episode_len]
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

        ep_indexes = np.expand_dims(tmp['index'], 0)  # [1, episode_len]
        ep_obses_list = [np.expand_dims(o, 0) for o in tmp['obs_list']]
        # List([1, episode_len, *obs_shape_si], ...)
        ep_actions = np.expand_dims(tmp['action'], 0)  # [1, episode_len, action_size]
        ep_rewards = np.expand_dims(tmp['reward'], 0)  # [1, episode_len]
        ep_dones = np.expand_dims(np.logical_and(tmp['done'],
                                                 ~tmp['max_reached']),
                                  0)  # [1, episode_len]
        ep_probs = np.expand_dims(tmp['prob'], 0)  # [1, episode_len, action_size]
        ep_seq_hidden_states = np.expand_dims(tmp['seq_hidden_state'], 0) if tmp['seq_hidden_state'] is not None else None
        # [1, episode_len, *seq_hidden_state_shape]

        return {
            'ep_indexes': ep_indexes,
            'ep_obses_list': ep_obses_list,
            'ep_actions': ep_actions,
            'ep_rewards': ep_rewards,
            'ep_dones': ep_dones,
            'ep_probs': ep_probs,
            'ep_seq_hidden_states': ep_seq_hidden_states
        }

    def is_empty(self) -> bool:
        return self.episode_length == 0

    def force_done(self) -> None:
        self.done = True
        self.max_reached = True

    def reset(self) -> None:
        """
        The agent may continue in a new iteration but save its last status
        """
        self.reward = self.current_reward
        self.steps = 0
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

        self.agents_dict: Dict[int, Agent] = {}  # {agent_id: Agent}
        self.agents_liveness: Dict[int, int] = {}  # {agent_id: int}

        self.rl: Optional[SAC_Base] = None
        self.seq_encoder = None

        self._logger = logging.getLogger(f'agent_mgr.{name}')
        self._profiler = UnifiedElapsedTimer(self._logger)

        self._tmp_episode_trans_list = []

        self._data = {}

    def __getitem__(self, k: str):
        return self._data[k]

    def __setitem__(self, k: str, v):
        self._data[k] = v

    @property
    def agents(self) -> List[Agent]:
        return list(self.agents_dict.values())

    @property
    def non_empty_agents(self) -> List[Agent]:
        """
        Get agents which steps > NON_EMPTY_STEPS.
        Agents in Unity enabled after disable may contain useless `next_step`
        """
        return [a for a in self.agents if a.steps > NON_EMPTY_STEPS]

    @property
    def done(self) -> bool:
        return all([a.done for a in self.agents])

    @property
    def max_reached(self) -> bool:
        return any([a.max_reached for a in self.agents])

    def set_config(self, config) -> None:
        self.config = deepcopy(config)

    def set_model_abs_dir(self, model_abs_dir: Path) -> None:
        model_abs_dir.mkdir(parents=True, exist_ok=True)
        self.model_abs_dir = model_abs_dir

    def set_rl(self, rl: SAC_Base) -> None:
        self.rl = rl
        self.seq_encoder = rl.seq_encoder

    def reset(self) -> None:
        """
        Remove all agents
        """
        self.agents_dict.clear()
        self.agents_liveness.clear()
        self.clear_tmp_episode_trans_list()

    def reset_dead_agents(self) -> None:
        """
        Remove agents where liveness <= 0
        """
        dead_agent_ids = [agent_id
                          for agent_id, liveness in self.agents_liveness.items() if liveness <= 0]
        for agent_id in dead_agent_ids:
            del self.agents_dict[agent_id]
            del self.agents_liveness[agent_id]

    def reset_and_continue(self) -> None:
        """
        Agents may continue in a new iteration but save its last status
        """
        self.reset_dead_agents()

        for agent in self.agents:
            agent.reset()

        self.clear_tmp_episode_trans_list()

    def _verify_agents(self,
                       agent_ids: np.ndarray):
        """
        Verify whether id in self.agents_dict and whether agent is active
        """

        for agent_id in self.agents_liveness:
            self.agents_liveness[agent_id] -= 1

        for agent_id in agent_ids:
            if agent_id not in self.agents_dict:
                self.agents_dict[agent_id] = Agent(
                    agent_id,
                    self.obs_shapes,
                    self.d_action_sizes,
                    self.c_action_size,
                    seq_hidden_state_shape=self.rl.seq_hidden_state_shape
                    if self.seq_encoder is not None else None
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

    def _get_merged_index(self, agent_ids: np.ndarray) -> np.ndarray:
        return np.stack([self.agents_dict[_id].get_tmp_index() for _id in agent_ids])

    def _get_merged_obs_list(self, agent_ids: np.ndarray) -> List[np.ndarray]:
        agents_obs_list = [self.agents_dict[_id].get_tmp_obs_list() for _id in agent_ids]
        return [np.stack(o) for o in zip(*agents_obs_list)]

    def _get_merged_action(self, agent_ids: np.ndarray) -> np.ndarray:
        return np.stack([self.agents_dict[_id].get_tmp_action() for _id in agent_ids])

    def _get_merged_seq_hidden_state(self, agent_ids: np.ndarray) -> np.ndarray:
        return np.stack([self.agents_dict[_id].get_tmp_seq_hidden_state() for _id in agent_ids])

    def _merge_agents_data(self,
                           agent_ids: np.ndarray,
                           key: int):
        agents = [self.agents_dict[_id] for _id in agent_ids]
        data_list = [a.get_tmp_data(key) for a in agents]
        if isinstance(data_list[0], list):
            return [np.stack(data, axis=0) for data in zip(*data_list)]
        else:
            return np.stack(data_list, axis=0)

    @unified_elapsed_timer('get_action', repeat=10)
    def get_action(self,
                   agent_ids: np.ndarray,
                   obs_list: List[np.ndarray],
                   last_reward: np.ndarray,
                   disable_sample: bool = False,
                   force_rnd_if_available: bool = False) -> np.ndarray:
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

        if self.seq_encoder == SEQ_ENCODER.RNN:
            pre_action = self._get_merged_action(agent_ids)
            seq_hidden_state = self._get_merged_seq_hidden_state(agent_ids)

            action, prob, next_seq_hidden_state = self.rl.choose_rnn_action(
                obs_list=obs_list,
                pre_action=pre_action,
                rnn_state=seq_hidden_state,

                disable_sample=disable_sample,
                force_rnd_if_available=force_rnd_if_available
            )

        elif self.seq_encoder == SEQ_ENCODER.ATTN:
            ep_length = min(512, max([self.agents_dict[agent_id].episode_length for agent_id in agent_ids]))

            all_episode_trans = [self.agents_dict[agent_id].get_episode_trans(ep_length).values() for agent_id in agent_ids]
            (all_ep_indexes,
             all_ep_obses_list,
             all_ep_actions,
             all_ep_rewards,
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
                             for o, t_o in zip(ep_obses_list, obs_list)]
            ep_pre_actions = gen_pre_n_actions(ep_actions, True)

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
            action, prob = self.rl.choose_action(obs_list,
                                                 disable_sample=disable_sample,
                                                 force_rnd_if_available=force_rnd_if_available)
            next_seq_hidden_state = None

        for i, agent_id in enumerate(agent_ids):
            agent = self.agents_dict[agent_id]
            agent.set_tmp_obs_action(
                obs_list=[o[i] for o in obs_list],
                action=action[i],
                prob=prob[i],
                seq_hidden_state=next_seq_hidden_state[i] if self.seq_encoder is not None else None
            )

        return action[..., :self.d_action_summed_size], action[..., self.d_action_summed_size:]

    def get_test_action(self,
                        agent_ids: np.ndarray,
                        obs_list: List[np.ndarray]) -> np.ndarray:
        assert len(agent_ids) == obs_list[0].shape[0]

        self._verify_agents(agent_ids)

        for i, agent_id in enumerate(agent_ids):
            agent = self.agents_dict[agent_id]
            agent.end_transition(
                reward=0,
                done=False,
                max_reached=False
            )

        n_agents = len(agent_ids)

        action = np.zeros((n_agents, self.action_size), dtype=np.float32)
        prob = np.random.rand(n_agents, self.action_size)

        if self.d_action_sizes:
            d_action_list = [np.random.randint(0, d_action_size, size=n_agents)
                             for d_action_size in self.d_action_sizes]
            d_action_list = [np.eye(d_action_size, dtype=np.int32)[d_action]
                             for d_action, d_action_size in zip(d_action_list, self.d_action_sizes)]
            d_action = np.concatenate(d_action_list, axis=-1)
            action[:, :self.d_action_summed_size] = d_action

        if self.c_action_size:
            c_action = np.random.randn(n_agents, self.c_action_size)
            # c_action = np.ones((n_agents, self.c_action_size), dtype=np.float32)
            action[:, self.d_action_summed_size:] = c_action

        for i, agent_id in enumerate(agent_ids):
            agent = self.agents_dict[agent_id]
            agent.set_tmp_obs_action(
                obs_list=[o[i] for o in obs_list],
                action=action[i],
                prob=prob[i]
            )

        return action[..., :self.d_action_summed_size], action[..., self.d_action_summed_size:]

    def end_episode(self,
                    agent_ids: np.ndarray,
                    obs_list: List[np.ndarray],
                    last_reward: np.ndarray,
                    max_reached: np.ndarray) -> Dict[str, np.ndarray | List[np.ndarray]]:
        for i, agent_id in enumerate(agent_ids):
            if agent_id not in self.agents_dict:
                continue
            agent = self.agents_dict[agent_id]
            ep_trans = agent.end_transition(
                reward=last_reward[i],
                done=True,
                max_reached=max_reached[i],
                next_obs_list=[o[i] for o in obs_list]
            )
            if ep_trans is not None:
                self._tmp_episode_trans_list.append(ep_trans)

    def force_end_all_episodes(self):
        for agent in self.agents:
            if agent.done:
                continue

            agent.end_transition(
                reward=0,
                done=True,
                max_reached=True
            )

    def train(self) -> int:
        # ep_indexes,
        # ep_obses_list, ep_actions, ep_rewards, ep_dones, ep_probs,
        # ep_seq_hidden_states
        for episode_trans in self._tmp_episode_trans_list:
            self.rl.put_episode(**episode_trans)
        self.clear_tmp_episode_trans_list()

        trained_steps = self.rl.train()

        return trained_steps

    def get_tmp_episode_trans_list(self) -> List[
        Dict[str, np.ndarray | List[np.ndarray]]
    ]:
        return self._tmp_episode_trans_list

    def clear_tmp_episode_trans_list(self) -> None:
        """
        Force clear temporary episode_trans_list to avoid memory leak
        """
        self._tmp_episode_trans_list.clear()

    def log_episode(self) -> None:
        # ep_indexes,
        # ep_obses_list, ep_actions, ep_rewards, ep_dones, ep_probs,
        # ep_seq_hidden_states
        for episode_trans in self._tmp_episode_trans_list:
            self.rl.log_episode(**episode_trans)


class MultiAgentsManager:
    _ma_manager: Dict[str, AgentManager] = {}

    def __init__(self,
                 ma_obs_names: Dict[str, List[str]],
                 ma_obs_shapes: Dict[str, List[Tuple[int, ...]]],
                 ma_d_action_sizes: Dict[str, List[int]],
                 ma_c_action_size: Dict[str, int],
                 inference_ma_names: Set[str],
                 model_abs_dir: Path):
        self._inference_ma_names = inference_ma_names
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

    @property
    def done(self) -> bool:
        return all([mgr.done for n, mgr in self])

    @property
    def max_reached(self) -> bool:
        return any([mgr.max_reached for n, mgr in self])

    def reset(self) -> None:
        """
        Remove all agents
        """
        for n, mgr in self:
            mgr.reset()

    def reset_dead_agents(self) -> None:
        """
        Remove agents where liveness <= 0
        """
        for n, mgr in self:
            mgr.reset_dead_agents()

    def reset_and_continue(self) -> None:
        """
        Agents may continue in a new iteration but save its last status
        """
        for n, mgr in self:
            mgr.reset_and_continue()

    def set_train_mode(self, train_mode: bool = True):
        for n, mgr in self:
            if n in self._inference_ma_names or mgr.rl is None:
                continue
            mgr.rl.set_train_mode(train_mode)

    def get_ma_action(self,
                      ma_agent_ids: Dict[str, np.ndarray],
                      ma_obs_list: Dict[str, List[np.ndarray]],
                      ma_last_reward: Dict[str, np.ndarray],

                      disable_sample: bool = False,
                      force_rnd_if_available: bool = False) -> Tuple[Dict[str, np.ndarray],
                                                                     Dict[str, np.ndarray]]:

        ma_d_action = {}
        ma_c_action = {}

        for n, mgr in self:
            if len(ma_agent_ids[n]) == 0:
                ma_d_action[n] = ma_c_action[n] = None
                continue

            d_action, c_action = mgr.get_action(
                agent_ids=ma_agent_ids[n],
                obs_list=ma_obs_list[n],
                last_reward=ma_last_reward[n],
                disable_sample=disable_sample,
                force_rnd_if_available=force_rnd_if_available
            )
            ma_d_action[n] = d_action
            ma_c_action[n] = c_action

        return ma_d_action, ma_c_action

    def get_test_ma_action(self,
                           ma_agent_ids: Dict[str, np.ndarray],
                           ma_obs_list: Dict[str, List[np.ndarray]],
                           ma_last_reward=None,
                           disable_sample=None) -> Tuple[Dict[str, np.ndarray],
                                                         Dict[str, np.ndarray]]:
        ma_d_action = {}
        ma_c_action = {}

        for n, mgr in self:
            d_action, c_action = mgr.get_test_action(
                agent_ids=ma_agent_ids[n],
                obs_list=ma_obs_list[n]
            )

            ma_d_action[n] = d_action
            ma_c_action[n] = c_action

        return ma_d_action, ma_c_action

    def end_episode(self,
                    ma_agent_ids: Dict[str, np.ndarray],
                    ma_obs_list: Dict[str, List[np.ndarray]],
                    ma_last_reward: Dict[str, float],
                    ma_max_reached: Dict[str, bool]) -> None:
        for n, mgr in self:
            mgr.end_episode(
                agent_ids=ma_agent_ids[n],
                obs_list=ma_obs_list[n],
                last_reward=ma_last_reward[n],
                max_reached=ma_max_reached[n]
            )

    def force_end_all_episode(self):
        for n, mgr in self:
            mgr.force_end_all_episodes()

    def train(self, trained_steps: int) -> int:
        for n, mgr in self:
            if n in self._inference_ma_names:
                continue
            trained_steps = max(mgr.train(), trained_steps)

        return trained_steps

    def log_episode(self) -> None:
        for n, mgr in self:
            if n in self._inference_ma_names:
                continue
            mgr.log_episode()

    def save_model(self, save_replay_buffer=False) -> None:
        for n, mgr in self:
            if n in self._inference_ma_names or mgr.rl is None:
                continue
            mgr.rl.save_model(save_replay_buffer)

    def clear_tmp_episode_trans_list(self) -> None:
        """
        Force clear temporary episode_trans_list to avoid memory leak
        """
        for n, mgr in self:
            mgr.clear_tmp_episode_trans_list()
