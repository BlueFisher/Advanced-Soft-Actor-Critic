from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from algorithm.sac_base import SAC_Base

from algorithm.utils.enums import *
from algorithm.utils.operators import gen_pre_n_actions


class Agent:
    reward = 0  # The reward of the first complete episode
    _last_reward = 0  # The reward of the last episode
    steps = 0  # The step count of the first complete episode
    _last_steps = 0  # The step count of the last episode
    done = False  # If has one complete episode
    max_reached = False

    def __init__(self, agent_id: int,
                 obs_shapes: List[Tuple],
                 action_size: int,
                 seq_hidden_state_shape=None,
                 max_return_episode_trans=-1):
        self.agent_id = agent_id
        self.obs_shapes = obs_shapes
        self.action_size = action_size
        self.seq_hidden_state_shape = seq_hidden_state_shape
        self.max_return_episode_trans = max_return_episode_trans

        self._tmp_episode_trans = self._generate_empty_episode_trans()

    def _generate_empty_episode_trans(self, episode_length: int = 0):
        return {
            'index': -np.ones((episode_length, ), dtype=int),
            'padding_mask': np.ones((episode_length, ), dtype=bool),
            'obs_list': [np.zeros((episode_length, *s), dtype=np.float32) for s in self.obs_shapes],
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

    @property
    def episode_length(self):
        return len(self._tmp_episode_trans['obs_list'][0])

    def _extra_log(self,
                   obs_list,
                   action,
                   reward,
                   local_done,
                   max_reached,
                   next_obs_list,
                   prob):
        pass

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
                ep_actions,
                ep_rewards,
                next_obs_list,
                ep_dones,
                ep_probs,
                ep_seq_hidden_states)

    def is_empty(self):
        return self.episode_length == 0

    def clear(self):
        self.reward = 0
        self.steps = 0
        self.done = False
        self.max_reached = False
        self._tmp_episode_trans = self._generate_empty_episode_trans()

    def reset(self):
        """
        The agent may continue in a new iteration but save its last status
        """
        self.reward = self._last_reward
        self.steps = self._last_steps
        self.done = False
        self.max_reached = False


class AgentManager:
    def __init__(self,
                 agent_class: Agent,
                 obs_shapes: List[Tuple[int]],
                 d_action_size: int,
                 c_action_size: int):
        self._agent_class = agent_class

        self.obs_shapes = obs_shapes
        self.d_action_size = d_action_size
        self.c_action_size = c_action_size
        self.action_size = d_action_size + c_action_size

        self._data = {}

    def __getitem__(self, k):
        return self._data[k]

    def __setitem__(self, k, v):
        self._data[k] = v

    def set_sac(self, sac: SAC_Base):
        self.sac = sac
        self.seq_encoder = sac.seq_encoder

    def set_agents(self, num_agents: int):
        self.agents: List[Agent] = [
            self._agent_class(i, self.obs_shapes, self.action_size,
                              seq_hidden_state_shape=self.sac.seq_hidden_state_shape
                              if self.seq_encoder is not None else None)
            for i in range(num_agents)
        ]

    def get_action(self,
                   disable_sample: bool = False,
                   force_rnd_if_available: bool = False):
        if self.seq_encoder == SEQ_ENCODER.RNN:
            action, prob, next_seq_hidden_state = self.sac.choose_rnn_action(self['obs_list'],
                                                                             self['pre_action'],
                                                                             self['seq_hidden_state'],
                                                                             disable_sample=disable_sample,
                                                                             force_rnd_if_available=force_rnd_if_available)

        elif self.seq_encoder == SEQ_ENCODER.ATTN:
            ep_length = min(512, max([a.episode_length for a in self.agents]))

            all_episode_trans = [a.get_episode_trans(ep_length) for a in self.agents]
            (all_ep_indexes,
                all_ep_padding_masks,
                all_ep_obses_list,
                all_ep_actions,
                all_all_ep_rewards,
                all_next_obs_list,
                all_ep_dones,
                all_ep_probs,
                all_ep_attn_states) = zip(*all_episode_trans)

            ep_indexes = np.concatenate(all_ep_indexes)
            ep_padding_masks = np.concatenate(all_ep_padding_masks)
            ep_obses_list = [np.concatenate(o) for o in zip(*all_ep_obses_list)]
            ep_actions = np.concatenate(all_ep_actions)
            ep_attn_states = np.concatenate(all_ep_attn_states)

            ep_indexes = np.concatenate([ep_indexes, ep_indexes[:, -1:] + 1], axis=1)
            ep_padding_masks = np.concatenate([ep_padding_masks,
                                               np.zeros_like(ep_padding_masks[:, -1:], dtype=bool)], axis=1)
            ep_obses_list = [np.concatenate([o, np.expand_dims(t_o, 1)], axis=1)
                             for o, t_o in zip(ep_obses_list, self['obs_list'])]
            ep_pre_actions = gen_pre_n_actions(ep_actions, True)

            action, prob, next_seq_hidden_state = self.sac.choose_attn_action(ep_indexes,
                                                                              ep_padding_masks,
                                                                              ep_obses_list,
                                                                              ep_pre_actions,
                                                                              ep_attn_states,
                                                                              disable_sample=disable_sample,
                                                                              force_rnd_if_available=force_rnd_if_available)

        else:
            action, prob = self.sac.choose_action(self['obs_list'],
                                                  disable_sample=disable_sample,
                                                  force_rnd_if_available=force_rnd_if_available)
            next_seq_hidden_state = None

        self['action'] = action
        self['d_action'] = action[..., :self.d_action_size]
        self['c_action'] = action[..., self.d_action_size:]
        self['prob'] = prob
        self['next_seq_hidden_state'] = next_seq_hidden_state


class MultiAgentsManager:
    def __init__(self,
                 agent_class: Agent,
                 ma_obs_shapes,
                 ma_d_action_size,
                 ma_c_action_size):
        self._agent_class = agent_class

        self._ma_manager: Dict[str, AgentManager] = {}
        for n in ma_obs_shapes:
            self._ma_manager[n] = AgentManager(agent_class,
                                               ma_obs_shapes[n],
                                               ma_d_action_size[n],
                                               ma_c_action_size[n])

    def __iter__(self) -> Iterator[Tuple[str, AgentManager]]:
        return iter(self._ma_manager.items())

    def __getitem__(self, k) -> AgentManager:
        return self._ma_manager[k]

    def __len__(self):
        return len(self._ma_manager)

    def init(self, num_agents):
        for n, mgr in self:
            mgr['initial_pre_action'] = mgr.sac.get_initial_action(num_agents)  # [n_agents, action_size]
            mgr['pre_action'] = mgr['initial_pre_action']
            if mgr.seq_encoder is not None:
                mgr['initial_seq_hidden_state'] = mgr.sac.get_initial_seq_hidden_state(num_agents)  # [n_agents, *seq_hidden_state_shape]
                mgr['seq_hidden_state'] = mgr['initial_seq_hidden_state']

            mgr.set_agents(num_agents)

    def is_max_reached(self):
        return any([any([a.max_reached for a in mgr.agents]) for n, mgr in self])

    def is_done(self):
        return all([all([a.done for a in mgr.agents]) for n, mgr in self])

    def set_obs_list(self, ma_obs_list):
        for n, mgr in self:
            mgr['obs_list'] = ma_obs_list[n]

    def clear(self):
        for n, mgr in self:
            for a in mgr.agents:
                a.clear()

    def reset(self):
        for n, mgr in self:
            for a in mgr.agents:
                a.reset()

    def burn_in_padding(self):
        for n, mgr in self:
            for a in [a for a in mgr.agents if a.is_empty()]:
                for _ in range(mgr.sac.burn_in_step):
                    a.add_transition(
                        obs_list=[np.zeros(t, dtype=np.float32) for t in mgr.obs_shapes],
                        action=mgr['initial_pre_action'][0],
                        reward=0.,
                        local_done=False,
                        max_reached=False,
                        next_obs_list=[np.zeros(t, dtype=np.float32) for t in mgr.obs_shapes],
                        prob=0.,
                        is_padding=True,
                        seq_hidden_state=mgr['initial_seq_hidden_state'][0]
                    )

    def get_ma_action(self,
                      disable_sample: bool = False,
                      force_rnd_if_available: bool = False):
        for n, mgr in self:
            mgr.get_action(disable_sample, force_rnd_if_available)

        ma_d_action = {n: mgr['d_action'] for n, mgr in self}
        ma_c_action = {n: mgr['c_action'] for n, mgr in self}

        return ma_d_action, ma_c_action

    def post_step(self, ma_next_obs_list, ma_local_done):
        self.set_obs_list(ma_next_obs_list)

        for n, mgr in self:
            mgr['pre_action'] = mgr['action']
            mgr['pre_action'][ma_local_done[n]] = mgr['initial_pre_action'][ma_local_done[n]]
            if mgr.seq_encoder is not None:
                mgr['seq_hidden_state'] = mgr['next_seq_hidden_state']
                mgr['seq_hidden_state'][ma_local_done[n]] = mgr['initial_seq_hidden_state'][ma_local_done[n]]

    def save_model(self):
        for n, mgr in self:
            mgr.sac.save_model()


if __name__ == "__main__":
    obs_shapes = [(4,), (4, 4, 3)]
    action_size = 2
    rnn_state_size = 6
    agent = Agent(0, obs_shapes, action_size, rnn_state_size)

    def print_episode_trans(episode_trans):
        for e in episode_trans:
            if isinstance(e, list):
                print([o.shape for o in e])
            elif e is not None:
                print(e.shape)
            else:
                print('None')

    rnn_state = np.random.randn(rnn_state_size)
    for i in range(10):
        a = agent.add_transition([np.random.randn(*s) for s in obs_shapes],
                                 np.random.randn(action_size), 1., False, False,
                                 [np.random.randn(*s) for s in obs_shapes],
                                 2.,
                                 rnn_state=rnn_state)

    # episode_trans = agent.add_transition([np.random.randn(*s) for s in obs_shapes],
    #                                      np.random.randn(action_size), 1, True, False,
    #                                      [np.random.randn(*s) for s in obs_shapes],
    #                                      2.,
    #                                      rnn_state=rnn_state)
    episode_trans = agent.get_episode_trans(20)

    for i in range(3):
        a = agent.add_transition([np.random.randn(*s) for s in obs_shapes],
                                 np.random.randn(action_size), 1., False, False,
                                 [np.random.randn(*s) for s in obs_shapes],
                                 2.,
                                 rnn_state=rnn_state)

    episode_trans = agent.get_episode_trans(20)
    print_episode_trans(episode_trans)
    print(agent.episode_length)
    for i in range(3):
        a = agent.add_transition([np.random.randn(*s) for s in obs_shapes],
                                 np.random.randn(action_size), 1., False, False,
                                 [np.random.randn(*s) for s in obs_shapes],
                                 2.,
                                 rnn_state=rnn_state)

    print(agent.episode_length)
