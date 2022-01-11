from typing import List, Optional, Tuple

import numpy as np
from numpy.lib.shape_base import expand_dims


class Agent(object):
    reward = 0  # The reward of the first complete episode
    _last_reward = 0  # The reward of the last episode
    steps = 0  # The step count of the first complete episode
    _last_steps = 0  # The step count of the last episode
    done = False  # If has one complete episode
    max_reached = False

    def __init__(self, agent_id: int,
                 obs_shapes: List[Tuple],
                 action_size: int,
                 rnn_state_size=None,
                 max_return_episode_trans=-1):
        self.agent_id = agent_id
        self.obs_shapes = obs_shapes
        self.action_size = action_size
        self.rnn_state_size = rnn_state_size
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
            'rnn_state': np.zeros((episode_length, self.rnn_state_size), dtype=np.float32) if self.rnn_state_size is not None else None,
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
                       rnn_state: Optional[np.ndarray] = None):
        """
        Args:
            obs_list: List([*obs_shapes_i], ...)
            action: [action_size, ]
            reward: float
            local_done: bool
            max_reached: bool
            next_obs_list: List([*obs_shapes_i], ...)
            prob: float
            rnn_state: [rnn_state_size, ]

        Returns:
            ep_indexes: [1, episode_len], int
            ep_padding_masks: [1, episode_len], bool
            ep_obses_list: List([1, episode_len, *obs_shapes_i], ...), np.float32
            ep_actions: [1, episode_len, action_size], np.float32
            ep_rewards: [1, episode_len], np.float32
            next_obs_list: List([1, *obs_shapes_i], ...), np.float32
            ep_dones: [1, episode_len], bool
            ep_probs: [1, episode_len], np.float32
            ep_rnn_states: [1, episode_len, rnn_state_size], np.float32
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
            'rnn_state': np.expand_dims(rnn_state, 0).astype(np.float32) if rnn_state is not None else None,
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
        ep_rnn_states = np.expand_dims(tmp['rnn_state'], 0) if tmp['rnn_state'] is not None else None
        # [1, episode_len, rnn_state_size]

        return (ep_indexes,
                ep_padding_masks,
                ep_obses_list,
                ep_actions,
                ep_rewards,
                next_obs_list,
                ep_dones,
                ep_probs,
                ep_rnn_states)

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
