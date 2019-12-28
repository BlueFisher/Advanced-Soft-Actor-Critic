from collections import deque
import numpy as np


class Agent(object):
    reward = 0
    last_reward = 0
    done = False

    def __init__(self, agent_id,
                 tran_len=1, stagger=1,
                 use_rnn=False):
        self.agent_id = agent_id
        self.tran_len = tran_len
        self._curr_stagger = self.stagger = stagger
        self.use_rnn = use_rnn

        self._tmp_trans = deque(maxlen=self.tran_len)
        self._tmp_episode_trans = list()

    def add_transition(self,
                       state,
                       action,
                       reward,
                       local_done,
                       max_reached,
                       state_,
                       rnn_state=None):

        transition = {
            'state': state,
            'action': action,
            'reward': reward,
            'local_done': local_done,
            'max_reached': max_reached,
            'state_': state_,
            'rnn_state': rnn_state
        }
        self._tmp_trans.append(transition)
        self._tmp_episode_trans.append(transition)

        if not self.done:
            self.reward += reward
        self.last_reward += reward

        self._extra_log(state,
                        action,
                        reward,
                        local_done,
                        max_reached,
                        state_)

        trans = self._get_trans()
        if trans is not None:
            trans = [np.asarray([t], dtype=np.float32) for t in trans]

        episode_trans = None

        if local_done:
            self.done = True
            self.last_reward = 0

            self._tmp_trans.clear()
            self._curr_stagger = self.stagger

            episode_trans = self._get_episode_trans()
            episode_trans = [np.asarray([t], dtype=np.float32) for t in episode_trans]
            self._tmp_episode_trans.clear()

        return trans, episode_trans

    def _extra_log(self,
                   state,
                   action,
                   reward,
                   local_done,
                   max_reached,
                   state_):
        pass

    def _get_trans(self):
        if len(self._tmp_trans) < self.tran_len:
            return None

        if self._curr_stagger != self.stagger:
            self._curr_stagger += 1
            return None

        self._curr_stagger = 1

        state_ = self._tmp_trans[-1]['state_']
        done = self._tmp_trans[-1]['local_done'] and not self._tmp_trans[-1]['max_reached']

        if self.use_rnn:
            rnn_state = self._tmp_trans[0]['rnn_state']

            # n_states, n_actions, n_rewards, state_, done, rnn_state
            return ([t['state'] for t in self._tmp_trans],
                    [t['action'] for t in self._tmp_trans],
                    [t['reward'] for t in self._tmp_trans],
                    state_,
                    [done],
                    rnn_state)
        else:
            return ([t['state'] for t in self._tmp_trans],
                    [t['action'] for t in self._tmp_trans],
                    [t['reward'] for t in self._tmp_trans],
                    state_,
                    [done])

    def _get_episode_trans(self):
        state_ = self._tmp_episode_trans[-1]['state_']
        done = self._tmp_episode_trans[-1]['local_done'] and not self._tmp_episode_trans[-1]['max_reached']

        return ([t['state'] for t in self._tmp_episode_trans],
                [t['action'] for t in self._tmp_episode_trans],
                [t['reward'] for t in self._tmp_episode_trans],
                state_,
                [done])

    def is_empty(self):
        return len(self._tmp_episode_trans) == 0

    def clear(self):
        self.reward = 0
        self.done = False
        self._tmp_trans.clear()
        self._tmp_episode_trans.clear()
    
    def reset(self):
        self.reward = self.last_reward
        self.done = False


if __name__ == "__main__":
    agent = Agent(0, 6)
    for i in range(20):
        print(i, '===')
        a, b = agent.add_transition(np.random.randn(4), np.random.randn(2), 1, False, False, np.random.randn(4))
        print(a, b)

    print(20, '===')
    a, b = agent.add_transition(np.random.randn(4), np.random.randn(2), 1, True, False, np.random.randn(4))
    print(a, b)
