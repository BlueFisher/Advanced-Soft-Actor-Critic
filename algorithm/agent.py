from collections import deque
import numpy as np


class Agent(object):
    reward = 0
    done = False

    def __init__(self, agent_id,
                 tran_len=1, stagger=1,
                 use_rnn=False):
        self.agent_id = agent_id
        self.deque_maxlen = tran_len
        self._curr_stagger = self.stagger = stagger
        self.use_rnn = use_rnn

        self._tmp_trans = deque(maxlen=self.deque_maxlen)

    def add_transition(self,
                       state,
                       action,
                       reward,
                       local_done,
                       max_reached,
                       state_,
                       rnn_state=None):

        self._tmp_trans.append({
            'state': state,
            'action': action,
            'reward': reward,
            'local_done': local_done,
            'max_reached': max_reached,
            'state_': state_,
            'rnn_state': rnn_state
        })

        if not self.done:
            self.reward += reward

        self._extra_log(state,
                        action,
                        reward,
                        local_done,
                        max_reached,
                        state_)

        trans_list = self._get_trans()
        if trans_list is not None:
            trans_list = [np.asarray([t], dtype=np.float32) for t in trans_list]

        if local_done:
            self.done = True
            self._tmp_trans.clear()
            self._curr_stagger = self.stagger

        return trans_list

    def _extra_log(self,
                   state,
                   action,
                   reward,
                   local_done,
                   max_reached,
                   state_):
        pass

    def _get_trans(self):
        if len(self._tmp_trans) < self.deque_maxlen:
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
