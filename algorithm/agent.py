from collections import deque
import numpy as np


class Agent(object):
    reward = 0
    done = False

    def __init__(self, agent_id,
                 gamma=0.99, tran_len=1, stagger=1,
                 use_rnn=False):
        self.agent_id = agent_id
        self.gamma = gamma
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
                       lstm_state_c=None,
                       lstm_state_h=None):

        if type(state) == np.ndarray:
            state = state.tolist()
        if type(action) == np.ndarray:
            action = action.tolist()
        if type(state_) == np.ndarray:
            state_ = state_.tolist()
        if type(lstm_state_c) == np.ndarray:
            lstm_state_c = lstm_state_c.tolist()
        if type(lstm_state_h) == np.ndarray:
            lstm_state_h = lstm_state_h.tolist()

        self._tmp_trans.append({
            'state': state,
            'action': action,
            'reward': reward,
            'local_done': local_done,
            'max_reached': max_reached,
            'state_': state_,
            'lstm_state_c': lstm_state_c,
            'lstm_state_h': lstm_state_h
        })

        if not self.done:
            self.reward += reward

        self._extra_log(state,
                        action,
                        reward,
                        local_done,
                        max_reached,
                        state_)

        trans = self._get_trans()
        if trans is None:
            if self.use_rnn:
                trans = [[]] * 7
            else:
                trans = [[]] * 5
        else:
            trans = [[t] for t in trans]

        if local_done:
            self.done = True
            self._tmp_trans.clear()
            self._curr_stagger = self.stagger

        return trans

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
            lstm_state_c = self._tmp_trans[0]['lstm_state_c']
            lstm_state_h = self._tmp_trans[0]['lstm_state_h']

            return ([t['state'] for t in self._tmp_trans],
                    [t['action'] for t in self._tmp_trans],
                    [t['reward'] for t in self._tmp_trans],
                    state_,
                    [done],
                    lstm_state_c, lstm_state_h)
        else:
            return ([t['state'] for t in self._tmp_trans],
                    [t['action'] for t in self._tmp_trans],
                    [t['reward'] for t in self._tmp_trans],
                    state_,
                    [done])
