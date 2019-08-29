from collections import deque
import numpy as np


class Agent(object):
    reward = 0
    done = False

    def __init__(self, agent_id, gamma=0.99, n_step=1, use_rnn=False):
        self.agent_id = agent_id
        self.gamma = gamma
        self.n_step = n_step
        self.use_rnn = use_rnn

        self._tmp_trans = deque(maxlen=self.n_step)

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
                trans = [[]] * 6
            else:
                trans = [[]] * 4
        else:
            trans = [[t] for t in self._get_trans()]

        if local_done:
            self.done = True
            self._tmp_trans.clear()

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
        if len(self._tmp_trans) < self.n_step:
            return None

        r = 0.
        lstm_state_c = self._tmp_trans[0]['lstm_state_c']
        lstm_state_h = self._tmp_trans[0]['lstm_state_h']

        for i, tran in enumerate(self._tmp_trans):
            r += np.power(self.gamma, i) * tran['reward']

        done = self._tmp_trans[-1]['local_done'] and not self._tmp_trans[-1]['max_reached']

        if self.use_rnn:
            return ([t['state'] for t in self._tmp_trans],
                    [t['action'] for t in self._tmp_trans],
                    [r], [done],
                    lstm_state_c, lstm_state_h)
        else:
            return ([t['state'] for t in self._tmp_trans],
                    [t['action'] for t in self._tmp_trans],
                    [r], [done])
