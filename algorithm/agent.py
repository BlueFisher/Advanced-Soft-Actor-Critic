from collections import deque
import numpy as np


class Agent(object):
    reward = 0
    done = False

    def __init__(self, agent_id, gamma=0.99, n_step=1):
        self.agent_id = agent_id
        self.gamma = gamma
        self.n_step = n_step

        self._tmp_trans = deque(maxlen=self.n_step)

    def add_transition(self,
                       state,
                       action,
                       reward,
                       local_done,
                       max_reached,
                       state_):

        if type(state) == np.ndarray:
            state = state.tolist()
        if type(action) == np.ndarray:
            action = action.tolist()
        if type(state_) == np.ndarray:
            state_ = state_.tolist()

        self._tmp_trans.append({
            'state': state,
            'action': action,
            'reward': reward,
            'local_done': local_done,
            'max_reached': max_reached,
            'state_': state_,
        })

        if not self.done:
            self.reward += reward

        self._extra_log(state,
                        action,
                        reward,
                        local_done,
                        max_reached,
                        state_)

        trans = [[t] for t in self._get_trans()]

        if local_done:
            self.done = True

            self._tmp_trans.popleft()
            while len(self._tmp_trans) != 0:
                _trans = self._get_trans()
                for i in range(len(trans)):
                    trans[i].append(_trans[i])
                self._tmp_trans.popleft()

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
        s, a, r = self._tmp_trans[0]['state'], self._tmp_trans[0]['action'], 0.

        for i, tran in enumerate(self._tmp_trans):
            r += np.power(self.gamma, i) * tran['reward']

        s_ = self._tmp_trans[-1]['state_']
        done = self._tmp_trans[-1]['local_done'] and not self._tmp_trans[-1]['max_reached']
        gamma = np.power(self.gamma, len(self._tmp_trans))

        return s, a, [r], s_, [done], [gamma], [t['state'] for t in self._tmp_trans], [t['action'] for t in self._tmp_trans]
