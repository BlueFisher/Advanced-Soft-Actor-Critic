import time
import sys
import threading
from pathlib import Path

import numpy as np
import tensorflow as tf

from sac_ds_base import SAC_DS_Base

sys.path.append(str(Path(__file__).resolve().parent.parent))
from algorithm.replay_buffer import PrioritizedReplayBuffer


class SAC_DS_with_Replay_Base(SAC_DS_Base):
    def __init__(self,
                 state_dim,
                 action_dim,
                 only_actor=False,
                 saver_model_path='model',
                 summary_path='log',
                 summary_name=None,
                 write_summary_graph=False,
                 seed=None,
                 gamma=0.99,
                 tau=0.005,
                 save_model_per_step=5000,
                 write_summary_per_step=20,
                 update_target_per_step=1,
                 init_log_alpha=-4.6,
                 use_auto_alpha=True,
                 lr=3e-4,
                 batch_size=256,
                 replay_buffer_capacity=1e6):

        self.replay_buffer = PrioritizedReplayBuffer(batch_size, replay_buffer_capacity)
        super().__init__(state_dim,
                         action_dim,
                         only_actor,
                         saver_model_path,
                         summary_path,
                         summary_name,
                         write_summary_graph,
                         seed,
                         gamma,
                         tau,
                         save_model_per_step,
                         write_summary_per_step,
                         update_target_per_step,
                         init_log_alpha,
                         use_auto_alpha,
                         lr)

    _replay_lock = threading.Lock()

    def add(self, s, a, r, s_, done):
        assert not self.only_actor

        self._replay_lock.acquire()
        self.replay_buffer.add(s, a, r, s_, done)
        self._replay_lock.release()


    def train(self):
        assert not self.only_actor

        if self.replay_buffer.size == 0:
            return

        global_step = self.sess.run(self.global_step)

        start = time.time()
        self._replay_lock.acquire()
        points, (s, a, r, s_, done), is_weight = self.replay_buffer.sample()
        self._replay_lock.release()
        print(time.time() - start)

        # update target networks
        if global_step % self.update_target_per_step == 0:
            self.sess.run(self.update_target_op)

        if global_step % self.write_summary_per_step == 0:
            summaries = self.sess.run(self.summaries, {
                self.pl_s: s,
                self.pl_a: a,
                self.pl_r: r,
                self.pl_s_: s_,
                self.pl_done: done,
                self.pl_is: is_weight
            })
            self.summary_writer.add_summary(summaries, global_step)

        if global_step % self.save_model_per_step == 0:
            self.saver.save(global_step)

        if self.replay_buffer.is_lg_batch_size:
            self.sess.run(self.train_q_ops, {
                self.pl_s: s,
                self.pl_a: a,
                self.pl_r: r,
                self.pl_s_: s_,
                self.pl_done: done,
                self.pl_is: np.zeros((1, 1)) if not self.use_priority else is_weight
            })

            self.sess.run(self.train_policy_op, {
                self.pl_s: s,
            })

            if self.use_auto_alpha:
                self.sess.run(self.train_alpha_op, {
                    self.pl_s: s,
                })

            if self.use_priority:
                td_error = self.sess.run(self.td_error, {
                    self.pl_s: s,
                    self.pl_a: a,
                    self.pl_r: r,
                    self.pl_s_: s_,
                    self.pl_done: done
                })

                self.replay_buffer.update(points, td_error.flatten())

        return global_step
