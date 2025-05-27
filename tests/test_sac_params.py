import importlib
import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from algorithm.sac_base import SAC_Base
from algorithm.utils.enums import *
from tests.get_synthesis_data import *

OBS_NAMES = ['vector', 'image']
OBS_SHAPES = [(10,), (3, 30, 30)]


class TestModel(unittest.TestCase):
    def _test(self, param_dict):
        convert_config_to_enum(param_dict)

        if param_dict['seq_encoder'] is None:
            import tests.nn_conv_vanilla as nn_conv
        elif param_dict['seq_encoder'] == SEQ_ENCODER.RNN:
            import tests.nn_conv_rnn as nn_conv
        elif param_dict['seq_encoder'] == SEQ_ENCODER.ATTN:
            import tests.nn_conv_attn as nn_conv

        importlib.reload(nn_conv)

        sac = SAC_Base(
            obs_names=OBS_NAMES,
            obs_shapes=OBS_SHAPES,
            model_abs_dir=None,
            nn=nn_conv,
            **param_dict
        )

        seq_hidden_state_shape = sac.seq_hidden_state_shape

        step = 0
        while step < 10:
            if param_dict['seq_encoder'] in (None, SEQ_ENCODER.RNN):
                sac.choose_action(**gen_batch_obs(OBS_SHAPES,
                                                  d_action_sizes=param_dict['d_action_sizes'],
                                                  c_action_size=param_dict['c_action_size'],
                                                  seq_hidden_state_shape=seq_hidden_state_shape))
            elif param_dict['seq_encoder'] == SEQ_ENCODER.ATTN:
                sac.choose_attn_action(**gen_batch_obs_for_attn(OBS_SHAPES,
                                                                d_action_sizes=param_dict['d_action_sizes'],
                                                                c_action_size=param_dict['c_action_size'],
                                                                seq_hidden_state_shape=seq_hidden_state_shape))
            sac.put_episode(**gen_episode_trans(OBS_SHAPES,
                                                d_action_sizes=param_dict['d_action_sizes'],
                                                c_action_size=param_dict['c_action_size'],
                                                seq_hidden_state_shape=seq_hidden_state_shape,
                                                episode_len=random.randint(30, 100)))
            step = sac.train()

        sac.close()

    @staticmethod
    def gen(param_dict):
        def func(self):
            self._test(param_dict)
        return func


def __gen_test(test_from: int):
    param_dict_candidates = {
        'd_action_sizes': [[3, 3, 4]],
        'c_action_size': [4],
        'use_replay_buffer': [True, False],
        'burn_in_step': [5],
        'n_step': [3],
        'seq_encoder': [None, 'RNN', 'ATTN'],
        'discrete_dqn_like': [True, False],
        'siamese': [None, 'ATC', 'BYOL'],
        'siamese_use_q': [False, True],
        'siamese_use_adaptive': [False, True],
        'use_n_step_is': [True, False],
        'use_rnd': [True, False],
        # 'action_noise': [None, [0.1, 0.1]]
    }

    possible_param_dicts = get_product(param_dict_candidates)

    i = 0
    for param_dict in possible_param_dicts:
        if not param_dict['d_action_sizes'] and not param_dict['c_action_size']:
            continue

        if 'siamese' in param_dict and not param_dict['siamese']:
            if param_dict['siamese_use_q'] or param_dict['siamese_use_adaptive']:
                continue

        if i < test_from:
            i += 1
            continue

        func_name = f'test_{i:04d}'
        for k, v in param_dict.items():
            v = str(v).replace('.', '_')
            if len(param_dict_candidates[k]) > 1:
                func_name += f', {k}={v}'

        setattr(TestModel, func_name,
                TestModel.gen(param_dict))

        i += 1


test_from = 0
for arg in sys.argv:
    if arg.startswith('--from'):
        test_from = int(arg.split('--from=')[1])

__gen_test(test_from)
