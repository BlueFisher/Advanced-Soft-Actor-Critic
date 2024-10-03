import importlib
import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from algorithm.oc.option_selector_base import OptionSelectorBase
from algorithm.utils.enums import *
from tests.get_synthesis_data import *

OBS_NAMES = ['vector', 'image']
OBS_SHAPES = [(10,), (30, 30, 3)]


class TestOCModel(unittest.TestCase):
    def _test(self, param_dict):
        convert_config_to_enum(param_dict)

        import tests.nn_oc_combined as nn_oc_combined
        importlib.reload(nn_oc_combined)

        if param_dict['seq_encoder'] is None:
            nn_oc_combined.ModelOptionSelectorRep = nn_oc_combined.ModelOptionSelectorVanillaRep
        elif param_dict['seq_encoder'] == SEQ_ENCODER.RNN:
            nn_oc_combined.ModelOptionSelectorRep = nn_oc_combined.ModelOptionSelectorRNNRep
        elif param_dict['seq_encoder'] == SEQ_ENCODER.ATTN:
            nn_oc_combined.ModelOptionSelectorRep = nn_oc_combined.ModelOptionSelectorAttentionRep

        if param_dict['option_seq_encoder'] is None:
            nn_oc_combined.ModelRep = nn_oc_combined.ModelVanillaOptionRep
        elif param_dict['option_seq_encoder'] == SEQ_ENCODER.RNN:
            nn_oc_combined.ModelRep = nn_oc_combined.ModelRNNOptionRep

        sac = OptionSelectorBase(
            option_nn_config=None,

            obs_names=OBS_NAMES,
            obs_shapes=OBS_SHAPES,
            model_abs_dir=None,
            nn=nn_oc_combined,
            batch_size=10,
            **param_dict
        )

        seq_hidden_state_shape = sac.seq_hidden_state_shape
        low_seq_hidden_state_shape = sac.low_seq_hidden_state_shape

        step = 0
        while step < 10:
            if param_dict['seq_encoder'] in (None, SEQ_ENCODER.RNN):
                # The same as dialated
                sac.choose_action(**gen_batch_oc_obs(OBS_SHAPES,
                                                     d_action_sizes=param_dict['d_action_sizes'],
                                                     c_action_size=param_dict['c_action_size'],
                                                     seq_hidden_state_shape=seq_hidden_state_shape,
                                                     low_seq_hidden_state_shape=low_seq_hidden_state_shape))
            elif param_dict['seq_encoder'] == SEQ_ENCODER.ATTN:
                if param_dict['use_dilation']:
                    sac.choose_dilated_attn_action(**gen_batch_oc_obs_for_dilated_attn(OBS_SHAPES,
                                                                                       d_action_sizes=param_dict['d_action_sizes'],
                                                                                       c_action_size=param_dict['c_action_size'],
                                                                                       seq_hidden_state_shape=seq_hidden_state_shape,
                                                                                       low_seq_hidden_state_shape=low_seq_hidden_state_shape))
                else:
                    sac.choose_attn_action(**gen_batch_oc_obs_for_attn(OBS_SHAPES,
                                                                       d_action_sizes=param_dict['d_action_sizes'],
                                                                       c_action_size=param_dict['c_action_size'],
                                                                       seq_hidden_state_shape=seq_hidden_state_shape,
                                                                       low_seq_hidden_state_shape=low_seq_hidden_state_shape))
            sac.put_episode(**gen_episode_oc_trans(OBS_SHAPES,
                                                   d_action_sizes=param_dict['d_action_sizes'],
                                                   c_action_size=param_dict['c_action_size'],
                                                   seq_hidden_state_shape=seq_hidden_state_shape,
                                                   low_seq_hidden_state_shape=low_seq_hidden_state_shape,
                                                   episode_len=40))
            step = sac.train()

    @staticmethod
    def gen(param_dict):
        def func(self):
            self._test(param_dict)
        return func


def __gen_test(test_from: int):
    param_dict_candidates = {
        'seq_encoder': [None, 'RNN', 'ATTN'],
        'use_dilation': [False],
        'option_burn_in_step': [-1, 2],
        'option_seq_encoder': [None, 'RNN'],
        'd_action_sizes': [[3, 3, 4]],
        'c_action_size': [4],
        'use_replay_buffer': [True, False],
        'burn_in_step': [5],
        'n_step': [3],
        'discrete_dqn_like': [True, False],
        'use_rnd': [True, False],
    }

    possible_param_dicts = get_product(param_dict_candidates)

    i = 0
    for param_dict in possible_param_dicts:
        if param_dict['option_seq_encoder'] is None:
            if param_dict['option_burn_in_step'] == -1:
                param_dict['option_burn_in_step'] = 0
            else:
                continue

        if not param_dict['d_action_sizes'] and not param_dict['c_action_size']:
            continue

        if 'use_prediction' in param_dict and not param_dict['use_prediction']:
            if param_dict['use_extra_data']:
                continue

        if i < test_from:
            i += 1
            continue

        func_name = f'test_{i:04d}'
        for k, v in param_dict.items():
            v = str(v).replace('.', '_')
            if len(param_dict_candidates[k]) > 1:
                func_name += f', {k}={v}'

        setattr(TestOCModel, func_name,
                TestOCModel.gen(param_dict))

        i += 1


test_from = 0
for arg in sys.argv:
    if arg.startswith('--from'):
        test_from = int(arg.split('--from=')[1])

__gen_test(test_from)
