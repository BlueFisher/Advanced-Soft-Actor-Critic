import importlib
import unittest

from algorithm.oc.option_selector_base import OptionSelectorBase
from algorithm.utils.enums import *
from tests.get_synthesis_data import *

OBS_NAMES = ['vector', 'image']
OBS_SHAPES = [(10,), (30, 30, 3)]
NUM_OPTIONS = 4


class TestOCVanillaModel(unittest.TestCase):
    def _test(self, param_dict):
        convert_config_to_enum(param_dict)

        import tests.nn_conv_vanilla as nn_conv

        importlib.reload(nn_conv)

        sac = OptionSelectorBase(
            num_options=NUM_OPTIONS,
            option_burn_in_step=-1,
            option_nn_config=None,

            obs_names=OBS_NAMES,
            obs_shapes=OBS_SHAPES,
            model_abs_dir=None,
            nn=nn_conv,
            **param_dict
        )

        step = 0
        while step < 10:
            sac.choose_action(**gen_batch_oc_obs(OBS_SHAPES))
            sac.put_episode(**gen_episode_oc_trans(OBS_SHAPES,
                                                   d_action_sizes=param_dict['d_action_sizes'],
                                                   c_action_size=param_dict['c_action_size'],
                                                   episode_len=10))
            step = sac.train()

    @staticmethod
    def gen(param_dict):
        def func(self):
            self._test(param_dict)
        return func


class TestOCSeqEncoderModel(unittest.TestCase):
    def _test(self, param_dict):
        convert_config_to_enum(param_dict)

        if param_dict['seq_encoder'] == SEQ_ENCODER.RNN:
            import tests.nn_conv_rnn as nn_conv
        elif param_dict['seq_encoder'] == SEQ_ENCODER.ATTN:
            import tests.nn_conv_attn as nn_conv

        importlib.reload(nn_conv)

        sac = OptionSelectorBase(
            num_options=NUM_OPTIONS,
            option_nn_config=None,

            obs_names=OBS_NAMES,
            obs_shapes=OBS_SHAPES,
            model_abs_dir=None,
            nn=nn_conv,
            **param_dict
        )

        seq_hidden_state_shape = sac.seq_hidden_state_shape
        low_seq_hidden_state_shape = sac.low_seq_hidden_state_shape

        step = 0
        while step < 10:
            if param_dict['seq_encoder'] == SEQ_ENCODER.RNN:
                sac.choose_rnn_action(**gen_batch_oc_obs_for_rnn(OBS_SHAPES,
                                                                 d_action_sizes=param_dict['d_action_sizes'],
                                                                 c_action_size=param_dict['c_action_size'],
                                                                 seq_hidden_state_shape=seq_hidden_state_shape,
                                                                 low_seq_hidden_state_shape=low_seq_hidden_state_shape))
            elif param_dict['seq_encoder'] == SEQ_ENCODER.ATTN:
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


def __gen_vanilla():
    param_dict_candidates = {
        'd_action_sizes': [[2, 3, 4]],
        'c_action_size': [4],
        'use_replay_buffer': [True, False],
        'use_priority': [True, False],
        'n_step': [3],
        'discrete_dqn_like': [True, False],
        'siamese': [None, 'ATC', 'BYOL'],
        'siamese_use_q': [False, True],
        'use_rnd': [True, False],
        'siamese_use_adaptive': [False, True],
        'use_add_with_td': [False, True],
        'action_noise': [None, [0.1, 0.1]]
    }
    possible_param_dicts = get_product(param_dict_candidates)

    i = 0
    for param_dict in possible_param_dicts:
        if not param_dict['d_action_sizes'] and not param_dict['c_action_size']:
            continue

        if not param_dict['siamese']:
            if param_dict['siamese_use_q'] or param_dict['siamese_use_adaptive']:
                continue

        func_name = f'test_{i:03d}'
        for k, v in param_dict.items():
            v = str(v).replace('.', '_')
            if len(param_dict_candidates[k]) > 1:
                func_name += f', {k}={v}'

        setattr(TestOCVanillaModel, func_name,
                TestOCVanillaModel.gen(param_dict))

        i += 1


def __gen_seq_encoder():
    param_dict_candidates = {
        'd_action_sizes': [[2, 3, 4]],
        'c_action_size': [4],
        'use_replay_buffer': [True, False],
        'burn_in_step': [5],
        'option_burn_in_step': [-1, 2],
        'n_step': [3],
        'discrete_dqn_like': [True, False],
        'seq_encoder': ['RNN', 'ATTN'],
        'use_prediction': [True, False],
        'use_extra_data': [True, False],
        'siamese': [None, 'ATC', 'BYOL'],
        'siamese_use_q': [False, True],
        'siamese_use_adaptive': [False, True],
        'use_add_with_td': [False, True]
    }

    possible_param_dicts = get_product(param_dict_candidates)

    i = 0
    for param_dict in possible_param_dicts:
        if not param_dict['d_action_sizes'] and not param_dict['c_action_size']:
            continue

        if not param_dict['use_prediction']:
            if param_dict['use_extra_data']:
                continue

        if not param_dict['siamese']:
            if param_dict['siamese_use_q'] or param_dict['siamese_use_adaptive']:
                continue

        func_name = f'test_{i:03d}'
        for k, v in param_dict.items():
            v = str(v).replace('.', '_')
            if len(param_dict_candidates[k]) > 1:
                func_name += f', {k}={v}'

        setattr(TestOCSeqEncoderModel, func_name,
                TestOCSeqEncoderModel.gen(param_dict))

        i += 1


__gen_vanilla()
__gen_seq_encoder()
