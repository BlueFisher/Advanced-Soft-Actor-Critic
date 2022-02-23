import unittest

from algorithm.sac_base import SAC_Base
from algorithm.utils.enums import *
from tests.get_synthesis_data import *

OBS_SHAPES = [(10,), (30, 30, 3)]


class TestVanillaModel(unittest.TestCase):
    def _test(self, param_dict):
        import tests.nn_conv_vanilla as nn_conv

        sac = SAC_Base(
            obs_shapes=OBS_SHAPES,
            model_abs_dir=None,
            model=nn_conv,
            **param_dict
        )

        step = 0
        while step < 10:
            sac.choose_action(*gen_batch_obs(OBS_SHAPES))
            sac.fill_replay_buffer(*gen_episode_trans(OBS_SHAPES,
                                                      d_action_size=param_dict['d_action_size'],
                                                      c_action_size=param_dict['c_action_size'],
                                                      episode_len=10))
            step = sac.train()

    @staticmethod
    def gen(param_dict):
        def func(self):
            self._test(param_dict)
        return func


class TestSeqEncoderModel(unittest.TestCase):
    def _test(self, param_dict):
        if param_dict['seq_encoder'] == SEQ_ENCODER.RNN:
            import tests.nn_conv_rnn as nn_conv
        elif param_dict['seq_encoder'] == SEQ_ENCODER.ATTN:
            import tests.nn_conv_attn as nn_conv

        sac = SAC_Base(
            obs_shapes=OBS_SHAPES,
            model_abs_dir=None,
            model=nn_conv,
            **param_dict
        )

        seq_hidden_state_shape = sac.seq_hidden_state_shape

        step = 0
        while step < 10:
            if param_dict['seq_encoder'] == SEQ_ENCODER.RNN:
                sac.choose_rnn_action(*gen_batch_obs_for_rnn(OBS_SHAPES,
                                                             d_action_size=param_dict['d_action_size'],
                                                             c_action_size=param_dict['c_action_size'],
                                                             seq_hidden_state_shape=seq_hidden_state_shape))
            elif param_dict['seq_encoder'] == SEQ_ENCODER.ATTN:
                sac.choose_attn_action(*gen_batch_obs_for_attn(OBS_SHAPES,
                                                               d_action_size=param_dict['d_action_size'],
                                                               c_action_size=param_dict['c_action_size'],
                                                               seq_hidden_state_shape=seq_hidden_state_shape))
            sac.fill_replay_buffer(*gen_episode_trans(OBS_SHAPES,
                                                      d_action_size=param_dict['d_action_size'],
                                                      c_action_size=param_dict['c_action_size'],
                                                      seq_hidden_state_shape=seq_hidden_state_shape,
                                                      episode_len=40))
            step = sac.train()

    @staticmethod
    def gen(param_dict):
        def func(self):
            self._test(param_dict)
        return func


def __gen_vanilla():
    param_dict_candidates = {
        'd_action_size': [10],
        'c_action_size': [4],
        'n_step': [3],
        'siamese': [None, SIAMESE.ATC, SIAMESE.BYOL],
        'siamese_use_q': [False, True],
        'siamese_use_adaptive': [False, True],
        'use_add_with_td': [False, True]
    }
    possible_param_dicts = get_product(param_dict_candidates)

    i = 0
    for param_dict in possible_param_dicts:
        if param_dict['d_action_size'] == 0 and param_dict['c_action_size'] == 0:
            continue

        func_name = f'test_{i:03d}'
        for k, v in param_dict.items():
            if len(param_dict_candidates[k]) > 1:
                func_name += f', {k}={v}'

        setattr(TestVanillaModel, func_name,
                TestVanillaModel.gen(param_dict))

        i += 1


def __gen_seq_encoder():
    param_dict_candidates = {
        'd_action_size': [10],
        'c_action_size': [4],
        'burn_in_step': [5],
        'n_step': [3],
        'seq_encoder': [SEQ_ENCODER.RNN, SEQ_ENCODER.ATTN],
        'use_prediction': [True, False],
        'use_extra_data': [True, False],
        'siamese': [None, SIAMESE.ATC, SIAMESE.BYOL],
        'siamese_use_q': [False, True],
        'siamese_use_adaptive': [False, True],
        'use_add_with_td': [False, True]
    }

    possible_param_dicts = get_product(param_dict_candidates)

    i = 0
    for param_dict in possible_param_dicts:
        if param_dict['d_action_size'] == 0 and param_dict['c_action_size'] == 0:
            continue

        func_name = f'test_{i:03d}'
        for k, v in param_dict.items():
            if len(param_dict_candidates[k]) > 1:
                func_name += f', {k}={v}'

        setattr(TestSeqEncoderModel, func_name,
                TestSeqEncoderModel.gen(param_dict))

        i += 1


__gen_vanilla()
__gen_seq_encoder()
