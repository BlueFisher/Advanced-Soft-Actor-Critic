import logging
import math
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import autograd, distributions, nn, optim
from torch.nn import functional
from torch.utils.tensorboard import SummaryWriter

from .batch_buffer import BatchBuffer
from .nn_models import *
from .replay_buffer import PrioritizedReplayBuffer
from .utils import *


class SAC_Base(object):
    def __init__(self,
                 obs_shapes: List[Tuple],
                 d_action_size: int,
                 c_action_size: int,
                 model_abs_dir: Optional[Path],
                 device: Optional[str] = None,
                 ma_name: Optional[str] = None,
                 summary_path: str = 'log',
                 train_mode: bool = True,
                 last_ckpt: Optional[str] = None,

                 nn_config: Optional[dict] = None,

                 nn=None,
                 seed: Optional[float] = None,
                 write_summary_per_step: float = 1e3,
                 save_model_per_step: float = 1e5,

                 use_replay_buffer: bool = True,
                 use_priority: bool = True,

                 ensemble_q_num: int = 2,
                 ensemble_q_sample: int = 2,

                 burn_in_step: int = 0,
                 n_step: int = 1,
                 seq_encoder: Optional[SEQ_ENCODER] = None,

                 batch_size: int = 256,
                 tau: float = 0.005,
                 update_target_per_step: int = 1,
                 init_log_alpha: float = -2.3,
                 use_auto_alpha: bool = True,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 v_lambda: float = 1.,
                 v_rho: float = 1.,
                 v_c: float = 1.,
                 clip_epsilon: float = 0.2,

                 discrete_dqn_like: bool = False,
                 use_n_step_is: bool = True,
                 siamese: Optional[SIAMESE] = None,
                 siamese_use_q: bool = False,
                 siamese_use_adaptive: bool = False,
                 use_prediction: bool = False,
                 transition_kl: float = 0.8,
                 use_extra_data: bool = True,
                 curiosity: Optional[CURIOSITY] = None,
                 curiosity_strength: float = 1.,
                 use_rnd: bool = False,
                 rnd_n_sample: int = 10,
                 use_normalization: bool = False,
                 use_add_with_td: bool = False,
                 action_noise: Optional[List[float]] = None,

                 replay_config: Optional[dict] = None):
        """
        obs_shapes: List of dimensions of observations
        d_action_size: Dimension of discrete actions
        c_action_size: Dimension of continuous actions
        model_abs_dir: The directory that saves summary, checkpoints, config etc.
        device: Training in CPU or GPU
        ma_name: Multi-agent name
        train_mode: Is training or inference
        last_ckpt: The checkpoint to restore

        nn_config: nn model config

        nn: nn # Neural network models file
        seed: null # Random seed
        write_summary_per_step: 1000 # Write summaries in TensorBoard every N steps
        save_model_per_step: 5000 # Save model every N steps

        use_replay_buffer: true # Whether using prioritized replay buffer
        use_priority: true # Whether using PER importance ratio

        ensemble_q_num: 2 # Number of Qs
        ensemble_q_sample: 2 # Number of min Qs

        burn_in_step: 0 # Burn-in steps in R2D2
        n_step: 1 # Update Q function by N-steps
        seq_encoder: null # RNN | ATTN

        batch_size: 256 # Batch size for training

        tau: 0.005 # Coefficient of updating target network
        update_target_per_step: 1 # Update target network every N steps

        init_log_alpha: -2.3 # The initial log_alpha
        use_auto_alpha: true # Whether using automating entropy adjustment

        learning_rate: 0.0003 # Learning rate of all optimizers

        gamma: 0.99 # Discount factor
        v_lambda: 1.0 # Discount factor for V-trace
        v_rho: 1.0 # Rho for V-trace
        v_c: 1.0 # C for V-trace
        clip_epsilon: 0.2 # Epsilon for q clip

        discrete_dqn_like: false # Whether using policy or only Q network if discrete is in action spaces
        use_n_step_is: true # Whether using importance sampling
        siamese: null # ATC | BYOL
        siamese_use_q: false # Whether using contrastive q
        siamese_use_adaptive: false # Whether using adaptive weights
        use_prediction: false # Whether training a transition model
        transition_kl: 0.8 # The coefficient of KL of transition and standard normal
        use_extra_data: true # Whether using extra data to train prediction model
        curiosity: null # FORWARD | INVERSE
        curiosity_strength: 1 # Curiosity strength if using curiosity
        use_rnd: false # Whether using RND
        rnd_n_sample: 10 # RND sample times
        use_normalization: false # Whether using observation normalization
        use_add_with_td: false # Whether add transitions in replay buffer with td-error
        action_noise: null # [noise_min, noise_max]
        """
        self.obs_shapes = obs_shapes
        self.d_action_size = d_action_size
        self.c_action_size = c_action_size
        self.model_abs_dir = model_abs_dir
        self.train_mode = train_mode

        self.use_replay_buffer = use_replay_buffer
        self.use_priority = use_priority

        self.ensemble_q_num = ensemble_q_num
        self.ensemble_q_sample = ensemble_q_sample

        self.burn_in_step = burn_in_step
        self.n_step = n_step
        self.seq_encoder = seq_encoder

        self.write_summary_per_step = int(write_summary_per_step)
        self.save_model_per_step = int(save_model_per_step)
        self.batch_size = batch_size
        self.tau = tau
        self.update_target_per_step = update_target_per_step
        self.use_auto_alpha = use_auto_alpha
        self.gamma = gamma
        self.v_lambda = v_lambda
        self.v_rho = v_rho
        self.v_c = v_c
        self.clip_epsilon = clip_epsilon

        self.discrete_dqn_like = discrete_dqn_like
        self.use_n_step_is = use_n_step_is
        self.siamese = siamese
        self.siamese_use_q = siamese_use_q
        self.siamese_use_adaptive = siamese_use_adaptive
        self.use_prediction = use_prediction
        self.transition_kl = transition_kl
        self.use_extra_data = use_extra_data
        self.curiosity = curiosity
        self.curiosity_strength = curiosity_strength
        self.use_rnd = use_rnd
        self.rnd_n_sample = rnd_n_sample
        self.use_normalization = use_normalization
        self.use_add_with_td = use_add_with_td
        self.action_noise = action_noise

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        self.summary_writer = None
        if self.model_abs_dir and self.train_mode:
            summary_path = Path(self.model_abs_dir).joinpath(summary_path)
            self.summary_writer = SummaryWriter(str(summary_path))
            self.summary_available = True

        if self.train_mode:
            if self.use_replay_buffer:
                replay_config = {} if replay_config is None else replay_config
                self.replay_buffer = PrioritizedReplayBuffer(batch_size=batch_size, **replay_config)
            else:
                self.batch_buffer = BatchBuffer(self.burn_in_step,
                                                self.n_step,
                                                self.batch_size)

        if ma_name is None:
            self._logger = logging.getLogger('sac.base')
        else:
            self._logger = logging.getLogger(f'sac.base.{ma_name}')

        self._build_model(nn, nn_config, init_log_alpha, learning_rate)
        self._init_or_restore(int(last_ckpt) if last_ckpt is not None else None)

    def _build_model(self, nn, nn_config: Optional[dict], init_log_alpha: float, learning_rate: float):
        """
        Initialize variables, network models and optimizers
        """
        if nn_config is None:
            nn_config = {}
        nn_config = defaultdict(dict, nn_config)
        if nn_config['rep'] is None:
            nn_config['rep'] = {}
        if nn_config['policy'] is None:
            nn_config['policy'] = {}

        self.global_step = torch.tensor(0, dtype=torch.int64, requires_grad=False, device='cpu')

        self._gamma_ratio = torch.logspace(0, self.n_step - 1, self.n_step, self.gamma, device=self.device)
        self._lambda_ratio = torch.logspace(0, self.n_step - 1, self.n_step, self.v_lambda, device=self.device)

        def adam_optimizer(params):
            return optim.Adam(params, lr=learning_rate)

        """ NORMALIZATION """
        if self.use_normalization:
            self.normalizer_step = torch.tensor(0, dtype=torch.int32, device=self.device, requires_grad=False)
            self.running_means = []
            self.running_variances = []
            for shape in self.obs_shapes:
                self.running_means.append(torch.zeros(shape, device=self.device))
                self.running_variances.append(torch.ones(shape, device=self.device))

            p_self = self

            class ModelRep(nn.ModelRep):
                def forward(self, obs_list, *args, **kwargs):
                    obs_list = [
                        torch.clamp(
                            (obs - mean) / torch.sqrt(variance / (p_self.normalizer_step + 1)),
                            -5, 5
                        ) for obs, mean, variance in zip(obs_list,
                                                         p_self.running_means,
                                                         p_self.running_variances)
                    ]

                    return super().forward(obs_list, *args, **kwargs)
        else:
            ModelRep = nn.ModelRep

        """ REPRESENTATION """
        if self.seq_encoder == SEQ_ENCODER.RNN:
            self.model_rep: ModelBaseRNNRep = ModelRep(self.obs_shapes,
                                                       self.d_action_size, self.c_action_size,
                                                       False, self.train_mode,
                                                       self.model_abs_dir,
                                                       **nn_config['rep']).to(self.device)
            self.model_target_rep: ModelBaseRNNRep = ModelRep(self.obs_shapes,
                                                              self.d_action_size, self.c_action_size,
                                                              True, self.train_mode,
                                                              self.model_abs_dir,
                                                              **nn_config['rep']).to(self.device)
            # Get represented state and seq_hidden_state_shape
            test_obs_list = [torch.rand(self.batch_size, 1, *obs_shape, device=self.device) for obs_shape in self.obs_shapes]
            test_pre_action = torch.rand(self.batch_size, 1, self.d_action_size + self.c_action_size, device=self.device)
            test_state, test_rnn_state = self.model_rep(test_obs_list,
                                                        test_pre_action)
            state_size, self.seq_hidden_state_shape = test_state.shape[-1], test_rnn_state.shape[1:]

        elif self.seq_encoder == SEQ_ENCODER.ATTN:
            self.model_rep: ModelBaseAttentionRep = ModelRep(self.obs_shapes,
                                                             self.d_action_size, self.c_action_size,
                                                             False, self.train_mode,
                                                             self.model_abs_dir,
                                                             **nn_config['rep']).to(self.device)
            self.model_target_rep: ModelBaseAttentionRep = ModelRep(self.obs_shapes,
                                                                    self.d_action_size, self.c_action_size,
                                                                    True, self.train_mode,
                                                                    self.model_abs_dir,
                                                                    **nn_config['rep']).to(self.device)
            # Get represented state and seq_hidden_state_shape
            test_index = torch.zeros((self.batch_size, 1), dtype=torch.int32, device=self.device)
            test_obs_list = [torch.rand(self.batch_size, 1, *obs_shape, device=self.device) for obs_shape in self.obs_shapes]
            test_pre_action = torch.rand(self.batch_size, 1, self.d_action_size + self.c_action_size, device=self.device)
            test_state, test_attn_state, _ = self.model_rep(test_index,
                                                            test_obs_list,
                                                            test_pre_action)
            state_size, self.seq_hidden_state_shape = test_state.shape[-1], test_attn_state.shape[2:]

        else:
            self.model_rep: ModelBaseSimpleRep = ModelRep(self.obs_shapes,
                                                          False, self.train_mode,
                                                          self.model_abs_dir,
                                                          **nn_config['rep']).to(self.device)
            self.model_target_rep: ModelBaseSimpleRep = ModelRep(self.obs_shapes,
                                                                 True, self.train_mode,
                                                                 self.model_abs_dir,
                                                                 **nn_config['rep']).to(self.device)
            # Get represented state dimension
            test_obs_list = [torch.rand(self.batch_size, *obs_shape, device=self.device) for obs_shape in self.obs_shapes]
            test_state = self.model_rep(test_obs_list)
            state_size = test_state.shape[-1]

        for param in self.model_target_rep.parameters():
            param.requires_grad = False

        self._logger.info(f'State size: {state_size}')

        if len(list(self.model_rep.parameters())) > 0:
            self.optimizer_rep = adam_optimizer(self.model_rep.parameters())
        else:
            self.optimizer_rep = None

        """ Q """
        self.model_q_list: List[ModelBaseQ] = [nn.ModelQ(state_size,
                                                         self.d_action_size,
                                                         self.c_action_size,
                                                         False,
                                                         self.train_mode,
                                                         self.model_abs_dir).to(self.device)
                                               for _ in range(self.ensemble_q_num)]

        self.model_target_q_list: List[ModelBaseQ] = [nn.ModelQ(state_size,
                                                                self.d_action_size,
                                                                self.c_action_size,
                                                                True,
                                                                self.train_mode,
                                                                self.model_abs_dir).to(self.device)
                                                      for _ in range(self.ensemble_q_num)]
        for model_target_q in self.model_target_q_list:
            for param in model_target_q.parameters():
                param.requires_grad = False

        self.optimizer_q_list = [adam_optimizer(self.model_q_list[i].parameters()) for i in range(self.ensemble_q_num)]

        """ POLICY """
        self.model_policy: ModelBasePolicy = nn.ModelPolicy(state_size, self.d_action_size, self.c_action_size,
                                                            self.train_mode,
                                                            self.model_abs_dir,
                                                            **nn_config['policy']).to(self.device)
        self.optimizer_policy = adam_optimizer(self.model_policy.parameters())

        """ SIAMESE REPRESENTATION LEARNING """
        if self.siamese in (SIAMESE.ATC, SIAMESE.BYOL):
            test_encoder_list = self.model_rep.get_augmented_encoders(test_obs_list)
            if not isinstance(test_encoder_list, tuple):
                test_encoder_list = [test_encoder_list, ]

            if self.siamese == SIAMESE.ATC:
                self.contrastive_weight_list = [torch.randn((test_encoder.shape[-1], test_encoder.shape[-1]),
                                                            requires_grad=True,
                                                            device=self.device) for test_encoder in test_encoder_list]
                self.optimizer_siamese = adam_optimizer(self.contrastive_weight_list)

            elif self.siamese == SIAMESE.BYOL:
                self.model_rep_projection_list: List[ModelBaseRepProjection] = [
                    nn.ModelRepProjection(test_encoder.shape[-1]).to(self.device) for test_encoder in test_encoder_list]
                self.model_target_rep_projection_list: List[ModelBaseRepProjection] = [
                    nn.ModelRepProjection(test_encoder.shape[-1]).to(self.device) for test_encoder in test_encoder_list]

                test_projection_list = [pro(test_encoder) for pro, test_encoder in zip(self.model_rep_projection_list, test_encoder_list)]
                self.model_rep_prediction_list: List[ModelBaseRepPrediction] = [
                    nn.ModelRepPrediction(test_projection.shape[-1]).to(self.device) for test_projection in test_projection_list]
                self.optimizer_siamese = adam_optimizer(chain(*[pro.parameters() for pro in self.model_rep_projection_list],
                                                              *[pre.parameters() for pre in self.model_rep_prediction_list]))

        """ RECURRENT PREDICTION MODELS """
        if self.use_prediction:
            self.model_transition: ModelBaseTransition = nn.ModelTransition(state_size,
                                                                            self.d_action_size,
                                                                            self.c_action_size,
                                                                            self.use_extra_data).to(self.device)
            self.model_reward: ModelBaseReward = nn.ModelReward(state_size).to(self.device)
            self.model_observation: ModelBaseObservation = nn.ModelObservation(state_size, self.obs_shapes,
                                                                               self.use_extra_data).to(self.device)

            self.optimizer_prediction = adam_optimizer(chain(self.model_transition.parameters(),
                                                             self.model_reward.parameters(),
                                                             self.model_observation.parameters()))

        """ ALPHA """
        self.log_d_alpha = torch.tensor(init_log_alpha, dtype=torch.float32, requires_grad=True, device=self.device)
        self.log_c_alpha = torch.tensor(init_log_alpha, dtype=torch.float32, requires_grad=True, device=self.device)

        if self.use_auto_alpha:
            self.optimizer_alpha = adam_optimizer([self.log_d_alpha, self.log_c_alpha])

        """ CURIOSITY """
        if self.curiosity == CURIOSITY.FORWARD:
            self.model_forward_dynamic: ModelBaseForwardDynamic = nn.ModelForwardDynamic(state_size,
                                                                                         self.d_action_size + self.c_action_size).to(self.device)
            self.optimizer_curiosity = adam_optimizer(self.model_forward_dynamic.parameters())

        elif self.curiosity == CURIOSITY.INVERSE:
            self.model_inverse_dynamic: ModelBaseInverseDynamic = nn.ModelInverseDynamic(state_size,
                                                                                         self.d_action_size + self.c_action_size).to(self.device)
            self.optimizer_curiosity = adam_optimizer(self.model_inverse_dynamic.parameters())

        """ RANDOM NETWORK DISTILLATION """
        if self.use_rnd:
            self.model_rnd: ModelBaseRND = nn.ModelRND(state_size, self.d_action_size, self.c_action_size).to(self.device)
            self.model_target_rnd: ModelBaseRND = nn.ModelRND(state_size, self.d_action_size, self.c_action_size).to(self.device)
            for param in self.model_target_rnd.parameters():
                param.requires_grad = False
            self.optimizer_rnd = adam_optimizer(self.model_rnd.parameters())

    def _init_or_restore(self, last_ckpt: int):
        """
        Initialize network weights from scratch or restore from model_abs_dir
        """
        self.ckpt_dict = ckpt_dict = {
            'global_step': self.global_step
        }

        """ NORMALIZATION & REPRESENTATION """
        if self.use_normalization:
            ckpt_dict['normalizer_step'] = self.normalizer_step
            for i, v in enumerate(self.running_means):
                ckpt_dict[f'running_means_{i}'] = v
            for i, v in enumerate(self.running_variances):
                ckpt_dict[f'running_variances_{i}'] = v

        if len(list(self.model_rep.parameters())) > 0:
            ckpt_dict['model_rep'] = self.model_rep
            ckpt_dict['model_target_rep'] = self.model_target_rep
            ckpt_dict['optimizer_rep'] = self.optimizer_rep

        """ Q """
        for i in range(self.ensemble_q_num):
            ckpt_dict[f'model_q_{i}'] = self.model_q_list[i]
            ckpt_dict[f'model_target_q_{i}'] = self.model_target_q_list[i]
            ckpt_dict[f'optimizer_q_{i}'] = self.optimizer_q_list[i]

        """ POLICY """
        ckpt_dict['model_policy'] = self.model_policy
        ckpt_dict['optimizer_policy'] = self.optimizer_policy

        """ SIAMESE REPRESENTATION LEARNING """
        if self.siamese == SIAMESE.ATC:
            for i, weight in enumerate(self.contrastive_weight_list):
                ckpt_dict[f'contrastive_weights_{i}'] = weight
        elif self.siamese == SIAMESE.BYOL:
            for i, model_rep_projection in enumerate(self.model_rep_projection_list):
                ckpt_dict[f'model_rep_projection_{i}'] = model_rep_projection
            for i, model_target_rep_projection in enumerate(self.model_target_rep_projection_list):
                ckpt_dict[f'model_target_rep_projection_{i}'] = model_target_rep_projection
            for i, model_rep_prediction in enumerate(self.model_rep_prediction_list):
                ckpt_dict[f'model_rep_prediction_{i}'] = model_rep_prediction
            ckpt_dict['optimizer_siamese'] = self.optimizer_siamese

        """ RECURRENT PREDICTION MODELS """
        if self.use_prediction:
            ckpt_dict['model_transition'] = self.model_transition
            ckpt_dict['model_reward'] = self.model_reward
            ckpt_dict['model_observation'] = self.model_observation
            ckpt_dict['optimizer_prediction'] = self.optimizer_prediction

        """ ALPHA """
        ckpt_dict['log_d_alpha'] = self.log_d_alpha
        ckpt_dict['log_c_alpha'] = self.log_c_alpha
        if self.use_auto_alpha:
            ckpt_dict['optimizer_alpha'] = self.optimizer_alpha

        """ CURIOSITY """
        if self.curiosity is not None:
            if self.curiosity == CURIOSITY.FORWARD:
                ckpt_dict['model_forward_dynamic'] = self.model_forward_dynamic
            elif self.curiosity == CURIOSITY.INVERSE:
                ckpt_dict['model_inverse_dynamic'] = self.model_inverse_dynamic
            ckpt_dict['optimizer_curiosity'] = self.optimizer_curiosity

        """ RANDOM NETWORK DISTILLATION """
        if self.use_rnd:
            ckpt_dict['model_rnd'] = self.model_rnd
            ckpt_dict['optimizer_rnd'] = self.optimizer_rnd

        self.ckpt_dir = None
        if self.model_abs_dir:
            self.ckpt_dir = ckpt_dir = self.model_abs_dir.joinpath('model')

            ckpts = []
            if ckpt_dir.exists():
                for ckpt_path in ckpt_dir.glob('*.pth'):
                    ckpts.append(int(ckpt_path.stem))
                ckpts.sort()
            else:
                ckpt_dir.mkdir()

            if ckpts:
                if last_ckpt is None:
                    last_ckpt = ckpts[-1]
                else:
                    assert last_ckpt in ckpts

                ckpt_restore_path = ckpt_dir.joinpath(f'{last_ckpt}.pth')
                ckpt_restore = torch.load(ckpt_restore_path, map_location=self.device)
                self.global_step = self.global_step.to('cpu')
                for name, model in ckpt_dict.items():
                    if name not in ckpt_restore:
                        self._logger.warning(f'{name} not in {last_ckpt}.pth')
                        continue

                    if isinstance(model, torch.Tensor):
                        model.data = ckpt_restore[name]
                    else:
                        try:
                            model.load_state_dict(ckpt_restore[name])
                        except RuntimeError as e:
                            self._logger.error(e)
                        if isinstance(model, nn.Module):
                            if self.train_mode:
                                model.train()
                            else:
                                model.eval()

                self._logger.info(f'Restored from {ckpt_restore_path}')

                if self.train_mode and self.use_replay_buffer:
                    self.replay_buffer.load(ckpt_dir, last_ckpt)

                    self._logger.info(f'Replay buffer restored')
            else:
                self._logger.info('Initializing from scratch')
                self._update_target_variables()

    def save_model(self, save_replay_buffer=False):
        if self.ckpt_dir:
            global_step = self.get_global_step()
            ckpt_path = self.ckpt_dir.joinpath(f'{global_step}.pth')

            torch.save({
                k: v if isinstance(v, torch.Tensor) else v.state_dict()
                for k, v in self.ckpt_dict.items()
            }, ckpt_path)
            self._logger.info(f"Model saved at {ckpt_path}")

            if self.use_replay_buffer and save_replay_buffer:
                self.replay_buffer.save(self.ckpt_dir, global_step)

    def write_constant_summaries(self, constant_summaries, iteration=None):
        """
        Write constant information from sac_main.py, such as reward, iteration, etc.
        """
        if self.summary_writer is not None:
            for s in constant_summaries:
                self.summary_writer.add_scalar(s['tag'], s['simple_value'],
                                               self.get_global_step() if iteration is None else iteration)

        self.summary_writer.flush()

    def _increase_global_step(self):
        self.global_step.add_(1)

        return self.global_step.item()

    def get_global_step(self):
        return self.global_step.item()

    def get_initial_action(self, batch_size):
        if self.d_action_size:
            d_action = np.eye(self.d_action_size)[np.random.rand(batch_size, self.d_action_size).argmax(axis=-1)].astype(np.float32)
        else:
            d_action = np.empty((batch_size, 0), dtype=np.float32)

        c_action = np.zeros([batch_size, self.c_action_size], dtype=np.float32)

        return np.concatenate([d_action, c_action], axis=-1)

    def get_initial_seq_hidden_state(self, batch_size, get_numpy=True):
        assert self.seq_encoder is not None

        if get_numpy:
            return np.zeros([batch_size, *self.seq_hidden_state_shape], dtype=np.float32)
        else:
            return torch.zeros([batch_size, *self.seq_hidden_state_shape], device=self.device)

    @torch.no_grad()
    def _update_target_variables(self, tau=1.):
        """
        Soft (momentum) update target networks (default hard)
        """
        target = self.model_target_rep.parameters()
        source = self.model_rep.parameters()

        for i in range(self.ensemble_q_num):
            target = chain(target, self.model_target_q_list[i].parameters())
            source = chain(source, self.model_q_list[i].parameters())

        if self.siamese == 'BYOL':
            target = chain(target, *[t_pro.parameters() for t_pro in self.model_target_rep_projection_list])
            source = chain(source, *[pro.parameters() for pro in self.model_rep_projection_list])

        for target_param, param in zip(target, source):
            target_param.data.copy_(
                target_param.data * (1. - tau) + param.data * tau
            )

    @torch.no_grad()
    def _udpate_normalizer(self, obs_list: List[torch.Tensor]):
        self.normalizer_step.add_(obs_list[0].shape[0])

        input_to_old_means = [obs_list[i] - self.running_means[i] for i in range(len(obs_list))]
        new_means = [self.running_means[i] + torch.sum(
            input_to_old_means[i] / self.normalizer_step, dim=0
        ) for i in range(len(obs_list))]

        input_to_new_means = [obs_list[i] - new_means[i] for i in range(len(obs_list))]
        new_variances = [self.running_variances[i] + torch.sum(
            input_to_new_means[i] * input_to_old_means[i], dim=0
        ) for i in range(len(obs_list))]

        for t_p, p in zip(self.running_means + self.running_variances, new_means + new_variances):
            t_p.copy_(p)

    @torch.no_grad()
    def get_l_probs(self,
                    l_indexes: torch.Tensor,
                    l_padding_masks: torch.Tensor,
                    l_obses_list: List[torch.Tensor],
                    l_actions: torch.Tensor,
                    f_seq_hidden_states: torch.Tensor = None):
        """
        Args:
            l_indexes: [Batch, l]
            l_padding_masks: [Batch, l]
            l_obses_list: list([Batch, l, *obs_shapes_i], ...)
            l_states: [Batch, l, state_size]
            l_actions: [Batch, l, action_size]
            f_seq_hidden_states: [Batch, 1, *seq_hidden_state_shape]

        Returns:
            l_probs: [Batch, l]
        """

        if self.seq_encoder == SEQ_ENCODER.RNN:
            l_states, _ = self.model_rep(l_obses_list,
                                         gen_pre_n_actions(l_actions),
                                         f_seq_hidden_states[:, 0])
        elif self.seq_encoder == SEQ_ENCODER.ATTN:
            l_states, *_ = self.model_rep(l_indexes,
                                          l_obses_list,
                                          gen_pre_n_actions(l_actions),
                                          query_length=l_indexes.shape[1],
                                          hidden_state=f_seq_hidden_states,
                                          is_prev_hidden_state=True,
                                          padding_mask=l_padding_masks)
        else:
            l_states = self.model_rep(l_obses_list)
        #  l_states: [Batch, l, state_size]

        d_policy, c_policy = self.model_policy(l_states, l_obses_list)

        policy_prob = torch.ones((l_states.shape[:2]), device=self.device)  # [Batch, l]

        if self.d_action_size:
            n_selected_d_actions = l_actions[..., :self.d_action_size]
            policy_prob *= torch.exp(d_policy.log_prob(n_selected_d_actions))   # [Batch, l]

        if self.c_action_size:
            l_selected_c_actions = l_actions[..., self.d_action_size:]
            c_policy_prob = squash_correction_prob(c_policy, torch.atanh(l_selected_c_actions))
            # [Batch, l, c_action_size]
            policy_prob *= torch.prod(c_policy_prob, dim=-1)  # [Batch, l]

        return policy_prob

    @torch.no_grad()
    def get_l_seq_hidden_states(self,
                                l_indexes: torch.Tensor,
                                l_padding_masks: torch.Tensor,
                                l_obses_list: List[torch.Tensor],
                                l_actions: torch.Tensor,
                                f_seq_hidden_states: torch.Tensor):
        """
        Args:
            l_indexes: [Batch, l]
            l_padding_masks: [Batch, l]
            l_obses_list: list([Batch, l, *obs_shapes_i], ...)
            l_actions: [Batch, l, action_size]
            f_seq_hidden_states: [Batch, 1, *seq_hidden_state_shape]

        Returns:
            l_seq_hidden_states: [Batch, l, *seq_hidden_state_shape]
        """

        if self.seq_encoder == SEQ_ENCODER.RNN:
            l_rnn_states = []
            l_actions = gen_pre_n_actions(l_actions)
            rnn_state = f_seq_hidden_states[:, 0]
            for i in range(l_obses_list[0].shape[1]):
                _, rnn_state = self.model_rep([o[:, i:i + 1, ...] for o in l_obses_list],
                                              l_actions[:, i:i + 1, ...],
                                              rnn_state)
                l_rnn_states.append(rnn_state)

            return torch.stack(l_rnn_states, dim=1)

        elif self.seq_encoder == SEQ_ENCODER.ATTN:
            _, l_attn_states, _ = self.model_rep(l_indexes,
                                                 l_obses_list,
                                                 gen_pre_n_actions(l_actions),
                                                 query_length=l_indexes.shape[1],
                                                 hidden_state=f_seq_hidden_states,
                                                 is_prev_hidden_state=True,
                                                 padding_mask=l_padding_masks)

            return l_attn_states

    @torch.no_grad()
    def get_dqn_like_d_y(self,
                         n_rewards: torch.Tensor,
                         n_dones: torch.Tensor,
                         stacked_next_q: torch.Tensor,
                         stacked_next_target_q: torch.Tensor):
        """
        Args:
            n_rewards: [Batch, n]
            n_dones: [Batch, n], dtype=torch.bool
            stacked_next_q: [ensemble_q_sample, Batch, n, d_action_size]
            stacked_next_target_q: [ensemble_q_sample, Batch, n, d_action_size]

        Returns:
            y: [Batch, 1]
        """

        stacked_next_q = stacked_next_q[..., -1, :]  # [ensemble_q_sample, Batch, d_action_size]
        stacked_next_target_q = stacked_next_target_q[..., -1, :]  # [ensemble_q_sample, Batch, d_action_size]

        done = n_dones[:, -1:]  # [Batch, 1]

        mask_stacked_q = functional.one_hot(torch.argmax(stacked_next_q, dim=-1),
                                            self.d_action_size)
        # [ensemble_q_sample, Batch, d_action_size]

        stacked_max_next_target_q = torch.sum(stacked_next_target_q * mask_stacked_q,
                                              dim=-1,
                                              keepdim=True)
        # [ensemble_q_sample, Batch, 1]

        next_q, _ = torch.min(stacked_max_next_target_q, dim=0)
        # [Batch, 1]

        g = torch.sum(self._gamma_ratio * n_rewards, dim=-1, keepdim=True)  # [Batch, 1]
        y = g + self.gamma**self.n_step * next_q * ~done  # [Batch, 1]

        return y

    @torch.no_grad()
    def _v_trace(self, n_rewards: torch.Tensor, n_dones: torch.Tensor,
                 n_mu_probs: torch.Tensor, n_pi_probs: torch.Tensor,
                 v: torch.Tensor, next_v: torch.Tensor):
        """
        Args:
            n_rewards: [Batch, n]
            n_dones: [Batch, n], dtype=torch.bool
            n_mu_probs: [Batch, n]
            n_pi_probs: [Batch, n]
            v: [Batch, n]
            next_v: [Batch, n]

        Returns:
            y: [Batch, 1]
        """

        td_error = n_rewards + self.gamma * ~n_dones * next_v - v  # [Batch, n]
        td_error = self._gamma_ratio * td_error

        if self.use_n_step_is:
            td_error = self._lambda_ratio * td_error

            n_step_is = n_pi_probs / n_mu_probs.clamp(min=1e-8)

            # \rho_t, t \in [s, s+n-1]
            rho = torch.minimum(n_step_is, torch.tensor(self.v_rho, device=self.device))  # [Batch, n]

            # \prod{c_i}, i \in [s, t-1]
            c = torch.minimum(n_step_is, torch.tensor(self.v_c, device=self.device))
            c = torch.cat([torch.ones((n_step_is.shape[0], 1), device=self.device), c[..., :-1]], dim=-1)
            c = torch.cumprod(c, dim=1)

            # \prod{c_i} * \rho_t * td_error
            td_error = c * rho * td_error

        # \sum{td_error}
        r = torch.sum(td_error, dim=1, keepdim=True)  # [Batch, 1]

        # V_s + \sum{td_error}
        y = v[:, 0:1] + r  # [Batch, 1]

        return y

    @torch.no_grad()
    def _get_y(self,
               n_obses_list: List[torch.Tensor],
               n_states: torch.Tensor,
               n_actions: torch.Tensor,
               n_rewards: torch.Tensor,
               next_obs_list: List[torch.Tensor],
               next_state: torch.Tensor,
               n_dones: torch.Tensor,
               n_mu_probs: torch.Tensor = None):
        """
        Args:
            n_obses_list: list([Batch, n, *obs_shapes_i], ...)
            n_states: [Batch, n, state_size]
            n_actions: [Batch, n, action_size]
            n_rewards: [Batch, n]
            state_: [Batch, state_size]
            n_dones: [Batch, n], dtype=torch.bool
            n_mu_probs: [Batch, n]

        Returns:
            y: [Batch, 1]
        """

        d_alpha = torch.exp(self.log_d_alpha)
        c_alpha = torch.exp(self.log_c_alpha)

        next_n_obses_list = [torch.cat([n_obses[:, 1:, ...], next_obs.unsqueeze(1)], dim=1)
                             for n_obses, next_obs in zip(n_obses_list, next_obs_list)]
        next_n_states = torch.cat([n_states[:, 1:, ...], next_state.unsqueeze(1)], dim=1)  # [Batch, n, state_size]

        d_policy, c_policy = self.model_policy(n_states, n_obses_list)
        next_d_policy, next_c_policy = self.model_policy(next_n_states, next_n_obses_list)

        if self.curiosity is not None:
            if self.curiosity == CURIOSITY.FORWARD:
                approx_next_n_states = self.model_forward_dynamic(n_states, n_actions)  # [Batch, n, state_size]
                in_n_rewards = torch.sum(torch.pow(approx_next_n_states - next_n_states, 2), dim=-1) * 0.5  # [Batch, n]

            elif self.curiosity == CURIOSITY.INVERSE:
                approx_n_actions = self.model_inverse_dynamic(n_states, next_n_states)  # [Batch, n, action_size]
                in_n_rewards = torch.sum(torch.pow(approx_n_actions - n_actions, 2), dim=-1) * 0.5  # [Batch, n]

            in_n_rewards = in_n_rewards * self.curiosity_strength  # [Batch, n]
            n_rewards += in_n_rewards  # [Batch, n]

        if self.c_action_size:
            n_c_actions_sampled = c_policy.rsample()  # [Batch, n, action_size]
            next_n_c_actions_sampled = next_c_policy.rsample()  # [Batch, n, action_size]
        else:
            n_c_actions_sampled = torch.empty(0, device=self.device)
            next_n_c_actions_sampled = torch.empty(0, device=self.device)

        # ([Batch, n, action_size], [Batch, n, 1])
        q_list = [q(n_states, torch.tanh(n_c_actions_sampled)) for q in self.model_target_q_list]
        next_q_list = [q(next_n_states, torch.tanh(next_n_c_actions_sampled)) for q in self.model_target_q_list]

        d_q_list = [q[0] for q in q_list]  # [Batch, n, action_size]
        c_q_list = [q[1] for q in q_list]  # [Batch, n, 1]

        next_d_q_list = [q[0] for q in next_q_list]  # [Batch, n, action_size]
        next_c_q_list = [q[1] for q in next_q_list]  # [Batch, n, 1]

        d_y, c_y = None, None

        if self.d_action_size:
            stacked_next_d_q = torch.stack(next_d_q_list)[torch.randperm(self.ensemble_q_num)[:self.ensemble_q_sample]]
            # [ensemble_q_num, Batch, n, d_action_size] -> [ensemble_q_sample, Batch, n, d_action_size]

            if self.discrete_dqn_like:
                next_d_eval_q_list = [q(next_n_states, torch.tanh(next_n_c_actions_sampled))[0] for q in self.model_q_list]
                stacked_next_d_eval_q = torch.stack(next_d_eval_q_list)[torch.randperm(self.ensemble_q_num)[:self.ensemble_q_sample]]
                # [ensemble_q_num, Batch, n, d_action_size] -> [ensemble_q_sample, Batch, n, d_action_size]

                d_y = self.get_dqn_like_d_y(n_rewards, n_dones,
                                            stacked_next_d_eval_q,
                                            stacked_next_d_q)
            else:
                stacked_d_q = torch.stack(d_q_list)[torch.randperm(self.ensemble_q_num)[:self.ensemble_q_sample]]
                # [ensemble_q_num, Batch, n, d_action_size] -> [ensemble_q_sample, Batch, n, d_action_size]

                min_q, _ = torch.min(stacked_d_q, dim=0)  # [Batch, n, d_action_size]
                min_next_q, _ = torch.min(stacked_next_d_q, dim=0)  # [Batch, n, d_action_size]

                probs = d_policy.probs  # [Batch, n, action_size]
                next_probs = next_d_policy.probs  # [Batch, n, action_size]
                clipped_probs = probs.clamp(min=1e-8)
                clipped_next_probs = next_probs.clamp(min=1e-8)
                tmp_v = min_q - d_alpha * torch.log(clipped_probs)  # [Batch, n, action_size]
                tmp_next_v = min_next_q - d_alpha * torch.log(clipped_next_probs)  # [Batch, n, action_size]

                v = torch.sum(probs * tmp_v, dim=-1)  # [Batch, n]
                next_v = torch.sum(next_probs * tmp_next_v, dim=-1)  # [Batch, n]

                if self.use_n_step_is:
                    n_d_actions = n_actions[..., :self.d_action_size]
                    n_pi_probs = torch.exp(d_policy.log_prob(n_d_actions))  # [Batch, n]

                d_y = self._v_trace(n_rewards, n_dones,
                                    n_mu_probs,
                                    n_pi_probs if self.use_n_step_is else None,
                                    v, next_v)

        if self.c_action_size:
            n_actions_log_prob = torch.sum(squash_correction_log_prob(c_policy, n_c_actions_sampled), dim=-1)  # [Batch, n]
            next_n_actions_log_prob = torch.sum(squash_correction_log_prob(next_c_policy, next_n_c_actions_sampled), dim=-1)

            stacked_c_q = torch.stack(c_q_list)[torch.randperm(self.ensemble_q_num)[:self.ensemble_q_sample]]
            # [ensemble_q_num, Batch, n, 1] -> [ensemble_q_sample, Batch, n, 1]
            stacked_next_c_q = torch.stack(next_c_q_list)[torch.randperm(self.ensemble_q_num)[:self.ensemble_q_sample]]
            # [ensemble_q_num, Batch, n, 1] -> [ensemble_q_sample, Batch, n, 1]

            min_q, _ = stacked_c_q.min(dim=0)
            min_q = min_q.squeeze(dim=-1)  # [Batch, n]
            min_next_q, _ = stacked_next_c_q.min(dim=0)
            min_next_q = min_next_q.squeeze(dim=-1)  # [Batch, n]

            v = min_q - c_alpha * n_actions_log_prob  # [Batch, n]
            next_v = min_next_q - c_alpha * next_n_actions_log_prob  # [Batch, n]

            # v = scale_inverse_h(v)
            # next_v = scale_inverse_h(next_v)

            if self.use_n_step_is:
                n_c_actions = n_actions[..., self.d_action_size:]
                n_pi_probs = squash_correction_prob(c_policy, torch.atanh(n_c_actions))
                # [Batch, n, action_size]
                n_pi_probs = n_pi_probs.prod(axis=-1)  # [Batch, n]

            c_y = self._v_trace(n_rewards, n_dones,
                                n_mu_probs,
                                n_pi_probs if self.use_n_step_is else None,
                                v, next_v)

        return d_y, c_y  # [Batch, 1]

    def _train_rep_q(self,
                     bn_indexes: torch.Tensor,
                     bn_padding_masks: torch.Tensor,
                     bn_obses_list: List[torch.Tensor],
                     bn_actions: torch.Tensor,
                     bn_rewards: torch.Tensor,
                     next_obs_list: List[torch.Tensor],
                     bn_dones: torch.Tensor,
                     bn_mu_probs: Optional[torch.Tensor] = None,
                     f_seq_hidden_states: Optional[torch.Tensor] = None,
                     priority_is: Optional[torch.Tensor] = None,):
        """
        Args:
            bn_indexes: [Batch, b + n]
            bn_padding_masks: [Batch, b + n]
            bn_obses_list: list([Batch, b + n, *obs_shapes_i], ...)
            bn_actions: [Batch, b + n, action_size]
            bn_rewards: [Batch, b + n]
            next_obs_list: list([Batch, *obs_shapes_i], ...)
            bn_dones: [Batch, b + n], dtype=torch.bool
            bn_mu_probs: [Batch, b + n]
            f_seq_hidden_states: [Batch, 1, *seq_hidden_state_shape]
            priority_is: [Batch, 1]
        """

        m_obses_list = [torch.cat([n_obses, next_obs.unsqueeze(1)], dim=1)
                        for n_obses, next_obs in zip(bn_obses_list, next_obs_list)]

        if self.seq_encoder is None:
            m_states = self.model_rep(m_obses_list)
            m_target_states = self.model_target_rep(m_obses_list)
        else:
            m_pre_actions = gen_pre_n_actions(bn_actions, keep_last_action=True)

            if self.seq_encoder == SEQ_ENCODER.RNN:
                rnn_state = f_seq_hidden_states[:, 0]
                m_states, _ = self.model_rep(m_obses_list,
                                             m_pre_actions,
                                             rnn_state)
                m_target_states, _ = self.model_target_rep(m_obses_list,
                                                           m_pre_actions,
                                                           rnn_state)

            elif self.seq_encoder == SEQ_ENCODER.ATTN:
                m_indexes = torch.concat([bn_indexes, bn_indexes[:, -1:] + 1], dim=1)
                m_padding_mask = torch.concat([bn_padding_masks,
                                               torch.zeros_like(bn_padding_masks[:, -1:], dtype=torch.bool)],
                                              dim=1)

                m_states, _, _ = self.model_rep(m_indexes,
                                                m_obses_list,
                                                m_pre_actions,
                                                query_length=bn_indexes.shape[1] + 1,
                                                hidden_state=f_seq_hidden_states,
                                                is_prev_hidden_state=True,
                                                padding_mask=m_padding_mask)
                m_target_states, _, _ = self.model_target_rep(m_indexes,
                                                              m_obses_list,
                                                              m_pre_actions,
                                                              query_length=bn_indexes.shape[1] + 1,
                                                              hidden_state=f_seq_hidden_states,
                                                              is_prev_hidden_state=True,
                                                              padding_mask=m_padding_mask)

        bn_states = m_states[:, :-1, ...]
        state = m_states[:, self.burn_in_step, ...]

        batch = state.shape[0]

        action = bn_actions[:, self.burn_in_step, ...]
        d_action = action[..., :self.d_action_size]
        c_action = action[..., self.d_action_size:]

        q_list = [q(state, c_action) for q in self.model_q_list]
        # ([Batch, action_size], [Batch, 1])
        d_q_list = [q[0] for q in q_list]  # [Batch, action_size]
        c_q_list = [q[1] for q in q_list]  # [Batch, 1]

        d_y, c_y = self._get_y([m_obses[:, self.burn_in_step:-1, ...] for m_obses in m_obses_list],
                               m_target_states[:, self.burn_in_step:-1, ...],
                               bn_actions[:, self.burn_in_step:, ...],
                               bn_rewards[:, self.burn_in_step:],
                               [m_obses[:, -1, ...] for m_obses in m_obses_list],
                               m_target_states[:, -1, ...],
                               bn_dones[:, self.burn_in_step:],
                               bn_mu_probs[:, self.burn_in_step:] if self.use_n_step_is else None)
        #  [Batch, 1], [Batch, 1]

        loss_q_list = [torch.zeros((batch, 1), device=self.device) for _ in range(self.ensemble_q_num)]
        loss_none_mse = nn.MSELoss(reduction='none')

        if self.d_action_size:
            for i in range(self.ensemble_q_num):
                q_single = torch.sum(d_action * d_q_list[i], dim=-1, keepdim=True)  # [Batch, 1]
                loss_q_list[i] += loss_none_mse(q_single, d_y)

        if self.c_action_size:
            if self.clip_epsilon > 0:
                target_c_q_list = [q(state, c_action)[1] for q in self.model_target_q_list]

                clipped_q_list = [target_c_q_list[i] + torch.clamp(
                    c_q_list[i] - target_c_q_list[i],
                    -self.clip_epsilon,
                    self.clip_epsilon,
                ) for i in range(self.ensemble_q_num)]

                loss_q_a_list = [loss_none_mse(clipped_q, c_y) for clipped_q in clipped_q_list]  # [Batch, 1]
                loss_q_b_list = [loss_none_mse(q, c_y) for q in c_q_list]  # [Batch, 1]

                for i in range(self.ensemble_q_num):
                    loss_q_list[i] += torch.maximum(loss_q_a_list[i], loss_q_b_list[i])  # [Batch, 1]
            else:
                for i in range(self.ensemble_q_num):
                    loss_q_list[i] += loss_none_mse(c_q_list[i], c_y)  # [Batch, 1]

        if self.use_replay_buffer and self.use_priority:
            loss_q_list = [loss_q * priority_is for loss_q in loss_q_list]

        loss_q_list = [torch.mean(loss) for loss in loss_q_list]

        if self.optimizer_rep:
            self.optimizer_rep.zero_grad()

        for loss_q, opt_q in zip(loss_q_list, self.optimizer_q_list):
            opt_q.zero_grad()
            loss_q.backward(retain_graph=True)

        grads_rep_main = [m.grad.detach() if m.grad is not None else None
                          for m in self.model_rep.parameters()]
        grads_q_main_list = [[m.grad.detach() if m.grad is not None else None for m in q.parameters()]
                             for q in self.model_q_list]

        """ Siamese Representation Learning """
        loss_siamese, loss_siamese_q = None, None
        if self.siamese is not None:
            loss_siamese, loss_siamese_q = self._train_siamese_representation_learning(
                grads_rep_main,
                grads_q_main_list,
                bn_indexes,
                bn_padding_masks,
                bn_obses_list,
                bn_actions)

        for opt_q in self.optimizer_q_list:
            opt_q.step()

        """ Recurrent Prediction Model """
        loss_predictions = None
        if self.use_prediction:
            loss_predictions = self._train_rpm(grads_rep_main,
                                               grads_q_main_list,
                                               m_obses_list,
                                               m_states,
                                               m_target_states,
                                               bn_actions,
                                               bn_rewards)

        if self.optimizer_rep:
            self.optimizer_rep.step()

        return loss_q_list[0], loss_siamese, loss_siamese_q, loss_predictions

    @torch.no_grad()
    def calculate_adaptive_weights(self,
                                   grads_main: List[torch.Tensor],
                                   loss_list: List[torch.Tensor],
                                   model: nn.Module):

        grads_aux_list = [autograd.grad(loss, model.parameters(),
                                        allow_unused=True,
                                        retain_graph=True)
                          for loss in loss_list]
        grads_aux_list = [[g_aux if g_aux is not None else torch.zeros_like(g_main)
                           for g_main, g_aux in zip(grads_main, grads_aux)]
                          for grads_aux in grads_aux_list]

        _grads_main = torch.cat([g.reshape(1, -1) for g in grads_main], dim=1)
        _grads_aux_list = [torch.cat([g.reshape(1, -1) for g in grads_aux], dim=1)
                           for grads_aux in grads_aux_list]

        cos_list = [functional.cosine_similarity(_grads_main, _grads_aux)
                    for _grads_aux in _grads_aux_list]
        cos_list = [torch.sign(cos).clamp(min=0) for cos in cos_list]

        for grads_aux, cos in zip(grads_aux_list, cos_list):
            for param, grad_aux in zip(model.parameters(), grads_aux):
                param.grad += cos * grad_aux

    def _train_siamese_representation_learning(self,
                                               grads_rep_main,
                                               grads_q_main_list,
                                               bn_indexes: torch.Tensor,
                                               bn_padding_masks: torch.Tensor,
                                               bn_obses_list: List[torch.Tensor],
                                               bn_actions: torch.Tensor):

        n_obses_list = [bn_obses[:, self.burn_in_step:, ...] for bn_obses in bn_obses_list]
        encoder_list = self.model_rep.get_augmented_encoders(n_obses_list)  # [Batch, n, f], ...
        target_encoder_list = self.model_target_rep.get_augmented_encoders(n_obses_list)  # [Batch, n, f], ...

        if not isinstance(encoder_list, tuple):
            encoder_list = (encoder_list, )
            target_encoder_list = (target_encoder_list, )

        batch, n, *_ = encoder_list[0].shape

        if self.siamese == SIAMESE.ATC:
            _encoder_list = [e.reshape(batch * n, -1) for e in encoder_list]  # [Batch * n, f], ...
            _target_encoder_list = [t_e.reshape(batch * n, -1) for t_e in target_encoder_list]  # [Batch * n, f], ...
            logits_list = [torch.mm(e, weight) for e, weight in zip(_encoder_list, self.contrastive_weight_list)]
            logits_list = [torch.mm(logits, t_e.t()) for logits, t_e in zip(logits_list, _target_encoder_list)]  # [Batch * n, Batch * n], ...
            if not hasattr(self, '_contrastive_labels'):
                self._contrastive_labels = torch.block_diag(*torch.ones(batch, n, n, device=self.device))

            loss_siamese_list = [functional.binary_cross_entropy_with_logits(logits, self._contrastive_labels)
                                 for logits in logits_list]

        elif self.siamese == SIAMESE.BYOL:
            _encoder_list = [e.reshape(batch * n, -1) for e in encoder_list]  # [Batch * n, f], ...
            projection_list = [pro(encoder) for pro, encoder in zip(self.model_rep_projection_list, _encoder_list)]
            prediction_list = [pre(projection) for pre, projection in zip(self.model_rep_prediction_list, projection_list)]
            _target_encoder_list = [t_e.reshape(batch * n, -1) for t_e in target_encoder_list]  # [Batch * n, f], ...
            t_projection_list = [t_pro(t_e) for t_pro, t_e in zip(self.model_target_rep_projection_list, _target_encoder_list)]

            loss_siamese_list = [functional.cosine_similarity(prediction, t_projection).mean()  # [Batch * n, ] -> [1, ]
                                 for prediction, t_projection in zip(prediction_list, t_projection_list)]

        if self.siamese_use_q:
            if self.seq_encoder is None:
                _obs = [n_obses[:, 0, ...] for n_obses in n_obses_list]

                _encoder = [e[:, 0, ...] for e in encoder_list]
                _target_encoder = [t_e[:, 0, ...] for t_e in target_encoder_list]

                state = self.model_rep.get_state_from_encoders(_obs,
                                                               _encoder if len(_encoder) > 1 else _encoder[0])
                target_state = self.model_target_rep.get_state_from_encoders(_obs,
                                                                             _target_encoder if len(_target_encoder) > 1 else _target_encoder[0])

            else:
                obses_list_at_n = [n_obses[:, 0:1, ...] for n_obses in n_obses_list]

                _encoder = [e[:, 0:1, ...] for e in encoder_list]
                _target_encoder = [t_e[:, 0:1, ...] for t_e in target_encoder_list]

                pre_actions_at_n = bn_actions[:, self.burn_in_step - 1:self.burn_in_step, ...]

                if self.seq_encoder == SEQ_ENCODER.RNN:
                    state = self.model_rep.get_state_from_encoders(obses_list_at_n,
                                                                   _encoder if len(_encoder) > 1 else _encoder[0],
                                                                   pre_actions_at_n,
                                                                   self.get_initial_seq_hidden_state(batch, False))
                    target_state = self.model_target_rep.get_state_from_encoders(obses_list_at_n,
                                                                                 _target_encoder if len(_target_encoder) > 1 else _target_encoder[0],
                                                                                 pre_actions_at_n,
                                                                                 self.get_initial_seq_hidden_state(batch, False))
                    state = state[:, 0, ...]
                    target_state = target_state[:, 0, ...]

                elif self.seq_encoder == SEQ_ENCODER.ATTN:
                    indexes_at_n = bn_indexes[:, self.burn_in_step:self.burn_in_step + 1]
                    padding_masks_at_n = bn_padding_masks[:, self.burn_in_step:self.burn_in_step + 1]
                    state = self.model_rep.get_state_from_encoders(indexes_at_n,
                                                                   obses_list_at_n,
                                                                   _encoder if len(_encoder) > 1 else _encoder[0],
                                                                   pre_actions_at_n,
                                                                   query_length=1,
                                                                   padding_mask=padding_masks_at_n)
                    target_state = self.model_target_rep.get_state_from_encoders(indexes_at_n,
                                                                                 obses_list_at_n,
                                                                                 _encoder if len(_encoder) > 1 else _encoder[0],
                                                                                 pre_actions_at_n,
                                                                                 query_length=1,
                                                                                 padding_mask=padding_masks_at_n)
                    state = state[:, 0, ...]
                    target_state = target_state[:, 0, ...]

            q_loss_list = []

            d_action = bn_actions[:, self.burn_in_step, :self.d_action_size]
            c_action = bn_actions[:, self.burn_in_step, self.d_action_size:]

            q_list = [q(state, c_action)
                      for q in self.model_q_list]  # [Batch, 1], ...
            target_q_list = [q(target_state, c_action)
                             for q in self.model_target_q_list]  # [Batch, 1], ...

            if self.d_action_size:
                q_single_list = [torch.sum(d_action * q[0], dim=-1)
                                 for q in q_list]
                # [Batch, d_action_size], ... -> [Batch, ], ...
                target_q_single_list = [torch.sum(d_action * t_q[0], dim=-1)
                                        for t_q in target_q_list]
                # [Batch, d_action_size], ... -> [Batch, ], ...

                q_loss_list += [functional.mse_loss(q, t_q)
                                for q, t_q in zip(q_single_list, target_q_single_list)]

            if self.c_action_size:
                c_q_list = [q[1] for q in q_list]  # [Batch, 1], ...
                target_c_q_list = [t_q[1] for t_q in target_q_list]  # [Batch, 1], ...

                q_loss_list += [functional.mse_loss(q, t_q)
                                for q, t_q in zip(c_q_list, target_c_q_list)]

            loss_list = loss_siamese_list + q_loss_list
        else:
            loss_list = loss_siamese_list

        if self.siamese_use_q:
            if self.siamese_use_adaptive:
                for grads_q_main, q_loss, q in zip(grads_q_main_list, q_loss_list, self.model_q_list):
                    self.calculate_adaptive_weights(grads_q_main, [q_loss], q)
            else:
                for q_loss, q in zip(q_loss_list, self.model_q_list):
                    q_loss.backward(inputs=list(q.parameters()), retain_graph=True)

        loss = sum(loss_list)

        if self.siamese_use_adaptive:
            self.calculate_adaptive_weights(grads_rep_main, loss_list, self.model_rep)
        else:
            loss.backward(inputs=list(self.model_rep.parameters()), retain_graph=True)

        self.optimizer_siamese.zero_grad()
        if self.siamese == SIAMESE.ATC:
            loss.backward(inputs=self.contrastive_weight_list, retain_graph=True)
        elif self.siamese == SIAMESE.BYOL:
            loss.backward(inputs=list(chain(*[pro.parameters() for pro in self.model_rep_projection_list],
                                            *[pre.parameters() for pre in self.model_rep_prediction_list])), retain_graph=True)
        self.optimizer_siamese.step()

        return sum(loss_siamese_list), sum(q_loss_list) if self.siamese_use_q else None

    def _train_rpm(self,
                   grads_rep_main,
                   m_obses_list,
                   m_states,
                   m_target_states,
                   bn_actions,
                   bn_rewards):
        bn_obses_list = [m_obs[:, :-1, ...] for m_obs in m_obses_list]
        bn_states = m_states[:, :-1, ...]

        approx_next_state_dist: torch.distributions.Normal = self.model_transition(
            [bn_obses[:, self.burn_in_step:, ...] for bn_obses in bn_obses_list],  # May for extra observations
            bn_states[:, self.burn_in_step:, ...],
            bn_actions[:, self.burn_in_step:, ...]
        )  # [Batch, n, action_size]

        loss_transition = -torch.mean(approx_next_state_dist.log_prob(m_target_states[:, self.burn_in_step + 1:, ...]))

        std_normal = distributions.Normal(torch.zeros_like(approx_next_state_dist.loc),
                                          torch.ones_like(approx_next_state_dist.scale))
        kl = distributions.kl.kl_divergence(approx_next_state_dist, std_normal)
        loss_transition += self.transition_kl * torch.mean(kl)

        approx_n_rewards = self.model_reward(m_states[:, self.burn_in_step + 1:, ...])  # [Batch, n, 1]
        loss_reward = functional.mse_loss(approx_n_rewards, torch.unsqueeze(bn_rewards[:, self.burn_in_step:], 2))
        loss_reward /= self.n_step

        loss_obs = self.model_observation.get_loss(m_states[:, self.burn_in_step:, ...],
                                                   [m_obses[:, self.burn_in_step:, ...] for m_obses in m_obses_list])
        loss_obs /= self.n_step

        self.calculate_adaptive_weights(grads_rep_main, [loss_transition, loss_reward, loss_obs], self.model_rep)

        loss_predictions = loss_transition + loss_reward + loss_obs
        self.optimizer_prediction.zero_grad()
        loss_predictions.backward(inputs=list(chain(self.model_transition.parameters(),
                                                    self.model_reward.parameters(),
                                                    self.model_observation.parameters())))
        self.optimizer_prediction.step()

        return torch.mean(approx_next_state_dist.entropy()), loss_reward, loss_obs

    def _train_policy(self,
                      obs_list: List[torch.Tensor],
                      state: torch.Tensor,
                      action: torch.Tensor):
        batch = state.shape[0]

        d_policy, c_policy = self.model_policy(state, obs_list)

        loss_d_policy = torch.zeros((batch, 1), device=self.device)
        loss_c_policy = torch.zeros((batch, 1), device=self.device)

        d_alpha = torch.exp(self.log_d_alpha)
        c_alpha = torch.exp(self.log_c_alpha)

        if self.d_action_size and not self.discrete_dqn_like:
            probs = d_policy.probs   # [Batch, action_size]
            clipped_probs = torch.maximum(probs, torch.tensor(1e-8, device=self.device))

            c_action = action[..., self.d_action_size:]

            q_list = [q(state, c_action) for q in self.model_q_list]
            # ([Batch, action_size], [Batch, 1])
            d_q_list = [q[0] for q in q_list]  # [Batch, action_size]

            stacked_d_q = torch.stack(d_q_list)[torch.randperm(self.ensemble_q_num)[:self.ensemble_q_sample]]
            # [ensemble_q_num, Batch, d_action_size] -> [ensemble_q_sample, Batch, d_action_size]
            min_d_q, _ = torch.min(stacked_d_q, dim=0)
            # [ensemble_q_sample, Batch, d_action_size] -> [Batch, d_action_size]

            _loss_policy = d_alpha.detach() * torch.log(clipped_probs) - min_d_q.detach()  # [Batch, d_action_size]
            loss_d_policy = torch.sum(probs * _loss_policy, dim=1, keepdim=True)  # [Batch, 1]

        if self.c_action_size:
            action_sampled = c_policy.rsample()
            c_q_for_gradient_list = [q(state, torch.tanh(action_sampled))[1] for q in self.model_q_list]
            # [[Batch, 1], ...]

            stacked_c_q_for_gradient = torch.stack(c_q_for_gradient_list)[torch.randperm(self.ensemble_q_num)[:self.ensemble_q_sample]]
            # [ensemble_q_num, Batch, 1] -> [ensemble_q_sample, Batch, 1]

            log_prob = torch.sum(squash_correction_log_prob(c_policy, action_sampled), dim=1, keepdim=True)
            # [Batch, 1]

            min_c_q_for_gradient, _ = torch.min(stacked_c_q_for_gradient, dim=0)
            # [ensemble_q_sample, Batch, 1] -> [Batch, 1]

            loss_c_policy = c_alpha.detach() * log_prob - min_c_q_for_gradient
            # [Batch, 1]

        loss_policy = torch.mean(loss_d_policy + loss_c_policy)

        if (self.d_action_size and not self.discrete_dqn_like) or self.c_action_size:
            self.optimizer_policy.zero_grad()
            loss_policy.backward(inputs=list(self.model_policy.parameters()))
            self.optimizer_policy.step()

        return (torch.mean(d_policy.entropy()) if self.d_action_size else None,
                torch.mean(c_policy.entropy()) if self.c_action_size else None)

    def _train_alpha(self,
                     obs_list: torch.Tensor,
                     state: torch.Tensor):
        batch = state.shape[0]

        d_policy, c_policy = self.model_policy(state, obs_list)

        d_alpha = torch.exp(self.log_d_alpha)
        c_alpha = torch.exp(self.log_c_alpha)

        loss_d_alpha = torch.zeros((batch, 1), device=self.device)
        loss_c_alpha = torch.zeros((batch, 1), device=self.device)

        if self.d_action_size and not self.discrete_dqn_like:
            probs = d_policy.probs   # [Batch, action_size]
            clipped_probs = probs.clamp(min=1e-8)

            _loss_alpha = -d_alpha * (torch.log(clipped_probs) - self.d_action_size)  # [Batch, action_size]
            loss_d_alpha = torch.sum(probs * _loss_alpha, dim=1, keepdim=True)  # [Batch, 1]

        if self.c_action_size:
            action_sampled = c_policy.sample()
            log_prob = torch.sum(squash_correction_log_prob(c_policy, action_sampled), dim=1, keepdim=True)
            # [Batch, 1]

            loss_c_alpha = -c_alpha * (log_prob - self.c_action_size)  # [Batch, 1]

        loss_alpha = torch.mean(loss_d_alpha + loss_c_alpha)

        self.optimizer_alpha.zero_grad()
        loss_alpha.backward(inputs=[self.log_d_alpha, self.log_c_alpha])
        self.optimizer_alpha.step()

        return d_alpha, c_alpha

    def _train_curiosity(self, m_states: torch.Tensor, bn_actions: torch.Tensor):
        n_states = m_states[:, self.burn_in_step:-1, ...]
        next_n_states = m_states[:, self.burn_in_step + 1:, ...]
        n_actions = bn_actions[:, self.burn_in_step:, ...]

        self.optimizer_curiosity.zero_grad()

        if self.curiosity == CURIOSITY.FORWARD:
            approx_next_n_states = self.model_forward_dynamic(n_states, n_actions)
            loss_curiosity = functional.mse_loss(approx_next_n_states, next_n_states)
            loss_curiosity.backward(inputs=list(self.model_forward_dynamic.parameters()))

        elif self.curiosity == CURIOSITY.INVERSE:
            approx_n_actions = self.model_inverse_dynamic(n_states, next_n_states)
            loss_curiosity = functional.mse_loss(approx_n_actions, n_actions)
            loss_curiosity.backward(inputs=list(self.model_inverse_dynamic.parameters()))

        self.optimizer_curiosity.step()

        return loss_curiosity

    def _train_rnd(self, bn_states: torch.Tensor, bn_actions: torch.Tensor):
        n_states = bn_states[:, self.burn_in_step:, ...]
        n_actions = bn_actions[:, self.burn_in_step:, ...]
        d_n_actions = n_actions[..., :self.d_action_size]  # [batch, n, d_action_size]
        c_n_actions = n_actions[..., self.d_action_size:]  # [batch, n, c_action_size]

        loss = torch.zeros((1, ), device=self.device)

        if self.d_action_size:
            d_rnd = self.model_rnd.cal_d_rnd(n_states)  # [batch, n, d_action_size, f]
            with torch.no_grad():
                t_d_rnd = self.model_target_rnd.cal_d_rnd(n_states)  # [batch, n, d_action_size, f]

            _i = functional.one_hot(d_n_actions.argmax(-1), self.d_action_size).unsqueeze(-1)  # [batch, n, d_action_size, 1]
            d_rnd = (torch.repeat_interleave(_i, d_rnd.size(-1), axis=-1) * d_rnd).sum(-2)
            # [batch, n, d_action_size, f] -> [batch, n, f]
            t_d_rnd = (torch.repeat_interleave(_i, t_d_rnd.size(-1), axis=-1) * t_d_rnd).sum(-2)
            # [batch, n, d_action_size, f] -> [batch, n, f]

            loss += functional.mse_loss(d_rnd, t_d_rnd)

        if self.c_action_size:
            c_rnd = self.model_rnd.cal_c_rnd(n_states, c_n_actions)  # [batch, n, f]
            with torch.no_grad():
                t_c_rnd = self.model_target_rnd.cal_c_rnd(n_states, n_actions)  # [batch, n, f]

            loss += functional.mse_loss(c_rnd, t_c_rnd)

        self.optimizer_rnd.zero_grad()
        loss.backward(inputs=list(self.model_rnd.parameters()))
        self.optimizer_rnd.step()

        return loss

    def _train(self,
               bn_indexes: torch.Tensor,
               bn_padding_masks: torch.Tensor,
               bn_obses_list: List[torch.Tensor],
               bn_actions: torch.Tensor,
               bn_rewards: torch.Tensor,
               next_obs_list: List[torch.Tensor],
               bn_dones: torch.Tensor,
               bn_mu_probs: torch.Tensor = None,
               f_seq_hidden_states: torch.Tensor = None,
               priority_is: torch.Tensor = None):
        """
        Args:
            bn_indexes: [Batch, b + n], dtype=torch.int32
            bn_padding_masks: [Batch, b + n], dtype=torch.bool
            bn_obses_list: list([Batch, b + n, *obs_shapes_i], ...)
            bn_actions: [Batch, b + n, action_size]
            bn_rewards: [Batch, b + n]
            next_obs_list: list([Batch, *obs_shapes_i], ...)
            bn_dones: [Batch, b + n], dtype=torch.bool
            bn_mu_probs: [Batch, b + n]
            f_seq_hidden_states: [Batch, 1, *seq_hidden_state_shape]
            priority_is: [Batch, 1]
        """

        if self.global_step % self.update_target_per_step == 0:
            self._update_target_variables(tau=self.tau)

        loss_q, loss_siamese, loss_siamese_q, loss_predictions = self._train_rep_q(bn_indexes,
                                                                                   bn_padding_masks,
                                                                                   bn_obses_list,
                                                                                   bn_actions,
                                                                                   bn_rewards,
                                                                                   next_obs_list,
                                                                                   bn_dones,
                                                                                   bn_mu_probs,
                                                                                   f_seq_hidden_states,
                                                                                   priority_is)

        with torch.no_grad():
            m_obses_list = [torch.cat([bn_obses, next_obs.unsqueeze(1)], dim=1)
                            for bn_obses, next_obs in zip(bn_obses_list, next_obs_list)]

            if self.seq_encoder == SEQ_ENCODER.RNN:
                m_states, _ = self.model_rep(m_obses_list,
                                             gen_pre_n_actions(bn_actions, keep_last_action=True),
                                             f_seq_hidden_states[:, 0])
            elif self.seq_encoder == SEQ_ENCODER.ATTN:
                m_states, *_ = self.model_rep(torch.concat([bn_indexes, bn_indexes[:, -1:] + 1], dim=1),
                                              m_obses_list,
                                              gen_pre_n_actions(bn_actions, keep_last_action=True),
                                              query_length=bn_indexes.shape[1] + 1,
                                              hidden_state=f_seq_hidden_states,
                                              is_prev_hidden_state=True,
                                              padding_mask=torch.concat([bn_padding_masks,
                                                                        torch.zeros_like(bn_padding_masks[:, -1:], dtype=torch.bool)], dim=1))
            else:
                m_states = self.model_rep(m_obses_list)

        obs_list = [m_obses[:, self.burn_in_step, ...] for m_obses in m_obses_list]
        state = m_states[:, self.burn_in_step, ...]
        action = bn_actions[:, self.burn_in_step, ...]

        d_policy_entropy, c_policy_entropy = self._train_policy(obs_list, state, action)

        if self.use_auto_alpha and ((self.d_action_size and not self.discrete_dqn_like) or self.c_action_size):
            d_alpha, c_alpha = self._train_alpha(obs_list, state)

        if self.curiosity is not None:
            loss_curiosity = self._train_curiosity(m_states, bn_actions)

        if self.use_rnd:
            bn_states = m_states[:, :-1, ...]
            loss_rnd = self._train_rnd(bn_states, bn_actions)

        if self.summary_writer is not None and self.global_step % self.write_summary_per_step == 0:
            self.summary_available = True

            with torch.no_grad():
                self.summary_writer.add_scalar('loss/q', loss_q, self.global_step)
                if self.d_action_size:
                    self.summary_writer.add_scalar('loss/d_entropy', d_policy_entropy, self.global_step)
                    if self.use_auto_alpha and not self.discrete_dqn_like:
                        self.summary_writer.add_scalar('loss/d_alpha', d_alpha, self.global_step)
                if self.c_action_size:
                    self.summary_writer.add_scalar('loss/c_entropy', c_policy_entropy, self.global_step)
                    if self.use_auto_alpha:
                        self.summary_writer.add_scalar('loss/c_alpha', c_alpha, self.global_step)

                if self.siamese is not None:
                    self.summary_writer.add_scalar('loss/siamese',
                                                   loss_siamese,
                                                   self.global_step)
                    if self.siamese_use_q:
                        self.summary_writer.add_scalar('loss/siamese_q',
                                                       loss_siamese_q,
                                                       self.global_step)

                if self.use_prediction:
                    approx_next_state_dist_entropy, loss_reward, loss_obs = loss_predictions
                    self.summary_writer.add_scalar('loss/transition',
                                                   approx_next_state_dist_entropy,
                                                   self.global_step)
                    self.summary_writer.add_scalar('loss/reward', loss_reward, self.global_step)
                    self.summary_writer.add_scalar('loss/observation', loss_obs, self.global_step)

                    approx_obs_list = self.model_observation(m_states[0:1, 0, ...])
                    if not isinstance(approx_obs_list, (list, tuple)):
                        approx_obs_list = [approx_obs_list]
                    for approx_obs in approx_obs_list:
                        if len(approx_obs.shape) > 3:
                            self.summary_writer.add_images('observation',
                                                           approx_obs.permute([0, 3, 1, 2]),
                                                           self.global_step)

                if self.curiosity is not None:
                    self.summary_writer.add_scalar('loss/curiosity', loss_curiosity, self.global_step)

                if self.use_rnd:
                    self.summary_writer.add_scalar('loss/rnd', loss_rnd, self.global_step)

            self.summary_writer.flush()

    @torch.no_grad()
    def rnd_sample_d_action(self, state: torch.Tensor,
                            d_policy: distributions.Categorical):
        """
        Sample action `self.rnd_n_sample` times, 
        choose the action that has the max (model_rnd(s, a) - model_target_rnd(s, a))**2

        Args:
            state: [Batch, state_size]
            d_policy: [Batch, d_action_size]

        Returns:
            d_action: [Batch, d_action_size]
        """
        batch = state.shape[0]
        n_sample = self.rnd_n_sample

        d_actions = d_policy.sample((n_sample,))  # [n_sample, batch, d_action_size]
        d_actions = d_actions.transpose(0, 1)  # [batch, n_sample, d_action_size]

        d_rnd = self.model_rnd.cal_d_rnd(state)  # [batch, d_action_size, f]
        t_d_rnd = self.model_target_rnd.cal_d_rnd(state)  # [batch, d_action_size, f]
        d_rnd = torch.repeat_interleave(torch.unsqueeze(d_rnd, 1), n_sample, dim=1)  # [batch, n_sample, d_action_size, f]
        t_d_rnd = torch.repeat_interleave(torch.unsqueeze(t_d_rnd, 1), n_sample, dim=1)  # [batch, n_sample, d_action_size, f]

        _i = functional.one_hot(d_actions.argmax(-1), self.d_action_size).unsqueeze(-1)  # [batch, n_sample, d_action_size, 1]
        d_rnd = (torch.repeat_interleave(_i, d_rnd.size(-1), axis=-1) * d_rnd).sum(-2)
        # [batch, n_sample, d_action_size, f] -> [batch, n_sample, f]
        t_d_rnd = (torch.repeat_interleave(_i, t_d_rnd.size(-1), axis=-1) * t_d_rnd).sum(-2)
        # [batch, n_sample, d_action_size, f] -> [batch, n_sample, f]

        d_loss = torch.sum(torch.pow(d_rnd - t_d_rnd, 2), dim=-1)  # [batch, n_sample]
        d_idx = torch.argmax(d_loss, dim=1)  # [batch, ]

        return d_actions[torch.arange(batch), d_idx]

    @torch.no_grad()
    def rnd_sample_c_action(self, state: torch.Tensor,
                            c_policy: distributions.Normal):
        """
        Sample action `self.rnd_n_sample` times, 
        choose the action that has the max (model_rnd(s, a) - model_target_rnd(s, a))**2

        Args:
            state: [Batch, state_size]
            c_policy: [Batch, c_action_size]

        Returns:
            c_action: [Batch, c_action_size]
        """
        batch = state.shape[0]
        n_sample = self.rnd_n_sample

        c_actions = torch.tanh(c_policy.sample((n_sample,)))  # [n_sample, batch, c_action_size]
        c_actions = c_actions.transpose(0, 1)  # [batch, n_sample, c_action_size]

        states = torch.repeat_interleave(torch.unsqueeze(state, 1), n_sample, dim=1)
        # [batch, state_size] -> [batch, 1, state_size] -> [batch, n_sample, state_size]
        c_rnd = self.model_rnd.cal_c_rnd(states, c_actions)  # [batch, n_sample, f]
        t_c_rnd = self.model_target_rnd.cal_c_rnd(states, c_actions)  # [batch, n_sample, f]

        c_loss = torch.sum(torch.pow(c_rnd - t_c_rnd, 2), dim=-1)  # [batch, n_sample]
        c_idx = torch.argmax(c_loss, dim=1)  # [batch, ]

        return c_actions[torch.arange(batch), c_idx]

    @torch.no_grad()
    def _random_action(self, d_action, c_action):
        if self.action_noise is None:
            return d_action, c_action

        batch = max(d_action.shape[0], c_action.shape[0])

        action_noise = torch.linspace(*self.action_noise, steps=batch, device=self.device)  # [Batch, ]

        if self.d_action_size:
            action_random = torch.eye(self.d_action_size, device=self.device)[torch.randint(0, self.d_action_size, size=(batch, ))]
            mask = torch.rand(batch, device=self.device) < action_noise
            d_action[mask] = action_random[mask]

        if self.c_action_size:
            c_action = torch.tanh(torch.atanh(c_action) + torch.randn(batch, self.c_action_size, device=self.device) * action_noise.unsqueeze(1))

        return d_action, c_action

    @torch.no_grad()
    def _choose_action(self,
                       obs_list: List[torch.Tensor],
                       state: torch.Tensor,
                       disable_sample: bool = False,
                       force_rnd_if_available: bool = False):
        """
        Args:
            state: [Batch, state_size]

        Returns:
            action: [Batch, d_action_size + c_action_size]
            prob: [Batch, ]
        """
        batch = state.shape[0]
        d_policy, c_policy = self.model_policy(state, obs_list)

        if self.d_action_size:
            if self.discrete_dqn_like:
                if torch.rand(1) < 0.2 and self.train_mode:
                    d_action = distributions.OneHotCategorical(
                        logits=torch.ones(batch, self.d_action_size)).sample().to(self.device)
                else:
                    d_q, _ = self.model_q_list[0](state, c_policy.sample() if self.c_action_size else None)
                    d_action = torch.argmax(d_q, dim=-1)
                    d_action = functional.one_hot(d_action, self.d_action_size)
            else:
                if disable_sample:
                    d_action = functional.one_hot(d_policy.logits.argmax(dim=-1),
                                                  self.d_action_size)
                elif self.use_rnd and (self.train_mode or force_rnd_if_available):
                    d_action = self.rnd_sample_d_action(state, d_policy)
                else:
                    d_action = d_policy.sample()
        else:
            d_action = torch.empty(0, device=self.device)

        if self.c_action_size:
            if disable_sample:
                c_action = torch.tanh(c_policy.mean)
            elif self.use_rnd and (self.train_mode or force_rnd_if_available):
                c_action = self.rnd_sample_c_action(state, c_policy)
            else:
                c_action = torch.tanh(c_policy.sample())
        else:
            c_action = torch.empty(0, device=self.device)

        d_action, c_action = self._random_action(d_action, c_action)

        policy_prob = torch.ones(state.shape[:1], device=self.device)  # [Batch, ]
        if self.d_action_size:
            policy_prob *= torch.exp(d_policy.log_prob(d_action))  # [Batch, ]
        if self.c_action_size:
            c_policy_prob = squash_correction_prob(c_policy, torch.atanh(c_action))
            # [Batch, action_size]
            policy_prob *= torch.prod(c_policy_prob, dim=-1)  # [Batch, ]

        return torch.cat([d_action, c_action], dim=-1), policy_prob

    @torch.no_grad()
    def choose_action(self,
                      obs_list: List[np.ndarray],
                      disable_sample: bool = False,
                      force_rnd_if_available: bool = False):
        """
        Args:
            obs_list: list([Batch, *obs_shapes_i], ...)

        Returns:
            action: [Batch, d_action_size + c_action_size] (numpy)
        """
        obs_list = [torch.from_numpy(obs).to(self.device) for obs in obs_list]
        state = self.model_rep(obs_list)

        action, prob = self._choose_action(obs_list, state, disable_sample, force_rnd_if_available)
        return action.detach().cpu().numpy(), prob.detach().cpu().numpy()

    @torch.no_grad()
    def choose_rnn_action(self,
                          obs_list: List[np.ndarray],
                          pre_action: np.ndarray,
                          rnn_state: np.ndarray,
                          disable_sample: bool = False,
                          force_rnd_if_available: bool = False):
        """
        Args:
            obs_list: list([Batch, *obs_shapes_i], ...)
            pre_action: [Batch, d_action_size + c_action_size]
            rnn_state: [Batch, *seq_hidden_state_shape]
        Returns:
            action: [Batch, d_action_size + c_action_size] (numpy)
            rnn_state: [Batch, *seq_hidden_state_shape] (numpy)
        """
        obs_list = [torch.from_numpy(obs).to(self.device) for obs in obs_list]
        pre_action = torch.from_numpy(pre_action).to(self.device)
        rnn_state = torch.from_numpy(rnn_state).to(self.device)

        obs_list = [obs.unsqueeze(1) for obs in obs_list]
        pre_action = pre_action.unsqueeze(1)
        state, next_rnn_state = self.model_rep(obs_list, pre_action, rnn_state)
        state = state.squeeze(1)
        obs_list = [obs.squeeze(1) for obs in obs_list]

        action, prob = self._choose_action(obs_list, state, disable_sample, force_rnd_if_available)

        return (action.detach().cpu().numpy(),
                prob.detach().cpu().numpy(),
                next_rnn_state.detach().cpu().numpy())

    @torch.no_grad()
    def choose_attn_action(self,
                           ep_indexes: np.ndarray,
                           ep_padding_masks: np.ndarray,
                           ep_obses_list: List[np.ndarray],
                           ep_pre_actions: np.ndarray,
                           ep_attn_hidden_states: np.ndarray,

                           disable_sample: bool = False,
                           force_rnd_if_available: bool = False):
        """
        Args:
            ep_indexes: [Batch, episode_len]
            ep_padding_masks: [Batch, episode_len]
            ep_obses_list: list([Batch, episode_len, *obs_shapes_i], ...)
            ep_pre_actions: [Batch, episode_len, d_action_size + c_action_size]
            ep_attn_hidden_states: [Batch, episode_len, *seq_hidden_state_shape]

        Returns:
            action: [Batch, d_action_size + c_action_size] (numpy)
            attn_hidden_state: [Batch, *rnn_state_shape] (numpy)
        """
        ep_indexes = torch.from_numpy(ep_indexes).to(self.device)
        ep_padding_masks = torch.from_numpy(ep_padding_masks).to(self.device)
        ep_obses_list = [torch.from_numpy(obs).to(self.device) for obs in ep_obses_list]
        ep_pre_actions = torch.from_numpy(ep_pre_actions).to(self.device)
        ep_attn_hidden_states = torch.from_numpy(ep_attn_hidden_states).to(self.device)

        state, next_attn_hidden_state, _ = self.model_rep(ep_indexes,
                                                          ep_obses_list, ep_pre_actions,
                                                          query_length=1,
                                                          hidden_state=ep_attn_hidden_states,
                                                          is_prev_hidden_state=False,
                                                          padding_mask=ep_padding_masks)
        state = state.squeeze(1)
        next_attn_hidden_state = next_attn_hidden_state.squeeze(1)

        action, prob = self._choose_action([o[:, -1] for o in ep_obses_list],
                                           state,
                                           disable_sample,
                                           force_rnd_if_available)

        return (action.detach().cpu().numpy(),
                prob.detach().cpu().numpy(),
                next_attn_hidden_state.detach().cpu().numpy())

    @torch.no_grad()
    def _get_td_error(self,
                      bn_indexes: torch.Tensor,
                      bn_padding_masks: torch.Tensor,
                      bn_obses_list: List[torch.Tensor],
                      bn_actions: torch.Tensor,
                      bn_rewards: torch.Tensor,
                      next_obs_list: List[torch.Tensor],
                      bn_dones: torch.Tensor,
                      bn_mu_probs: torch.Tensor = None,
                      f_seq_hidden_states: torch.Tensor = None):
        """
        Args:
            bn_indexes: [Batch, b + n]
            bn_padding_masks: [Batch, b + n]
            bn_obses_list: list([Batch, b + n, *obs_shapes_i], ...)
            bn_actions: [Batch, b + n, action_size]
            bn_rewards: [Batch, b + n]
            next_obs_list: list([Batch, *obs_shapes_i], ...)
            bn_dones: [Batch, b + n]
            bn_mu_probs: [Batch, b + n]
            f_seq_hidden_states: [Batch, 1, *seq_hidden_state_shape]

        Returns:
            The td-error of observations, [Batch, 1]
        """
        m_obses_list = [torch.cat([bn_obses, next_obs.unsqueeze(1)], dim=1)
                        for bn_obses, next_obs in zip(bn_obses_list, next_obs_list)]

        if self.seq_encoder == SEQ_ENCODER.RNN:
            rnn_state = f_seq_hidden_states[:, 0]
            tmp_states, _ = self.model_rep([m_obses[:, :self.burn_in_step + 1, ...] for m_obses in m_obses_list],
                                           gen_pre_n_actions(bn_actions[:, :self.burn_in_step + 1, ...]),
                                           rnn_state)
            state = tmp_states[:, self.burn_in_step, ...]
            m_target_states, *_ = self.model_target_rep(m_obses_list,
                                                        gen_pre_n_actions(bn_actions,
                                                                          keep_last_action=True),
                                                        rnn_state)
        elif self.seq_encoder == SEQ_ENCODER.ATTN:
            tmp_states, *_ = self.model_rep(bn_indexes[:, :self.burn_in_step + 1],
                                            [m_obses[:, :self.burn_in_step + 1, ...] for m_obses in m_obses_list],
                                            gen_pre_n_actions(bn_actions[:, :self.burn_in_step + 1, ...]),
                                            query_length=self.burn_in_step + 1,
                                            hidden_state=f_seq_hidden_states,
                                            is_prev_hidden_state=True,
                                            padding_mask=bn_padding_masks[:, :self.burn_in_step + 1])
            state = tmp_states[:, self.burn_in_step, ...]
            m_target_states, *_ = self.model_target_rep(torch.concat([bn_indexes, bn_indexes[:, -1:] + 1], dim=1),
                                                        m_obses_list,
                                                        gen_pre_n_actions(bn_actions,
                                                                          keep_last_action=True),
                                                        query_length=bn_indexes.shape[1] + 1,
                                                        hidden_state=f_seq_hidden_states,
                                                        is_prev_hidden_state=True,
                                                        padding_mask=torch.concat([bn_padding_masks,
                                                                                   torch.zeros_like(bn_padding_masks[:, -1:], dtype=torch.bool)], dim=1))
        else:
            state = self.model_rep([m_obses[:, self.burn_in_step, ...] for m_obses in m_obses_list])
            m_target_states = self.model_target_rep(m_obses_list)

        action = bn_actions[:, self.burn_in_step, ...]
        d_action = action[..., :self.d_action_size]
        c_action = action[..., self.d_action_size:]

        q_list = [q(state, c_action) for q in self.model_q_list]
        # ([Batch, action_size], [Batch, 1])
        d_q_list = [q[0] for q in q_list]  # [Batch, action_size]
        c_q_list = [q[1] for q in q_list]  # [Batch, 1]

        if self.d_action_size:
            d_q_list = [torch.sum(d_action * q, dim=-1, keepdim=True) for q in d_q_list]
            # [Batch, 1]

        d_y, c_y = self._get_y([m_obses[:, self.burn_in_step:-1, ...] for m_obses in m_obses_list],
                               m_target_states[:, self.burn_in_step:-1, ...],
                               bn_actions[:, self.burn_in_step:, ...],
                               bn_rewards[:, self.burn_in_step:],
                               [m_obses[:, -1, ...] for m_obses in m_obses_list],
                               m_target_states[:, -1, ...],
                               bn_dones[:, self.burn_in_step:],
                               bn_mu_probs[:, self.burn_in_step:] if self.use_n_step_is else None)
        # [Batch, 1]

        q_td_error_list = [torch.zeros((state.shape[0], 1), device=self.device) for _ in range(self.ensemble_q_num)]
        # [Batch, 1]
        if self.d_action_size:
            for i in range(self.ensemble_q_num):
                q_td_error_list[i] += torch.abs(d_q_list[i] - d_y)

        if self.c_action_size:
            for i in range(self.ensemble_q_num):
                q_td_error_list[i] += torch.abs(c_q_list[i] - c_y)

        td_error = torch.mean(torch.cat(q_td_error_list, dim=-1),
                              dim=-1, keepdim=True)
        return td_error

    def get_episode_td_error(self,
                             l_indexes: np.ndarray,
                             l_padding_masks: np.ndarray,
                             l_obses_list: List[np.ndarray],
                             l_actions: np.ndarray,
                             l_rewards: np.ndarray,
                             next_obs_list: List[np.ndarray],
                             l_dones: np.ndarray,
                             l_mu_probs: np.ndarray = None,
                             l_seq_hidden_states: np.ndarray = None):
        """
        Args:
            l_indexes: [1, episode_len]
            l_padding_masks: [1, episode_len]
            l_obses_list: list([1, episode_len, *obs_shapes_i], ...)
            l_actions: [1, episode_len, action_size]
            l_rewards: [1, episode_len]
            next_obs_list: list([1, *obs_shapes_i], ...)
            l_dones: [1, episode_len]
            l_mu_probs: [1, episode_len]
            l_seq_hidden_states: [1, episode_len, *seq_hidden_state_shape]

        Returns:
            The td-error of raw episode observations
            [episode_len, ]
        """
        ignore_size = self.burn_in_step + self.n_step

        (bn_indexes,
         bn_padding_masks,
         bn_obses_list,
         bn_actions,
         bn_rewards,
         next_obs_list,
         bn_dones,
         bn_mu_probs,
         f_seq_hidden_states) = episode_to_batch(self.burn_in_step + self.n_step,
                                                 l_indexes.shape[1],
                                                 l_indexes,
                                                 l_padding_masks,
                                                 l_obses_list,
                                                 l_actions,
                                                 l_rewards,
                                                 next_obs_list,
                                                 l_dones,
                                                 l_probs=l_mu_probs,
                                                 l_seq_hidden_states=l_seq_hidden_states)

        """
        bn_indexes: [episode_len - bn + 1, bn]
        bn_padding_masks: [episode_len - bn + 1, bn]
        bn_obses_list: list([episode_len - bn + 1, bn, *obs_shapes_i], ...)
        bn_actions: [episode_len - bn + 1, bn, action_size]
        bn_rewards: [episode_len - bn + 1, bn]
        next_obs_list: list([episode_len - bn + 1, *obs_shapes_i], ...)
        bn_dones: [episode_len - bn + 1, bn]
        bn_mu_probs: [episode_len - bn + 1, bn]
        f_seq_hidden_states: [episode_len - bn + 1, 1, *seq_hidden_state_shape]
        """

        td_error_list = []
        all_batch = bn_obses_list[0].shape[0]
        batch_size = self.batch_size
        for i in range(math.ceil(all_batch / batch_size)):
            b_i, b_j = i * batch_size, (i + 1) * batch_size

            _bn_indexes = torch.from_numpy(bn_indexes[b_i:b_j]).to(self.device)
            _bn_padding_masks = torch.from_numpy(bn_padding_masks[b_i:b_j]).to(self.device)
            _bn_obses_list = [torch.from_numpy(o[b_i:b_j]).to(self.device) for o in bn_obses_list]
            _bn_actions = torch.from_numpy(bn_actions[b_i:b_j]).to(self.device)
            _bn_rewards = torch.from_numpy(bn_rewards[b_i:b_j]).to(self.device)
            _next_obs_list = [torch.from_numpy(o[b_i:b_j]).to(self.device) for o in next_obs_list]
            _bn_dones = torch.from_numpy(bn_dones[b_i:b_j]).to(self.device)
            _bn_mu_probs = torch.from_numpy(bn_mu_probs[b_i:b_j]).to(self.device) if self.use_n_step_is else None
            _f_seq_hidden_states = torch.from_numpy(f_seq_hidden_states[b_i:b_j]).to(self.device) if self.seq_encoder is not None else None

            td_error = self._get_td_error(bn_indexes=_bn_indexes,
                                          bn_padding_masks=_bn_padding_masks,
                                          bn_obses_list=_bn_obses_list,
                                          bn_actions=_bn_actions,
                                          bn_rewards=_bn_rewards,
                                          next_obs_list=_next_obs_list,
                                          bn_dones=_bn_dones,
                                          bn_mu_probs=_bn_mu_probs,
                                          f_seq_hidden_states=_f_seq_hidden_states).detach().cpu().numpy()
            td_error_list.append(td_error.flatten())

        td_error = np.concatenate([*td_error_list,
                                   np.zeros(ignore_size, dtype=np.float32)])
        return td_error

    def put_episode(self, *episode_trans):
        # Ignore episodes which length is too short
        if episode_trans[0].shape[1] < self.burn_in_step + self.n_step:
            return

        if self.use_replay_buffer:
            self._fill_replay_buffer(*episode_trans)
        else:
            self.batch_buffer.put_episode(*episode_trans)

    def _fill_replay_buffer(self,
                            l_indexes: np.ndarray,
                            l_padding_masks: np.ndarray,
                            l_obses_list: List[np.ndarray],
                            l_actions: np.ndarray,
                            l_rewards: np.ndarray,
                            next_obs_list: List[np.ndarray],
                            l_dones: np.ndarray,
                            l_probs: List[np.ndarray],
                            l_seq_hidden_states: np.ndarray = None):
        """
        Args:
            l_indexes: [1, episode_len]
            l_padding_masks: [1, episode_len]
            l_obses_list: list([1, episode_len, *obs_shapes_i], ...)
            l_actions: [1, episode_len, action_size]
            l_rewards: [1, episode_len]
            next_obs_list: list([1, *obs_shapes_i], ...)
            l_dones: [1, episode_len]
            l_probs: [1, episode_len]
            l_seq_hidden_states: [1, episode_len, *seq_hidden_state_shape]
        """
        # Reshape [1, episode_len, ...] to [episode_len, ...]
        index = l_indexes.squeeze(0)
        padding_mask = l_padding_masks.squeeze(0)
        obs_list = [l_obses.squeeze(0) for l_obses in l_obses_list]
        if self.use_normalization:
            self._udpate_normalizer([torch.from_numpy(obs).to(self.device) for obs in obs_list])
        action = l_actions.squeeze(0)
        reward = l_rewards.squeeze(0)
        done = l_dones.squeeze(0)

        # Padding next_obs for episode experience replay
        index = np.concatenate([index,
                                index[-1:] + 1])
        padding_mask = np.concatenate([padding_mask,
                                       np.zeros([1], dtype=bool)])
        obs_list = [np.concatenate([obs, next_obs]) for obs, next_obs in zip(obs_list, next_obs_list)]
        action = np.concatenate([action,
                                 np.empty([1, action.shape[-1]], dtype=np.float32)])
        reward = np.concatenate([reward,
                                 np.zeros([1], dtype=np.float32)])
        done = np.concatenate([done,
                               np.zeros([1], dtype=bool)])

        storage_data = {
            'index': index,
            'padding_mask': padding_mask,
            **{f'obs_{i}': obs for i, obs in enumerate(obs_list)},
            'action': action,
            'reward': reward,
            'done': done,
        }

        if self.use_n_step_is:
            l_mu_probs = l_probs
            mu_prob = l_mu_probs.squeeze(0)
            mu_prob = np.concatenate([mu_prob,
                                      np.empty([1], dtype=np.float32)])
            storage_data['mu_prob'] = mu_prob

        if self.seq_encoder is not None:
            seq_hidden_state = l_seq_hidden_states.squeeze(0)
            seq_hidden_state = np.concatenate([seq_hidden_state,
                                               np.empty([1, *seq_hidden_state.shape[1:]], dtype=np.float32)])
            storage_data['seq_hidden_state'] = seq_hidden_state

        # n_step transitions except the first one and the last obs_, n_step - 1 + 1
        if self.use_add_with_td:
            td_error = self.get_episode_td_error(l_indexes=l_indexes,
                                                 l_padding_masks=l_padding_masks,
                                                 l_obses_list=l_obses_list,
                                                 l_actions=l_actions,
                                                 l_rewards=l_rewards,
                                                 next_obs_list=next_obs_list,
                                                 l_dones=l_dones,
                                                 l_mu_probs=l_mu_probs if self.use_n_step_is else None,
                                                 l_seq_hidden_states=l_seq_hidden_states if self.seq_encoder is not None else None)
            self.replay_buffer.add_with_td_error(td_error, storage_data,
                                                 ignore_size=self.burn_in_step + self.n_step)
        else:
            self.replay_buffer.add(storage_data,
                                   ignore_size=self.burn_in_step + self.n_step)

        if self.seq_encoder == SEQ_ENCODER.ATTN and self.summary_writer is not None and self.summary_available:
            self.summary_available = False
            with torch.no_grad():
                l_indexes = l_indexes[:, self.burn_in_step:]
                l_obses_list = [o[:, self.burn_in_step:] for o in l_obses_list]
                l_pre_l_actions = gen_pre_n_actions(l_actions[:, self.burn_in_step:])
                l_padding_masks = l_padding_masks[:, self.burn_in_step:]
                *_, attn_weights_list = self.model_rep(torch.from_numpy(l_indexes).to(self.device),
                                                       [torch.from_numpy(o).to(self.device) for o in l_obses_list],
                                                       torch.from_numpy(l_pre_l_actions).to(self.device),
                                                       query_length=l_indexes.shape[1],
                                                       padding_mask=torch.from_numpy(l_padding_masks).to(self.device))

                for i, attn_weight in enumerate(attn_weights_list):
                    image = plot_attn_weight(attn_weight[0].cpu().numpy())
                    self.summary_writer.add_images(f'attn_weight/{i}', image, self.global_step)

    def _sample_from_replay_buffer(self):
        """
        Sample from replay buffer

        Returns:
            pointers: [Batch, ]
            (
                bn_indexes: [Batch, b + n]
                bn_padding_masks: [Batch, b + n]
                bn_obses_list: list([Batch, b + n, *obs_shapes_i], ...)
                bn_actions: [Batch, b + n, action_size]
                bn_rewards: [Batch, b + n]
                next_obs_list: list([Batch, *obs_shapes_i], ...)
                bn_dones: [Batch, b + n]
                bn_mu_probs: [Batch, b + n]
                bn_seq_hidden_states: [Batch, b + n, *seq_hidden_state_shape],
                priority_is: [Batch, 1]
            )
        """
        sampled = self.replay_buffer.sample()
        if sampled is None:
            return None

        """
        trans:
            index: [Batch, ]
            padding_mask: [Batch, ]
            obs_i: [Batch, *obs_shapes_i]
            action: [Batch, action_size]
            reward: [Batch, ]
            done: [Batch, ]
            mu_prob: [Batch, ]
            seq_hidden_state: [Batch, *seq_hidden_state_shape],
        """
        pointers, trans, priority_is = sampled

        # Get n_step transitions TODO: could be faster, no need get all data
        trans = {k: [v] for k, v in trans.items()}
        # k: [v, v, ...]
        for i in range(1, self.burn_in_step + self.n_step + 1):
            t_trans = self.replay_buffer.get_storage_data(pointers + i).items()
            for k, v in t_trans:
                trans[k].append(v)

        for k, v in trans.items():
            trans[k] = np.concatenate([np.expand_dims(t, 1) for t in v], axis=1)

        """
        m_indexes: [Batch, N + 1]
        m_padding_masks: [Batch, N + 1]
        m_obses_list: list([Batch, N + 1, *obs_shapes_i], ...)
        m_actions: [Batch, N + 1, action_size]
        m_rewards: [Batch, N + 1]
        m_dones: [Batch, N + 1]
        m_mu_probs: [Batch, N + 1]
        m_seq_hidden_state: [Batch, N + 1, *seq_hidden_state_shape]
        """
        m_indexes = trans['index']
        m_padding_masks = trans['padding_mask']
        m_obses_list = [trans[f'obs_{i}'] for i in range(len(self.obs_shapes))]
        m_actions = trans['action']
        m_rewards = trans['reward']
        m_dones = trans['done']

        bn_indexes = m_indexes[:, :-1]
        bn_padding_masks = m_padding_masks[:, :-1]
        bn_obses_list = [m_obses[:, :-1, ...] for m_obses in m_obses_list]
        bn_actions = m_actions[:, :-1, ...]
        bn_rewards = m_rewards[:, :-1]
        next_obs_list = [m_obses[:, -1, ...] for m_obses in m_obses_list]
        bn_dones = m_dones[:, :-1]

        if self.use_n_step_is:
            m_mu_probs = trans['mu_prob']
            bn_mu_probs = m_mu_probs[:, :-1]

        if self.seq_encoder is not None:
            m_seq_hidden_states = trans['seq_hidden_state']
            bn_seq_hidden_states = m_seq_hidden_states[:, :-1, ...]

        return pointers, (bn_indexes,
                          bn_padding_masks,
                          bn_obses_list,
                          bn_actions,
                          bn_rewards,
                          next_obs_list,
                          bn_dones,
                          bn_mu_probs if self.use_n_step_is else None,
                          bn_seq_hidden_states if self.seq_encoder is not None else None,
                          priority_is if self.use_priority else None)

    def train(self):
        step = self.get_global_step()

        if self.use_replay_buffer:
            train_data = self._sample_from_replay_buffer()
            if train_data is None:
                return step

            pointers, batch = train_data
            batch_list = [batch]
        else:
            batch_list = self.batch_buffer.get_batch()
            batch_list = [(*batch, None) for batch in batch_list]

        for batch in batch_list:
            (bn_indexes,
             bn_padding_masks,
             bn_obses_list,
             bn_actions,
             bn_rewards,
             next_obs_list,
             bn_dones,
             bn_mu_probs,
             bn_seq_hidden_states,
             priority_is) = batch

            """
            bn_indexes: [Batch, b + n]
            bn_padding_masks: [Batch, b + n]
            bn_obses_list: list([Batch, b + n, *obs_shapes_i], ...)
            bn_actions: [Batch, b + n, action_size]
            bn_rewards: [Batch, b + n]
            next_obs_list: list([Batch, *obs_shapes_i], ...)
            bn_dones: [Batch, b + n]
            bn_mu_probs: [Batch, b + n]
            bn_seq_hidden_states: [Batch, b + n, *seq_hidden_state_shape]
            priority_is: [Batch, 1]
            """

            bn_indexes = torch.from_numpy(bn_indexes).to(self.device)
            bn_padding_masks = torch.from_numpy(bn_padding_masks).to(self.device)
            bn_obses_list = [torch.from_numpy(t).to(self.device) for t in bn_obses_list]
            bn_actions = torch.from_numpy(bn_actions).to(self.device)
            bn_rewards = torch.from_numpy(bn_rewards).to(self.device)
            next_obs_list = [torch.from_numpy(t).to(self.device) for t in next_obs_list]
            bn_dones = torch.from_numpy(bn_dones).to(self.device)
            if self.use_n_step_is:
                bn_mu_probs = torch.from_numpy(bn_mu_probs).to(self.device)
            if self.seq_encoder is not None:
                f_seq_hidden_states = bn_seq_hidden_states[:, :1]
                f_seq_hidden_states = torch.from_numpy(f_seq_hidden_states).to(self.device)
            if self.use_replay_buffer and self.use_priority:
                priority_is = torch.from_numpy(priority_is).to(self.device)

            self._train(bn_indexes=bn_indexes,
                        bn_padding_masks=bn_padding_masks,
                        bn_obses_list=bn_obses_list,
                        bn_actions=bn_actions,
                        bn_rewards=bn_rewards,
                        next_obs_list=next_obs_list,
                        bn_dones=bn_dones,
                        bn_mu_probs=bn_mu_probs if self.use_n_step_is else None,
                        f_seq_hidden_states=f_seq_hidden_states if self.seq_encoder is not None else None,
                        priority_is=priority_is if self.use_replay_buffer and self.use_priority else None)

            if step % self.save_model_per_step == 0:
                self.save_model()

            if self.use_replay_buffer:
                if self.use_n_step_is:
                    bn_pi_probs_tensor = self.get_l_probs(bn_indexes,
                                                          bn_padding_masks,
                                                          bn_obses_list,
                                                          bn_actions,
                                                          f_seq_hidden_states=f_seq_hidden_states if self.seq_encoder is not None else None)

                # Update td_error
                if self.use_priority:
                    td_error = self._get_td_error(bn_indexes=bn_indexes,
                                                  bn_padding_masks=bn_padding_masks,
                                                  bn_obses_list=bn_obses_list,
                                                  bn_actions=bn_actions,
                                                  bn_rewards=bn_rewards,
                                                  next_obs_list=next_obs_list,
                                                  bn_dones=bn_dones,
                                                  bn_mu_probs=bn_pi_probs_tensor if self.use_n_step_is else None,
                                                  f_seq_hidden_states=f_seq_hidden_states if self.seq_encoder is not None else None).detach().cpu().numpy()
                    self.replay_buffer.update(pointers, td_error)

                # Update seq_hidden_states
                if self.seq_encoder is not None:
                    pointers_list = [pointers + i for i in range(1, self.burn_in_step + self.n_step + 1)]
                    tmp_pointers = np.stack(pointers_list, axis=1).reshape(-1)
                    bn_seq_hidden_states = self.get_l_seq_hidden_states(bn_indexes,
                                                                        bn_padding_masks,
                                                                        bn_obses_list,
                                                                        bn_actions,
                                                                        f_seq_hidden_states=f_seq_hidden_states).detach().cpu().numpy()
                    seq_hidden_state = bn_seq_hidden_states.reshape(-1, *bn_seq_hidden_states.shape[2:])
                    self.replay_buffer.update_transitions(tmp_pointers, 'seq_hidden_state', seq_hidden_state)

                # Update n_mu_probs
                if self.use_n_step_is:
                    pointers_list = [pointers + i for i in range(0, self.burn_in_step + self.n_step)]
                    tmp_pointers = np.stack(pointers_list, axis=1).reshape(-1)
                    pi_probs = bn_pi_probs_tensor.detach().cpu().numpy().reshape(-1)
                    self.replay_buffer.update_transitions(tmp_pointers, 'mu_prob', pi_probs)

            step = self._increase_global_step()

        return step
