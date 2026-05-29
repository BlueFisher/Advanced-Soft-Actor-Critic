import logging
import random
from collections import defaultdict
from itertools import chain
from pathlib import Path

import numpy as np
import torch
from torch import autograd, distributions, nn, optim
from torch.nn import functional
from torch.utils.tensorboard import SummaryWriter

from .nn_models import *
from .utils import *

torch.set_printoptions(precision=4, sci_mode=False)


class PPO_Base:
    _closed = False

    def __init__(self,
                 obs_names: list[str],
                 obs_shapes: list[tuple[int]],
                 d_action_sizes: list[int],
                 c_action_size: int,
                 model_abs_dir: Path | None,
                 nn,

                 device: str | None = None,
                 ma_name: str | None = None,
                 summary_path: str | None = 'log',
                 train_mode: bool = True,
                 last_ckpt: str | None = None,

                 nn_config: dict | None = None,

                 seed: float | None = None,
                 write_summary_per_step: float = 1e3,
                 save_model_per_step: float = 1e5,

                 use_replay_buffer: bool = True,
                 use_priority: bool = True,

                 ensemble_q_num: int = 2,
                 ensemble_q_sample: int = 2,

                 burn_in_step: int = 0,
                 n_step: int = 1,
                 seq_encoder: SEQ_ENCODER | None = None,

                 batch_size: int = 256,
                 tau: float = 0.005,
                 update_target_per_step: int = 1,
                 init_log_alpha: float = -2.3,
                 use_auto_alpha: bool = True,
                 target_d_alpha: float = 0.98,
                 target_c_alpha: float = 1.,
                 d_policy_entropy_penalty: float = 0.5,

                 learning_rate: float = 3e-4,

                 gamma: float = 0.99,
                 v_lambda: float = 1.,
                 v_rho: float = 1.,
                 v_c: float = 1.,
                 clip_epsilon: float = 0.2,

                 discrete_dqn_like: bool = False,
                 discrete_dqn_epsilon: float = 0.2,
                 use_n_step_is: bool = True,

                 siamese: SIAMESE | None = None,
                 siamese_use_q: bool = False,
                 siamese_use_adaptive: bool = False,

                 use_prediction: bool = False,
                 transition_kl: float = 0.8,
                 use_extra_data: bool = True,

                 curiosity: CURIOSITY | None = None,
                 curiosity_strength: float = 1.,
                 use_rnd: bool = False,
                 rnd_n_sample: int = 10,

                 use_normalization: bool = False,

                 offline_enabled: bool = False,
                 offline_loss: bool = False,

                 action_noise: list[float] | None = None,

                 ppo_epoch: int = 5,
                 ppo_value_coef: float = 0.5,
                 ppo_entropy_coef: float = 0.01,

                 replay_config: dict | None = None):
        """
        obs_names: list of names of observations
        obs_shapes: list of dimensions of observations
        d_action_sizes: Dimensions of discrete actions
        c_action_size: Dimension of continuous actions
        model_abs_dir: The directory that saves summary, checkpoints, config etc.
        nn: nn # Neural network models file

        device: Training in CPU or GPU
        ma_name: Multi-agent name
        train_mode: Is training or inference
        last_ckpt: The checkpoint to restore

        nn_config: nn model config

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
        target_d_alpha: 0.98 # Target discrete alpha ratio
        target_c_alpha: 1.0 # Target continuous alpha ratio
        d_policy_entropy_penalty: 0.5 # Discrete policy entropy penalty ratio

        learning_rate: 0.0003 # Learning rate of all optimizers

        gamma: 0.99 # Discount factor
        v_lambda: 1.0 # Discount factor for V-trace
        v_rho: 1.0 # Rho for V-trace
        v_c: 1.0 # C for V-trace
        clip_epsilon: 0.2 # Epsilon for q clip

        discrete_dqn_like: false # Whether using policy or only Q network if discrete is in action spaces
        discrete_dqn_epsilon: 0.2 # Probability of using random action
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

        offline_loss: false # Whether using offline loss

        action_noise: null # [noise_min, noise_max]
        """
        self._kwargs = locals()

        self.obs_names = obs_names
        self.obs_shapes = obs_shapes
        self.d_action_sizes = d_action_sizes
        self.d_action_summed_size = sum(d_action_sizes)
        self.d_action_branch_size = len(d_action_sizes)
        self.c_action_size = c_action_size
        self.model_abs_dir = model_abs_dir
        self.ma_name = ma_name
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
        self.target_d_alpha = target_d_alpha
        self.target_c_alpha = target_c_alpha
        self.d_policy_entropy_penalty = d_policy_entropy_penalty

        self.learning_rate = learning_rate

        self.gamma = gamma
        self.v_lambda = v_lambda
        self.v_rho = v_rho
        self.v_c = v_c
        self.clip_epsilon = clip_epsilon

        self.discrete_dqn_like = discrete_dqn_like
        self.discrete_dqn_epsilon = discrete_dqn_epsilon

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

        self.offline_enabled = offline_enabled
        self.offline_loss = offline_loss

        self.action_noise = action_noise

        self.ppo_epoch = ppo_epoch
        self.ppo_value_coef = ppo_value_coef
        self.ppo_entropy_coef = ppo_entropy_coef

        self._set_logger()

        self.use_replay_buffer = False
        self.use_priority = False
        self.use_auto_alpha = False
        self.use_n_step_is = False
        self.discrete_dqn_like = False
        self.siamese_use_q = False

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device(f'cuda:{random.randint(0, torch.cuda.device_count() - 1)}')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        self._logger.info(f'Device: {self.device.type}:{self.device.index}')

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        distributions.Distribution.set_default_validate_args(False)

        self.summary_writer = None
        if summary_path and self.model_abs_dir and self.train_mode:
            summary_path = Path(self.model_abs_dir).joinpath(summary_path)
            self.summary_writer = SummaryWriter(str(summary_path))
            self.summary_available = True

        self._profiler = UnifiedElapsedTimer(self._logger)

        self._build_model(nn, nn_config, init_log_alpha, learning_rate)
        self._build_ckpt()
        self._init_replay_buffer()
        self._init_or_restore(int(last_ckpt) if last_ckpt is not None else None)

    def _set_logger(self):
        if self.ma_name is None:
            self._logger = logging.getLogger('sac.base')
        else:
            self._logger = logging.getLogger(f'sac.base.{self.ma_name}')

    def _build_model(self, nn, nn_config: dict | None, init_log_alpha: float, learning_rate: float) -> None:
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

        self.v_rho = torch.tensor(self.v_rho, device=self.device)
        self.v_c = torch.tensor(self.v_c, device=self.device)

        d_action_list = [np.eye(d_action_size, dtype=np.float32)[0]
                         for d_action_size in self.d_action_sizes]
        self._padding_action = np.concatenate(d_action_list + [np.zeros(self.c_action_size, dtype=np.float32)], axis=-1)

        def adam_optimizer(params, learning_rate: float) -> optim.Adam | None:
            params = list(params)
            if len(params) == 0:
                return
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
        if self.seq_encoder in (None, SEQ_ENCODER.RNN):
            self.model_rep: ModelBaseRep = ModelRep(self.obs_names,
                                                    self.obs_shapes,
                                                    self.d_action_sizes, self.c_action_size,
                                                    False,
                                                    self.model_abs_dir,
                                                    **nn_config['rep']).to(self.device)
            self.model_target_rep: ModelBaseRep = ModelRep(self.obs_names,
                                                           self.obs_shapes,
                                                           self.d_action_sizes, self.c_action_size,
                                                           True,
                                                           self.model_abs_dir,
                                                           **nn_config['rep']).to(self.device)
            # Get represented state and seq_hidden_state_shape
            test_obs_list = [torch.rand(self.batch_size, 1, *obs_shape, device=self.device) for obs_shape in self.obs_shapes]
            test_pre_action = torch.rand(self.batch_size, 1, self.d_action_summed_size + self.c_action_size, device=self.device)
            test_state, test_seq_hidden_states = self.model_rep(test_obs_list,
                                                                test_pre_action,
                                                                None)
            state_size = test_state.shape[-1]
            seq_hidden_state_shape = test_seq_hidden_states.shape[2:]  # [batch, 1, *seq_hidden_state_shape]

        elif self.seq_encoder == SEQ_ENCODER.ATTN:
            self.model_rep: ModelBaseAttentionRep = ModelRep(self.obs_names,
                                                             self.obs_shapes,
                                                             self.d_action_sizes, self.c_action_size,
                                                             False,
                                                             self.model_abs_dir,
                                                             **nn_config['rep']).to(self.device)
            self.model_target_rep: ModelBaseAttentionRep = ModelRep(self.obs_names,
                                                                    self.obs_shapes,
                                                                    self.d_action_sizes, self.c_action_size,
                                                                    True,
                                                                    self.model_abs_dir,
                                                                    **nn_config['rep']).to(self.device)
            # Get represented state and seq_hidden_state_shape
            test_index = torch.zeros((self.batch_size, 1), dtype=torch.int32, device=self.device)
            test_obs_list = [torch.rand(self.batch_size, 1, *obs_shape, device=self.device) for obs_shape in self.obs_shapes]
            test_pre_action = torch.rand(self.batch_size, 1, self.d_action_summed_size + self.c_action_size, device=self.device)
            test_state, test_attn_state, _ = self.model_rep(1,
                                                            test_index,
                                                            test_obs_list,
                                                            test_pre_action,
                                                            None)
            state_size, seq_hidden_state_shape = test_state.shape[-1], test_attn_state.shape[2:]

        for param in self.model_target_rep.parameters():
            param.requires_grad = False

        self.state_size = state_size
        self.seq_hidden_state_shape = seq_hidden_state_shape
        self._logger.info(f'State size: {state_size}')
        self._logger.info(f'Seq hidden state shape: {tuple(seq_hidden_state_shape)}')

        self.optimizer_rep = adam_optimizer(self.model_rep.parameters(), learning_rate)

        """ V """
        self.model_v: ModelBaseV = nn.ModelV(state_size,
                                             False,
                                             self.model_abs_dir).to(self.device)
        self.model_target_v: ModelBaseV = nn.ModelV(state_size,
                                                    True,
                                                    self.model_abs_dir).to(self.device)
        for param in self.model_target_v.parameters():
            param.requires_grad = False

        self.optimizer_v = adam_optimizer(self.model_v.parameters(), 0.001)

        """ POLICY """
        self.model_policy: ModelBasePolicy = nn.ModelPolicy(state_size, self.d_action_sizes, self.c_action_size,
                                                            self.model_abs_dir,
                                                            **nn_config['policy']).to(self.device)
        self.optimizer_policy = adam_optimizer(self.model_policy.parameters(), learning_rate)

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
                self.model_rep_projection_list: list[ModelBaseRepProjection] = [
                    nn.ModelRepProjection(test_encoder.shape[-1]).to(self.device) for test_encoder in test_encoder_list]
                self.model_target_rep_projection_list: list[ModelBaseRepProjection] = [
                    nn.ModelRepProjection(test_encoder.shape[-1]).to(self.device) for test_encoder in test_encoder_list]

                test_projection_list = [pro(test_encoder) for pro, test_encoder in zip(self.model_rep_projection_list, test_encoder_list)]
                self.model_rep_prediction_list: list[ModelBaseRepPrediction] = [
                    nn.ModelRepPrediction(test_projection.shape[-1]).to(self.device) for test_projection in test_projection_list]
                self.optimizer_siamese = adam_optimizer(chain(*[pro.parameters() for pro in self.model_rep_projection_list],
                                                              *[pre.parameters() for pre in self.model_rep_prediction_list]))

        """ RECURRENT PREDICTION MODELS """
        if self.use_prediction:
            self.model_transition: ModelBaseTransition = nn.ModelTransition(state_size,
                                                                            self.d_action_summed_size,
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

        if self.d_action_sizes:
            d_action_sizes = torch.tensor(self.d_action_sizes, device=self.device)
            d_action_sizes = torch.repeat_interleave(d_action_sizes.type(torch.float32), d_action_sizes)
            self.target_d_alpha = self.target_d_alpha * (-torch.log(1 / d_action_sizes))

        if self.c_action_size:
            self.target_c_alpha = self.target_c_alpha

        if self.use_auto_alpha:
            self.optimizer_alpha = adam_optimizer([self.log_d_alpha, self.log_c_alpha])

        """ CURIOSITY """
        if self.curiosity == CURIOSITY.FORWARD:
            self.model_forward_dynamic: ModelBaseForwardDynamic = nn.ModelForwardDynamic(state_size,
                                                                                         self.d_action_summed_size + self.c_action_size).to(self.device)
            self.optimizer_curiosity = adam_optimizer(self.model_forward_dynamic.parameters())

        elif self.curiosity == CURIOSITY.INVERSE:
            self.model_inverse_dynamic: ModelBaseInverseDynamic = nn.ModelInverseDynamic(state_size,
                                                                                         self.d_action_summed_size + self.c_action_size).to(self.device)
            self.optimizer_curiosity = adam_optimizer(self.model_inverse_dynamic.parameters())

        """ RANDOM NETWORK DISTILLATION """
        if self.use_rnd:
            self.model_rnd: ModelRND = nn.ModelRND(state_size, self.d_action_summed_size, self.c_action_size).to(self.device)
            self.model_target_rnd: ModelRND = nn.ModelRND(state_size, self.d_action_summed_size, self.c_action_size).to(self.device)
            for param in self.model_target_rnd.parameters():
                param.requires_grad = False
            self.optimizer_rnd = adam_optimizer(self.model_rnd.parameters())

    def _build_ckpt(self) -> None:
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

        if self.optimizer_rep is not None:
            ckpt_dict['model_rep'] = self.model_rep
            ckpt_dict['model_target_rep'] = self.model_target_rep
            ckpt_dict['optimizer_rep'] = self.optimizer_rep

        """ V """
        ckpt_dict['model_v'] = self.model_v
        ckpt_dict['model_target_v'] = self.model_target_v
        ckpt_dict['optimizer_v'] = self.optimizer_v

        """ POLICY """
        ckpt_dict['model_policy'] = self.model_policy
        ckpt_dict['optimizer_policy'] = self.optimizer_policy

        """ SIAMESE REPRESENTATION LEARNING """
        if self.siamese == SIAMESE.ATC:
            for i, weight in enumerate(self.contrastive_weight_list):
                ckpt_dict[f'contrastive_weights_{i}'] = weight
            ckpt_dict['optimizer_siamese'] = self.optimizer_siamese
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
            ckpt_dict['model_target_rnd'] = self.model_target_rnd
            ckpt_dict['optimizer_rnd'] = self.optimizer_rnd

        total_parameter_num = 0
        for m in ckpt_dict.values():
            if isinstance(m, nn.Module):
                total_parameter_num += sum([p.numel() for p in m.parameters()])
        self._logger.info(f'Parameters: {total_parameter_num}')

    def _init_or_restore(self, last_ckpt: int | None) -> None:
        """
        Initialize network weights from scratch or restore from model_abs_dir
        """
        self.ckpt_dir = None
        if not self.model_abs_dir:
            return

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
            elif last_ckpt not in ckpts:
                self._logger.warning(f'{last_ckpt} NOT IN {ckpts}, using {ckpts[-1]}')
                last_ckpt = ckpts[-1]

            ckpt_restore_path = ckpt_dir.joinpath(f'{last_ckpt}.pth')
            ckpt_restore = torch.load(ckpt_restore_path, map_location=self.device, weights_only=True)
            error_occurred = False
            for name, model in self.ckpt_dict.items():
                if name not in ckpt_restore:
                    self._logger.warning(f'{name} not in {last_ckpt}.pth')
                    continue

                if isinstance(model, torch.Tensor):
                    model.data = ckpt_restore[name]
                else:
                    try:
                        if error_occurred and name.startswith('optimizer'):
                            # If state_dict mismatch occurred, optimizer should not be restored
                            continue
                        model.load_state_dict(ckpt_restore[name])
                    except RuntimeError as e:
                        error_occurred = True
                        self._logger.error(e)
                    if isinstance(model, nn.Module):
                        if self.train_mode:
                            model.train()
                        else:
                            model.eval()

            self.global_step = self.global_step.to('cpu')
            self.ckpt_dict['global_step'] = self.global_step

            self._logger.info(f'Restored from {ckpt_restore_path}')

            if self.train_mode and self.use_replay_buffer:
                self.replay_buffer.load(ckpt_dir, last_ckpt)

                self._logger.info(f'Replay buffer restored')
        else:
            self._logger.info('Initializing from scratch')
            self._update_target_variables()

    def _init_replay_buffer(self) -> None:
        if self.train_mode:
            self.episode_buffer = []

    def set_train_mode(self, train_mode=True):
        self.train_mode = train_mode
        for m in self.ckpt_dict.values():
            if isinstance(m, nn.Module):
                m.train(mode=self.train_mode)

    def save_model(self, save_replay_buffer=False) -> None:
        if self.ckpt_dir is None:
            return

        global_step = self.get_global_step()
        ckpt_path = self.ckpt_dir.joinpath(f'{global_step}.pth')

        torch.save({
            k: v if isinstance(v, torch.Tensor) else v.state_dict()
            for k, v in self.ckpt_dict.items()
        }, ckpt_path)
        self._logger.info(f"Model saved at {ckpt_path}")

        if self.use_replay_buffer and save_replay_buffer:
            self.replay_buffer.save(self.ckpt_dir, global_step)

    def write_constant_summaries(self, constant_summaries: list[dict], iteration=None) -> None:
        """
        Write constant information from sac_main.py, such as reward, iteration, etc.
        """
        if self.summary_writer is None:
            return

        for s in constant_summaries:
            self.summary_writer.add_scalar(s['tag'], s['simple_value'],
                                           self.get_global_step() if iteration is None else iteration)

        self.summary_writer.flush()

    def write_histogram_summaries(self, histograms, iteration=None) -> None:
        if self.summary_writer is None:
            return

        for s in histograms:
            self.summary_writer.add_histogram(s['tag'], s['histogram'],
                                              self.get_global_step() if iteration is None else iteration)

        self.summary_writer.flush()

    def increase_global_step(self) -> int:
        self.global_step.add_(1)

        return self.global_step.item()

    def set_global_step(self, global_step: int | torch.Tensor):
        if global_step == self.get_global_step():
            return

        if isinstance(global_step, torch.Tensor):
            global_step = global_step.item()

        self._logger.warning(f'Global step {self.get_global_step()} -> {global_step}')
        self.global_step.copy_(global_step)

    def get_global_step(self) -> int:
        return self.global_step.item()

    def get_initial_action(self, batch_size, get_numpy=True) -> np.ndarray:
        if get_numpy:
            if self.d_action_sizes:
                d_actions = [np.random.randint(0, d_action_size, size=batch_size)
                             for d_action_size in self.d_action_sizes]
                d_actions = [np.eye(d_action_size, dtype=np.int32)[d_action]
                             for d_action, d_action_size in zip(d_actions, self.d_action_sizes)]
                d_action = np.concatenate(d_actions, axis=-1).astype(np.float32)
            else:
                d_action = np.zeros((batch_size, 0), dtype=np.float32)

            c_action = np.zeros([batch_size, self.c_action_size], dtype=np.float32)

            return np.concatenate([d_action, c_action], axis=-1)
        else:
            if self.d_action_sizes:
                d_actions = [torch.randint(0, d_action_size, (batch_size,), device=self.device)
                             for d_action_size in self.d_action_sizes]
                d_actions_one_hot = [functional.one_hot(d_action, num_classes=d_action_size)
                                     for d_action, d_action_size in zip(d_actions, self.d_action_sizes)]
                d_action = torch.cat(d_actions_one_hot, dim=-1)
            else:
                d_action = torch.zeros((batch_size, 0), dtype=torch.float32, device=self.device)

            c_action = torch.zeros((batch_size, self.c_action_size), dtype=torch.float32, device=self.device)

            return torch.cat([d_action, c_action], dim=-1)

    def get_initial_seq_hidden_state(self, batch_size, get_numpy=True) -> np.ndarray | torch.Tensor:
        if get_numpy:
            return np.zeros([batch_size, *self.seq_hidden_state_shape], dtype=np.float32)
        else:
            return torch.zeros([batch_size, *self.seq_hidden_state_shape], device=self.device)

    @torch.no_grad()
    def _update_target_variables(self, tau=1.) -> None:
        """
        Soft (momentum) update target networks (default hard)
        """
        target = self.model_target_rep.parameters()
        source = self.model_rep.parameters()

        target = chain(target, self.model_target_v.parameters())
        source = chain(source, self.model_v.parameters())

        if self.siamese == 'BYOL':
            target = chain(target, *[t_pro.parameters() for t_pro in self.model_target_rep_projection_list])
            source = chain(source, *[pro.parameters() for pro in self.model_rep_projection_list])

        for target_param, param in zip(target, source):
            target_param.data.copy_(
                target_param.data * (1. - tau) + param.data * tau
            )

    @torch.no_grad()
    def _udpate_normalizer(self, obs_list: list[torch.Tensor]) -> None:
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

    def _process_torch_obs_list(self, obs_list: torch.Tensor):
        for i, o in enumerate(obs_list):
            if o.dtype == torch.uint8:
                obs_list[i] = o.type(torch.float32) / 255.
            elif o.dtype == torch.bool:
                obs_list[i] = o.type(torch.float32)

    #################### ! GET ACTION ####################

    @torch.no_grad()
    def rnd_sample_d_action(self, state: torch.Tensor,
                            d_policy: distributions.Categorical) -> torch.Tensor:
        """
        Sample action `self.rnd_n_sample` times,
        choose the action that has the max (model_rnd(s, a) - model_target_rnd(s, a))**2

        Args:
            state: [batch, state_size]
            d_policy: [batch, d_action_summed_size]

        Returns:
            d_action: [batch, d_action_summed_size]
        """
        batch = state.shape[0]
        n_sample = self.rnd_n_sample

        d_actions = d_policy.sample((n_sample,))  # [n_sample, batch, d_action_summed_size]
        d_actions = d_actions.transpose(0, 1)  # [batch, n_sample, d_action_summed_size]

        d_rnd = self.model_rnd.cal_d_rnd(state)  # [batch, d_action_summed_size, f]
        t_d_rnd = self.model_target_rnd.cal_d_rnd(state)  # [batch, d_action_summed_size, f]
        d_rnd = torch.repeat_interleave(torch.unsqueeze(d_rnd, 1), n_sample, dim=1)  # [batch, n_sample, d_action_summed_size, f]
        t_d_rnd = torch.repeat_interleave(torch.unsqueeze(t_d_rnd, 1), n_sample, dim=1)  # [batch, n_sample, d_action_summed_size, f]

        _i = d_actions.unsqueeze(-1)  # [batch, n_sample, d_action_summed_size, 1]
        d_rnd = (torch.repeat_interleave(_i, d_rnd.shape[-1], dim=-1) * d_rnd).sum(-2)
        # [batch, n_sample, d_action_summed_size, f] -> [batch, n_sample, f]
        t_d_rnd = (torch.repeat_interleave(_i, t_d_rnd.shape[-1], dim=-1) * t_d_rnd).sum(-2)
        # [batch, n_sample, d_action_summed_size, f] -> [batch, n_sample, f]

        d_loss = torch.sum(torch.pow(d_rnd - t_d_rnd, 2), dim=-1)  # [batch, n_sample]
        d_idx = torch.argmax(d_loss, dim=1)  # [batch, ]

        return d_actions[torch.arange(batch), d_idx]

    @torch.no_grad()
    def rnd_sample_c_action(self, state: torch.Tensor,
                            c_policy: distributions.Normal) -> torch.Tensor:
        """
        Sample action `self.rnd_n_sample` times,
        choose the action that has the max (model_rnd(s, a) - model_target_rnd(s, a))**2

        Args:
            state: [batch, state_size]
            c_policy: [batch, c_action_size]

        Returns:
            c_action: [batch, c_action_size]
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
    def _random_action(self, d_action, c_action) -> tuple[torch.Tensor, torch.Tensor]:
        if self.action_noise is None:
            return d_action, c_action

        batch = max(d_action.shape[0], c_action.shape[0])

        action_noise = torch.linspace(*self.action_noise, steps=batch, device=self.device)  # [batch, ]

        if self.d_action_sizes:
            random_d_actions = [torch.argmax(torch.rand(batch, d_action_size, device=self.device), dim=-1)
                                for d_action_size in self.d_action_sizes]
            random_d_actions = [functional.one_hot(random_d_action, d_action_size)
                                for random_d_action, d_action_size in zip(random_d_actions, self.d_action_sizes)]
            random_d_action = torch.concat(random_d_actions, axis=-1).type(torch.float32)

            mask = torch.rand(batch, device=self.device) < action_noise
            d_action[mask] = random_d_action[mask]

        if self.c_action_size:
            c_action = torch.tanh(torch.atanh(c_action) + torch.randn(batch, self.c_action_size, device=self.device) * action_noise.unsqueeze(1))

        return d_action, c_action

    @torch.no_grad()
    def _choose_action(self,
                       obs_list: list[torch.Tensor],
                       state: torch.Tensor,
                       offline_action: torch.Tensor | None = None,
                       disable_sample: bool = False,
                       force_rnd_if_available: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs_list: list([batch, *obs_shapes_i], ...)
            state: [batch, state_size]
            offline_action: [batch, action_size]

        Returns:
            action: [batch, action_size]
            prob: [batch, action_size]
        """
        batch = state.shape[0]
        d_policy, c_policy = self.model_policy(state, obs_list)

        if offline_action is None:
            if self.d_action_sizes:
                if self.discrete_dqn_like:
                    d_qs, _ = self.model_q_list[0](state, c_policy.sample() if self.c_action_size else None, obs_list)
                    d_action_list = [torch.argmax(d_q, dim=-1) for d_q in d_qs.split(self.d_action_sizes, dim=-1)]
                    d_action_list = [functional.one_hot(d_action, d_action_size).type(torch.float32)
                                     for d_action, d_action_size in zip(d_action_list, self.d_action_sizes)]
                    d_action = torch.concat(d_action_list, dim=-1)  # [batch, d_action_summed_size]

                    if self.train_mode:
                        if self.use_rnd and (self.train_mode or force_rnd_if_available):
                            s_rnd = self.model_rnd.cal_s_rnd(state)  # [batch, f]
                            t_s_rnd = self.model_target_rnd.cal_s_rnd(state)  # [batch, f]
                            s_rnd, t_s_rnd = torch.sigmoid(s_rnd), torch.sigmoid(t_s_rnd)

                            s_loss = torch.mean(torch.abs(s_rnd - t_s_rnd), dim=-1)  # [batch, ]
                            # s_loss = torch.clip(s_loss - 0.1, 0., 0.2)
                            mask = torch.rand(batch).to(self.device) < s_loss
                        else:
                            mask = torch.rand(batch) < self.discrete_dqn_epsilon

                        # Generate random action
                        d_dist_list = [distributions.OneHotCategorical(logits=torch.ones((batch, d_action_size),
                                                                                         device=self.device),
                                                                       validate_args=False)
                                       for d_action_size in self.d_action_sizes]
                        random_d_action = torch.concat([dist.sample() for dist in d_dist_list], dim=-1)

                        d_action[mask] = random_d_action[mask]
                else:
                    if disable_sample:
                        d_action = d_policy.sample_deter()
                    elif self.use_rnd and (self.train_mode or force_rnd_if_available):
                        d_action = self.rnd_sample_d_action(state, d_policy)
                    else:
                        d_action = d_policy.sample()
            else:
                d_action = torch.zeros(0, device=self.device)

            if self.c_action_size:
                if disable_sample:
                    c_action = torch.tanh(c_policy.mean)
                elif self.use_rnd and (self.train_mode or force_rnd_if_available):
                    c_action = self.rnd_sample_c_action(state, c_policy)
                else:
                    c_action = torch.tanh(c_policy.sample())
            else:
                c_action = torch.zeros(0, device=self.device)

            d_action, c_action = self._random_action(d_action, c_action)

        else:
            d_action = offline_action[..., :self.d_action_summed_size]
            c_action = offline_action[..., self.d_action_summed_size:]

        prob = torch.ones((batch,
                           self.d_action_summed_size + self.c_action_size),
                          device=self.device)  # [batch, action_size]
        if self.d_action_sizes and not self.discrete_dqn_like:
            prob[:, :self.d_action_summed_size] = d_policy.probs  # [batch, d_action_summed_size]
        if self.c_action_size:
            c_prob = squash_correction_prob(c_policy, torch.atanh(c_action))  # [batch, c_action_size]
            prob[:, self.d_action_summed_size:] = c_prob  # [batch, c_action_size]

        return torch.cat([d_action, c_action], dim=-1), prob

    @torch.no_grad()
    def choose_action(self,
                      obs_list: list[np.ndarray],
                      pre_action: np.ndarray,
                      pre_seq_hidden_state: np.ndarray,

                      offline_action: np.ndarray | None = None,
                      disable_sample: bool = False,
                      force_rnd_if_available: bool = False) -> tuple[np.ndarray,
                                                                     np.ndarray,
                                                                     np.ndarray]:
        """
        Args:
            obs_list (np): list([batch, *obs_shapes_i], ...)
            pre_action (np): [batch, action_size]
            pre_seq_hidden_state (np): [batch, *seq_hidden_state_shape]

            offline_action (np): [batch, action_size]

        Returns:
            action (np): [batch, action_size]
            prob (np): [batch, action_size]
            seq_hidden_state (np): [batch, *seq_hidden_state_shape]
        """
        obs_list = [torch.from_numpy(obs).to(self.device) for obs in obs_list]
        self._process_torch_obs_list(obs_list)

        pre_action = torch.from_numpy(pre_action).to(self.device)
        pre_seq_hidden_state = torch.from_numpy(pre_seq_hidden_state).to(self.device)

        obs_list = [obs.unsqueeze(1) for obs in obs_list]
        pre_action = pre_action.unsqueeze(1)
        pre_seq_hidden_state = pre_seq_hidden_state.unsqueeze(1)

        state, seq_hidden_state = self.model_rep(obs_list, pre_action, pre_seq_hidden_state)
        # state: [batch, 1, state_size]
        # seq_hidden_state: [batch, 1, *seq_hidden_state_shape]

        state = state.squeeze(1)
        obs_list = [obs.squeeze(1) for obs in obs_list]
        seq_hidden_state = seq_hidden_state.squeeze(1)

        offline_action = torch.from_numpy(offline_action).to(self.device) if offline_action is not None else None
        action, prob = self._choose_action(obs_list,
                                           state,
                                           offline_action,
                                           disable_sample,
                                           force_rnd_if_available)
        return (action.detach().cpu().numpy(),
                prob.detach().cpu().numpy(),
                seq_hidden_state.detach().cpu().numpy())

    @torch.no_grad()
    def choose_attn_action(self,
                           ep_indexes: np.ndarray,
                           ep_padding_masks: np.ndarray,
                           ep_obses_list: list[np.ndarray],
                           ep_pre_actions: np.ndarray,
                           ep_pre_attn_states: np.ndarray,

                           offline_action: np.ndarray | None = None,

                           disable_sample: bool = False,
                           force_rnd_if_available: bool = False) -> tuple[np.ndarray,
                                                                          np.ndarray,
                                                                          np.ndarray]:
        """
        Args:
            ep_indexes (np.int32): [batch, ep_len]
            ep_padding_masks (bool): [batch, ep_len]
            ep_obses_list (np): list([batch, ep_len, *obs_shapes_i], ...)
            ep_pre_actions (np): [batch, ep_len, action_size]
            ep_pre_attn_states (np): [batch, ep_len, *seq_hidden_state_shape]

            offline_action (np): [batch, action_size]

        Returns:
            action (np): [batch, action_size]
            prob (np): [batch, action_size]
            attn_state (np): [batch, *attn_state_shape]
        """
        ep_indexes = ep_indexes[:, -self.burn_in_step:]
        ep_padding_masks = ep_padding_masks[:, -self.burn_in_step:]
        ep_obses_list = [ep_obses[:, -self.burn_in_step:] for ep_obses in ep_obses_list]
        ep_pre_actions = ep_pre_actions[:, -self.burn_in_step:]
        ep_pre_attn_states = ep_pre_attn_states[:, -self.burn_in_step:]

        ep_indexes = torch.from_numpy(ep_indexes).to(self.device)
        ep_padding_masks = torch.from_numpy(ep_padding_masks).to(self.device)
        ep_obses_list = [torch.from_numpy(obs).to(self.device) for obs in ep_obses_list]
        self._process_torch_obs_list(ep_obses_list)

        ep_pre_actions = torch.from_numpy(ep_pre_actions).to(self.device)
        ep_pre_attn_states = torch.from_numpy(ep_pre_attn_states).to(self.device)

        state, attn_state, _ = self.model_rep(1,
                                              ep_indexes,
                                              ep_obses_list,
                                              ep_pre_actions,
                                              pre_seq_hidden_state=ep_pre_attn_states,
                                              is_prev_hidden_state=False,
                                              padding_mask=ep_padding_masks)
        # state: [batch, 1, state_size]
        # attn_state: [batch, 1, *attn_state_shape]

        state = state.squeeze(1)
        attn_state = attn_state.squeeze(1)

        offline_action = torch.from_numpy(offline_action).to(self.device) if offline_action is not None else None
        action, prob = self._choose_action([ep_obses[:, -1] for ep_obses in ep_obses_list],
                                           state,
                                           offline_action,
                                           disable_sample,
                                           force_rnd_if_available)

        return (action.detach().cpu().numpy(),
                prob.detach().cpu().numpy(),
                attn_state.detach().cpu().numpy())

    #################### ! GET STATES ####################

    def get_l_states(
        self,
        l_indexes: torch.Tensor,
        l_obses_list: list[torch.Tensor],
        l_pre_actions: torch.Tensor,
        l_pre_seq_hidden_states: torch.Tensor | None,
        l_padding_masks: torch.Tensor | None = None,
        is_target=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            l_indexes: [batch, l]
            l_padding_masks: [batch, l]
            l_obses_list: list([batch, l, *obs_shapes_i], ...)
            l_pre_actions: [batch, l, action_size]
            l_pre_seq_hidden_states: [batch, l, *seq_hidden_state_shape]

        Returns:
            l_states: [batch, l, state_size]
            l_seq_hidden_states: [batch, l, *seq_hidden_state_shape]
        """

        model_rep = self.model_target_rep if is_target else self.model_rep

        if self.seq_encoder in (None, SEQ_ENCODER.RNN):
            l_states, l_hidden_states = model_rep(l_obses_list,
                                                  l_pre_actions,
                                                  l_pre_seq_hidden_states,
                                                  padding_mask=l_padding_masks)
            return l_states, l_hidden_states  # [batch, l, state_size], [batch, l, *seq_hidden_state_shape]

        elif self.seq_encoder == SEQ_ENCODER.ATTN:
            l_states, l_attn_states, _ = model_rep(l_indexes.shape[1],
                                                   l_indexes,
                                                   l_obses_list,
                                                   l_pre_actions,
                                                   l_pre_seq_hidden_states[:, :1] if l_pre_seq_hidden_states is not None else None,
                                                   is_prev_hidden_state=True,
                                                   padding_mask=l_padding_masks)

            return l_states, l_attn_states  # [batch, l, state_size], [batch, l, *seq_hidden_state_shape]

    def log_episode(self, force: bool = False, **episode_trans: np.ndarray) -> None:
        """
        Log the episode data to the summary writer and save it to a file.

        Args:
            force (bool): If True, log the episode even if summary_writer is None or summary_available is False.
        """

        if not force:
            if self.summary_writer is None or not self.summary_available:
                return

        ep_indexes = episode_trans['ep_indexes']
        ep_obses_list = episode_trans['ep_obses_list']
        ep_actions = episode_trans['ep_actions']
        ep_rewards = episode_trans['ep_rewards']
        ep_dones = episode_trans['ep_dones']

        if self.summary_writer is not None and self.seq_encoder == SEQ_ENCODER.ATTN:
            with torch.no_grad():
                l_pre_actions = gen_n_pre_actions(ep_actions)
                torch_obs_list = [torch.from_numpy(o).to(self.device) for o in ep_obses_list]
                self._process_torch_obs_list(torch_obs_list)
                *_, attn_weights_list = self.model_rep(ep_indexes.shape[1],
                                                       torch.from_numpy(ep_indexes).to(self.device),
                                                       torch_obs_list,
                                                       torch.from_numpy(l_pre_actions).to(self.device),
                                                       None)

                for i, attn_weight in enumerate(attn_weights_list):
                    image = plot_attn_weight(attn_weight[0].cpu().numpy())
                    self.summary_writer.add_figure(f'attn_weight/{i}', image, self.global_step)

        # eps_dir = self.model_abs_dir / 'episodes'

        # eps = []
        # if eps_dir.exists():
        #     for eps_path in eps_dir.glob('*.npz'):
        #         eps.append(int(eps_path.stem))
        #     eps.sort()
        # else:
        #     eps_dir.mkdir()

        # if len(eps) == 0:
        #     eps = [-1]
        # ep_idx = eps[-1] + 1

        # np.savez(eps_dir / f'{ep_idx}.npz',
        #          index=np.squeeze(ep_indexes, axis=0),
        #          **{f'obs-{obs_name}': np.squeeze(ep_obses, axis=0) for obs_name, ep_obses in zip(self.obs_names, ep_obses_list)},
        #          action=np.squeeze(ep_actions, axis=0),
        #          reward=np.squeeze(ep_rewards, axis=0),
        #          done=np.squeeze(ep_dones, axis=0))

        self.summary_available = False

    def put_episode(self,
                    ep_indexes: np.ndarray,
                    ep_obses_list: list[np.ndarray],
                    ep_actions: np.ndarray,
                    ep_rewards: np.ndarray,
                    ep_dones: np.ndarray,
                    ep_probs: np.ndarray,
                    ep_pre_seq_hidden_states: np.ndarray) -> None:
        """
        Args:
            ep_indexes (np.int32): [1, ep_len]
            ep_obses_list (np): list([1, ep_len, *obs_shapes_i], ...)
            ep_actions (np): [1, ep_len, action_size]
            ep_rewards (np): [1, ep_len]
            ep_dones (bool): [1, ep_len]
            ep_probs (np): [1, ep_len, action_size]
            ep_pre_seq_hidden_states (np): [1, ep_len, *seq_hidden_state_shape]
        """

        if ep_indexes.shape[1] < self.n_step:
            return

        ep_indexes = torch.from_numpy(ep_indexes).to(self.device)
        ep_obses_list = [torch.from_numpy(obs).to(self.device) for obs in ep_obses_list]
        self._process_torch_obs_list(ep_obses_list)
        ep_actions = torch.from_numpy(ep_actions).to(self.device)
        ep_rewards = torch.from_numpy(ep_rewards).to(self.device)
        ep_dones = torch.from_numpy(ep_dones).to(self.device)
        ep_probs = torch.from_numpy(ep_probs).to(self.device)

        if self.use_normalization:
            self._udpate_normalizer(ep_obses_list)

        ep_states, _ = self.get_l_states(l_indexes=ep_indexes,
                                         l_obses_list=ep_obses_list,
                                         l_pre_actions=gen_n_pre_actions(ep_actions),
                                         l_pre_seq_hidden_states=None,
                                         is_target=False)

        ep_vs = self.model_v(ep_states.detach(), ep_obses_list).detach()  # [1, ep_len, 1]
        ep_vs = ep_vs.squeeze(-1)  # [1, ep_len]
        ep_target_vs = self.model_v(ep_states.detach(), ep_obses_list).detach()  # [1, ep_len, 1]
        ep_target_vs = ep_target_vs.squeeze(-1)  # [1, ep_len]

        ep_discounts = torch.pow(self.gamma,
                                 torch.arange(ep_rewards.shape[1],
                                              device=self.device,
                                              dtype=torch.float32)).view(1, -1)  # [1, ep_len]
        # 1, γ, γ^2, γ^3, γ^4, γ^5

        ep_rewards[:, -1] = 0.  # The reward of the last step is always 0, because it is the reward after the episode ends.
        # r_0, r_1, r_2, r_3, r_4, 0

        ep_discounted_returns = torch.flip(
            torch.cumsum((ep_rewards * ep_discounts).flip(1), dim=1) / ep_discounts.flip(1),
            dims=[1])  # [1, ep_len]
        # r_0 + γr_1 + γ^2r_2 + γ^3r_3 + γ^4r_4,
        # r_1 + γr_2 + γ^2r_3 + γ^3r_4,
        # r_2 + γr_3 + γ^2r_4,
        # r_3 + γr_4,
        # r_4,
        # 0
        ep_discounted_returns += ~ep_dones[:, -2] * ep_target_vs[:, -1] * ep_discounts.flip(1)  # [1, ep_len]
        # r_0 + γr_1 + γ^2r_2 + γ^3r_3 + γ^4r_4 + γ^5V(s_5),
        # r_1 + γr_2 + γ^2r_3 + γ^3r_4 + γ^4V(s_5),
        # r_2 + γr_3 + γ^2r_4 + γ^3V(s_5),
        # r_3 + γr_4 + γ^2V(s_5),
        # r_4 + γV(s_5),
        # V(s_5)

        if self.v_lambda == 1.:
            ep_advantages = ep_discounted_returns - ep_target_vs  # [1, ep_len]
        else:
            ep_target_vs_ = torch.cat([ep_target_vs[:, 1:], ep_target_vs[:, -1:]], dim=1)  # [1, ep_len]
            ep_td_errors = ep_rewards + self.gamma * ep_target_vs_ * ~ep_dones - ep_vs  # [1, ep_len]

            ep_lambda_discounts = torch.pow(self.gamma * self.v_lambda,
                                            torch.arange(ep_rewards.shape[1],
                                                         device=self.device,
                                                         dtype=torch.float32)).view(1, -1)

            ep_advantages = torch.flip(
                torch.cumsum((ep_td_errors * ep_lambda_discounts).flip(1), dim=1) / ep_lambda_discounts.flip(1),
                dims=[1])
            # δ_0 + (γλ)δ_1 + (γλ)^2δ_2 + (γλ)^3δ_3 + (γλ)^4δ_4,
            # δ_1 + (γλ)δ_2 + (γλ)^2δ_3 + (γλ)^3δ_4,
            # δ_2 + (γλ)δ_3 + (γλ)^2δ_4,
            # δ_3 + (γλ)δ_4,
            # δ_4,
            # /

        epx_indexes = ep_indexes[:, :-1]
        epx_obses_list = [bn_obses[:, :-1] for bn_obses in ep_obses_list]
        epx_actions = ep_actions[:, :-1]
        epx_discounted_returns = ep_discounted_returns[:, :-1]
        epx_advantages = ep_advantages[:, :-1]
        epx_probs = ep_probs[:, :-1]

        self.episode_buffer.append((epx_indexes,
                                    epx_obses_list,
                                    epx_actions,
                                    epx_discounted_returns,
                                    epx_advantages.detach(),
                                    epx_probs))

    @unified_elapsed_timer('train a step', 10)
    def train(self) -> int:
        step = self.get_global_step()

        if len(self.episode_buffer) < 16:
            self._profiler('train a step').ignore()
            return step

        with self._profiler('train', repeat=10):
            self._train_ppo_episode()
            self.episode_buffer.clear()

        if step % self.save_model_per_step == 0:
            self.save_model()

        step = self.increase_global_step()

        return step

    def _train_ppo_episode(self) -> None:
        if self.global_step % self.update_target_per_step == 0:
            self._update_target_variables(tau=self.tau)

        loss_policy, loss_entropy, loss_v = None, None, None

        for i in range(self.ppo_epoch):
            weighted_loss_policy = torch.tensor(0., device=self.device)
            weighted_loss_v = torch.tensor(0., device=self.device)
            weighted_loss_entropy = torch.tensor(0., device=self.device)
            total_weight = torch.tensor(0., device=self.device)

            for (ep_indexes,
                 ep_obses_list,
                 ep_actions,
                 ep_discounted_returns,
                 ep_advantages,
                 ep_old_probs) in self.episode_buffer:
                # ep_indexes: [1, ep_len - 1]
                # ep_obses_list: list([1, ep_len - 1, *obs_shapes_i], ...)
                # ep_actions: [1, ep_len - 1, action_size]
                # ep_discounted_returns: [1, ep_len - 1]
                # ep_advantages: [1, ep_len - 1]
                # ep_old_probs: [1, ep_len - 1, action_size]

                ep_states, _ = self.get_l_states(l_indexes=ep_indexes,
                                                 l_obses_list=ep_obses_list,
                                                 l_pre_actions=gen_n_pre_actions(ep_actions),
                                                 l_pre_seq_hidden_states=None,
                                                 is_target=False)

                ep_d_policy, ep_c_policy = self.model_policy(ep_states, ep_obses_list)

                ep_actions = torch.clamp(ep_actions, -0.999, 0.999)
                ep_log_probs = squash_correction_log_prob(ep_c_policy, torch.atanh(ep_actions[..., self.d_action_summed_size:]))  # [1, ep_len - 1, c_action_size]
                log_prob = ep_log_probs.squeeze(0)  # [ep_len - 1, action_size]

                ep_entropys = sum_entropy(ep_c_policy.entropy())  # [1, ep_len - 1]
                entropy = ep_entropys.squeeze(0)  # [ep_len - 1]

                old_log_prob = torch.log(ep_old_probs.squeeze(0) + 1e-8)  # [ep_len - 1, action_size]
                ratio = torch.exp(log_prob - old_log_prob)  # [ep_len - 1, action_size]

                advantage = ep_advantages.squeeze(0).unsqueeze(-1)  # [ep_len - 1, 1]
                clipped_ratio = torch.clamp(ratio, 1. - self.clip_epsilon, 1. + self.clip_epsilon)  # [ep_len - 1, action_size]
                loss_policy = -torch.min(ratio * advantage, clipped_ratio * advantage).sum()

                ep_vs = self.model_v(ep_states, ep_obses_list)  # [1, ep_len - 1, 1]
                v = ep_vs.squeeze(0).squeeze(-1)  # [ep_len - 1]

                discounted_returns = ep_discounted_returns.squeeze(0)  # [ep_len - 1]
                loss_v = functional.mse_loss(v, discounted_returns, reduction='sum')

                loss_entropy = entropy.sum()

                weighted_loss_policy += loss_policy
                weighted_loss_v += loss_v
                weighted_loss_entropy += loss_entropy
                total_weight += ep_indexes.shape[1]

            if total_weight == 0:
                break

            loss_policy = weighted_loss_policy / total_weight
            loss_v = weighted_loss_v / total_weight
            loss_entropy = weighted_loss_entropy / total_weight

            loss = loss_policy + 0.5 * loss_v - 0.001 * loss_entropy

            if self.optimizer_rep is not None:
                self.optimizer_rep.zero_grad()
            self.optimizer_policy.zero_grad()
            self.optimizer_v.zero_grad()

            loss.backward()

            if self.optimizer_rep is not None:
                self.optimizer_rep.step()
            self.optimizer_policy.step()
            self.optimizer_v.step()

        if self.summary_writer is not None and self.global_step % self.write_summary_per_step == 0:
            self.summary_available = True

            if loss_policy is not None:
                with torch.no_grad():
                    self.summary_writer.add_scalar('loss/policy', loss_policy, self.global_step)
                    self.summary_writer.add_scalar('loss/value', loss_v, self.global_step)
                    self.summary_writer.add_scalar('metric/entropy', loss_entropy, self.global_step)

            self.summary_writer.flush()

    def close(self):
        self._closed = True
