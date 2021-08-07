import importlib
import json
import logging
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import shutil
import socket
import sys
import threading
import time
from concurrent import futures
from pathlib import Path
from typing import List

import grpc
import numpy as np

import algorithm.config_helper as config_helper
import algorithm.constants as C
from algorithm.agent import Agent
from algorithm.utils import (EnvException, ReadWriteLock, RLock,
                             UselessEpisodeException)

from .learner_trainer import Trainer
from .proto import evolver_pb2, evolver_pb2_grpc, learner_pb2, learner_pb2_grpc
from .proto.ndarray_pb2 import Empty
from .proto.numproto import ndarray_to_proto, proto_to_ndarray
from .proto.pingpong_pb2 import Ping, Pong
from .sac_ds_base import SAC_DS_Base
from .utils import PeerSet, rpc_error_inspector, traverse_lists


class Learner:
    _agent_class = Agent
    _policy_variables_cache = None

    def __init__(self, root_dir, config_dir, args):
        self.root_dir = root_dir
        self.cmd_args = args

        self._closed = False
        self._registered = False

        self._logger = logging.getLogger('ds.learner')

        constant_config, config_abs_dir = self._init_constant_config(root_dir, config_dir, args)
        learner_host = constant_config['net_config']['learner_host']
        learner_port = constant_config['net_config']['learner_port']

        self._evolver_stub = EvolverStubController(
            constant_config['net_config']['evolver_host'],
            constant_config['net_config']['evolver_port']
        )

        _id, name, reset_config, sac_config = self._register_evolver(learner_host,
                                                                     learner_port)
        constant_config['base_config']['name'] = name
        constant_config['reset_config'] = reset_config
        constant_config['sac_config'] = sac_config
        self._registered = True

        self._init_config(_id, constant_config)
        self._init_env()
        self._init_sac(config_abs_dir)

        threading.Thread(target=self._policy_evaluation, daemon=True).start()

        self._run_learner_server(learner_port)

        self.close()

    def _init_constant_config(self, root_dir, config_dir, args):
        default_config_abs_path = Path(__file__).resolve().parent.joinpath('default_config.yaml')
        config_abs_dir = Path(root_dir).joinpath(config_dir)
        config_abs_path = config_abs_dir.joinpath('config_ds.yaml')
        config = config_helper.initialize_config_from_yaml(default_config_abs_path,
                                                           config_abs_path,
                                                           args.config)

        # Initialize config from command line arguments
        self.device = args.device
        self.last_ckpt = args.ckpt
        self.render = args.render
        self.run_in_editor = args.editor

        if args.evolver_host is not None:
            config['net_config']['evolver_host'] = args.evolver_host
        if args.evolver_port is not None:
            config['net_config']['evolver_port'] = args.evolver_port
        if args.learner_host is not None:
            config['net_config']['learner_host'] = args.learner_host
        if args.learner_port is not None:
            config['net_config']['learner_port'] = args.learner_port

        if config['net_config']['evolver_host'] is None:
            self._logger.fatal('evolver_host is None')

        # If learner_host is not set, use ip as default
        if config['net_config']['learner_host'] is None:
            hostname = socket.gethostname()
            ip = socket.gethostbyname(hostname)
            config['net_config']['learner_host'] = ip

        return config, config_abs_dir

    def _register_evolver(self, learner_host, learner_port):
        # Get name from evolver registry
        register_response = self._evolver_stub.register_to_evolver(learner_host,
                                                                   learner_port)

        (_id, name,
         reset_config, sac_config) = register_response

        self._logger.info(f'Registered id: {_id}, name: {name}')

        return _id, name, reset_config, sac_config

    def _init_config(self, _id, config):
        if self.cmd_args.name is not None:
            config['base_config']['name'] = self.cmd_args.name
        if self.cmd_args.build_port is not None:
            config['base_config']['build_port'] = self.cmd_args.build_port
        if self.cmd_args.nn is not None:
            config['base_config']['nn'] = self.cmd_args.nn
        if self.cmd_args.agents is not None:
            config['base_config']['n_agents'] = self.cmd_args.agents

        self.config = config
        self.base_config = config['base_config']
        self.reset_config = config['reset_config']
        self.sac_config = config['sac_config']

        model_abs_dir = Path(self.root_dir).joinpath('models',
                                                     self.base_config['scene'],
                                                     self.base_config['name'],
                                                     f'learner{_id}')
        model_abs_dir.mkdir(parents=True, exist_ok=True)
        self.model_abs_dir = model_abs_dir

        if self.cmd_args.logger_in_file:
            config_helper.set_logger(Path(model_abs_dir).joinpath(f'learner.log'))

        config_helper.display_config(config, self._logger)

    def _init_env(self):
        if self.base_config['env_type'] == 'UNITY':
            from algorithm.env_wrapper.unity_wrapper import UnityWrapper

            if self.run_in_editor:
                self.env = UnityWrapper()
            else:
                self.env = UnityWrapper(file_name=self.base_config['build_path'][sys.platform],
                                        base_port=self.base_config['build_port'],
                                        no_graphics=self.base_config['no_graphics'],
                                        scene=self.base_config['scene'],
                                        additional_args=self.cmd_args.additional_args,
                                        n_agents=self.base_config['n_agents'])

        elif self.base_config['env_type'] == 'GYM':
            from algorithm.env_wrapper.gym_wrapper import GymWrapper

            self.env = GymWrapper(env_name=self.base_config['build_path'],
                                  n_agents=self.base_config['n_agents'])
        else:
            raise RuntimeError(f'Undefined Environment Type: {self.base_config["env_type"]}')

        self.obs_shapes, self.d_action_size, self.c_action_size = self.env.init()
        self.action_size = self.d_action_size + self.c_action_size

        self._logger.info(f'{self.base_config["build_path"]} initialized')

    def _init_sac(self, config_abs_dir):
        # If model exists, load saved model, or copy a new one
        nn_model_abs_path = Path(self.model_abs_dir).joinpath('nn_models.py')
        if nn_model_abs_path.exists():
            spec = importlib.util.spec_from_file_location('nn', str(nn_model_abs_path))
        else:
            nn_abs_path = Path(config_abs_dir).joinpath(f'{self.base_config["nn"]}.py')
            spec = importlib.util.spec_from_file_location('nn', str(nn_abs_path))
            shutil.copyfile(nn_abs_path, nn_model_abs_path)

        custom_nn_model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_nn_model)

        self._sac_bak_lock = ReadWriteLock(None, 2, 2, logger=self._logger)

        self.sac_bak = SAC_DS_Base(obs_shapes=self.obs_shapes,
                                   d_action_size=self.d_action_size,
                                   c_action_size=self.c_action_size,
                                   model_abs_dir=self.model_abs_dir,
                                   model=custom_nn_model,
                                   device=self.device,
                                   summary_path='log/bak',
                                   train_mode=False,
                                   last_ckpt=self.last_ckpt,

                                   **self.sac_config)

        self._logger.info('SAC_BAK started')

        # Initialize all queues for learner_trainer
        self._all_variables_buffer = self.sac_bak.get_all_variables()
        self.all_variables_avaiable_queue = mp.Queue(1)  # sac_bak get all variables from sac
        self.all_variables_shms = traverse_lists(self._all_variables_buffer,
                                                 lambda v: SharedMemory(create=True, size=v.nbytes))
        self.episode_queue = mp.Queue(C.EPISODE_QUEUE_SIZE)  # actors send episode to sac
        self.cmd_pipe_client, cmd_pipe_server = mp.Pipe()

        self.learner_trainer_process = mp.Process(target=Trainer, kwargs={
            'all_variables_avaiable_queue': self.all_variables_avaiable_queue,
            'all_variables_shms': self.all_variables_shms,
            'episode_queue': self.episode_queue,
            'cmd_pipe_server': cmd_pipe_server,

            'logger_in_file': self.cmd_args.logger_in_file,
            'obs_shapes': self.obs_shapes,
            'd_action_size': self.d_action_size,
            'c_action_size': self.c_action_size,
            'model_abs_dir': self.model_abs_dir,
            'model_spec': spec,
            'device': self.device,
            'last_ckpt': self.last_ckpt,
            'config': self.config
        })
        self.learner_trainer_process.start()

        nn_variables = self._evolver_stub.get_nn_variables()
        if nn_variables:
            self._udpate_nn_variables(nn_variables)
            self._logger.info(f'Initialized from evolver')

        self._logger.info('Waiting for updating sac_bak in first time')
        self._update_sac_bak()

        threading.Thread(target=self._forever_update_sac_bak, daemon=True).start()

    def _update_sac_bak(self):
        self.all_variables_avaiable_queue.get()  # Block, waiting for all variables available

        def _tra(v, shm):
            if len(v.shape) == 0:
                return
            shm_np = np.ndarray(v.shape, dtype=v.dtype, buffer=shm.buf)
            v[:] = shm_np[:]
        traverse_lists((self._all_variables_buffer, self.all_variables_shms), _tra)

        with self._sac_bak_lock.write():
            res = self.sac_bak.update_all_variables(self._all_variables_buffer)

        if not res:
            self._logger.warning('NAN in variables, closing...')
            self._force_close()
            return

        self._policy_variables_cache = self.sac_bak.get_policy_variables()

        self._logger.info('Updated sac_bak')

    def _forever_update_sac_bak(self):
        while True:
            self._update_sac_bak()

    def _get_actor_register_result(self, actors_num):
        if self._registered:

            noise = self.base_config['noise_increasing_rate'] * (actors_num - 1)
            actor_sac_config = self.sac_config
            actor_sac_config['noise'] = min(noise, self.base_config['noise_max'])

            return (str(self.model_abs_dir),
                    self.reset_config,
                    actor_sac_config)

    def _get_policy_variables(self):
        return self._policy_variables_cache

    def _get_nn_variables(self):
        self.cmd_pipe_client.send(('GET', None))
        nn_variables = self.cmd_pipe_client.recv()

        return nn_variables

    def _udpate_nn_variables(self, variables):
        self.cmd_pipe_client.send(('UPDATE', variables))

    def _save_model(self):
        self.cmd_pipe_client.send(('SAVE_MODEL', None))

    def _get_action(self, obs_list, rnn_state=None):
        if self.sac_bak.use_rnn:
            assert rnn_state is not None

        with self._sac_bak_lock.read():
            if self.sac_bak.use_rnn:
                action, next_rnn_state = self.sac_bak.choose_rnn_action(obs_list, rnn_state)
                next_rnn_state = next_rnn_state
                return action, next_rnn_state
            else:
                action = self.sac_bak.choose_action(obs_list)
                return action

    def _add_episode(self,
                     n_obses_list: List[np.ndarray],
                     n_actions: np.ndarray,
                     n_rewards: np.ndarray,
                     next_obs_list: List[np.ndarray],
                     n_dones: np.ndarray,
                     n_mu_probs: np.ndarray,
                     n_rnn_states: np.ndarray = None):

        if self.episode_queue.full():
            self.episode_queue.get()

        self.episode_queue.put((
            n_obses_list,
            n_actions,
            n_rewards,
            next_obs_list,
            n_dones,
            n_mu_probs,
            n_rnn_states
        ))

    def _policy_evaluation(self):
        use_rnn = self.sac_bak.use_rnn

        obs_list = self.env.reset(reset_config=self.reset_config)

        agents = [self._agent_class(i, use_rnn=use_rnn)
                  for i in range(self.base_config['n_agents'])]

        if use_rnn:
            initial_rnn_state = self.sac_bak.get_initial_rnn_state(len(agents))
            rnn_state = initial_rnn_state

        force_reset = False
        iteration = 0
        start_time = time.time()

        while not self._closed:
            if self.base_config['reset_on_iteration'] or any([a.max_reached for a in agents]) or force_reset:
                obs_list = self.env.reset(reset_config=self.reset_config)
                for agent in agents:
                    agent.clear()

                if use_rnn:
                    rnn_state = initial_rnn_state

                force_reset = False
            else:
                for agent in agents:
                    agent.reset()

            action = np.zeros([len(agents), self.action_size], dtype=np.float32)
            step = 0

            try:
                while not all([a.done for a in agents]) and not self._closed:
                    with self._sac_bak_lock.read('choose_action'):
                        if use_rnn:
                            action, next_rnn_state = self.sac_bak.choose_rnn_action([o.astype(np.float32) for o in obs_list],
                                                                                    action,
                                                                                    rnn_state)

                            if np.isnan(np.min(next_rnn_state)):
                                raise UselessEpisodeException()
                        else:
                            action = self.sac_bak.choose_action([o.astype(np.float32) for o in obs_list])

                    next_obs_list, reward, local_done, max_reached = self.env.step(action[..., :self.d_action_size],
                                                                                   action[..., self.d_action_size:])

                    if step == self.base_config['max_step_each_iter']:
                        local_done = [True] * len(agents)
                        max_reached = [True] * len(agents)

                    for i, agent in enumerate(agents):
                        agent.add_transition([o[i] for o in obs_list],
                                             action[i],
                                             reward[i],
                                             local_done[i],
                                             max_reached[i],
                                             [o[i] for o in next_obs_list],
                                             rnn_state[i] if use_rnn else None)

                    obs_list = next_obs_list
                    action[local_done] = np.zeros(self.action_size)
                    if use_rnn:
                        rnn_state = next_rnn_state
                        rnn_state[local_done] = initial_rnn_state[local_done]

                    step += 1

            except EnvException as e:
                self._logger.error(e)
                self.env.close()
                self._logger.info(f'Restarting {self.base_config["build_path"]}...')
                self._init_env()
                continue

            except UselessEpisodeException:
                self._logger.warning('Useless episode')
                force_reset = True

                if self.base_config['evolver_enabled']:
                    self._evolver_stub.post_reward(float('-inf'))

                continue

            except Exception as e:
                self._logger.error(e)
                self._logger.error('Exiting...')
                break

            self._log_episode_summaries(agents, iteration)
            self._log_episode_info(iteration, start_time, agents)

            if (p := self.model_abs_dir.joinpath('save_model')).exists():
                self._save_model()
                p.unlink()

            if self.base_config['evolver_enabled']:
                self._evolver_stub.post_reward(np.mean([a.reward for a in agents]))

            iteration += 1

        self._logger.warning('Evaluation exits')

    def _log_episode_summaries(self, agents, iteration):
        rewards = np.array([a.reward for a in agents])
        self.sac_bak.write_constant_summaries([
            {'tag': 'reward/mean', 'simple_value': rewards.mean()},
            {'tag': 'reward/max', 'simple_value': rewards.max()},
            {'tag': 'reward/min', 'simple_value': rewards.min()}
        ], iteration)

    def _log_episode_info(self, iteration, start_time, agents):
        time_elapse = (time.time() - start_time) / 60
        rewards = [a.reward for a in agents]

        rewards = ", ".join([f"{i:6.1f}" for i in rewards])
        steps = [a.steps for a in agents]
        self._logger.info(f'{iteration}, {time_elapse:.2f}m, S {max(steps)}, R {rewards}')

    def _run_learner_server(self, learner_port):
        servicer = LearnerService(self)
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=C.MAX_THREAD_WORKERS),
                                  options=[
            ('grpc.max_send_message_length', C.MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', C.MAX_MESSAGE_LENGTH)
        ])
        learner_pb2_grpc.add_LearnerServiceServicer_to_server(servicer, self.server)
        self.server.add_insecure_port(f'[::]:{learner_port}')
        self.server.start()
        self._logger.info(f'Learner server is running on [{learner_port}]...')

        self.server.wait_for_termination()

    def _force_close(self):
        self._logger.warning('Force closing')
        f = open(self.model_abs_dir.joinpath('force_closed'), 'w')
        f.close()
        self.close()

    def close(self):
        self._closed = True

        if hasattr(self, 'env'):
            self.env.close()
        if hasattr(self, 'server'):
            self.server.stop(None)

        self._evolver_stub.close()
        self.learner_trainer_process.close()

        self._logger.warning('Closed')


class LearnerService(learner_pb2_grpc.LearnerServiceServicer):
    _policy_variables_id_cache = None
    _proto_policy_variables_cache = None

    def __init__(self, learner: Learner):
        self._get_actor_register_result = learner._get_actor_register_result

        self._get_action = learner._get_action
        self._add_episode = learner._add_episode
        self._get_policy_variables = learner._get_policy_variables
        self._get_nn_variables = learner._get_nn_variables
        self._udpate_nn_variables = learner._udpate_nn_variables

        self._force_close = learner._force_close

        self._logger = logging.getLogger('ds.learner.service')
        self._peer_set = PeerSet(self._logger)

        self._lock = RLock(timeout=1, logger=self._logger)
        self._actor_id = 0

    def _record_peer(self, context):
        peer = context.peer()

        def _unregister_peer():
            with self._lock:
                self._logger.warning(f'Actor {peer} disconnected')
                self._peer_set.disconnect(context.peer())

        context.add_callback(_unregister_peer)
        self._peer_set.connect(peer)

    def Persistence(self, request_iterator, context):
        self._record_peer(context)
        for request in request_iterator:
            yield Pong(time=int(time.time() * 1000))

    def RegisterActor(self, request, context):
        peer = context.peer()
        self._logger.info(f'{peer} is registering actor...')

        with self._lock:
            res = self._get_actor_register_result(len(self._peer_set))
            if res is not None:
                actor_id = self._actor_id

                self._peer_set.add_info(peer, {
                    'policy_variables_id_cache': None
                })

                self._logger.info(f'Actor {peer} registered')

                self._actor_id += 1

                (model_abs_dir,
                 reset_config,
                 sac_config) = res
                return learner_pb2.RegisterActorResponse(model_abs_dir=model_abs_dir,
                                                         unique_id=actor_id,
                                                         reset_config_json=json.dumps(reset_config),
                                                         sac_config_json=json.dumps(sac_config))
            else:
                return learner_pb2.RegisterActorResponse(unique_id=-1)

    # From actor
    def GetAction(self, request: learner_pb2.GetActionRequest, context):
        obs_list = [proto_to_ndarray(obs) for obs in request.obs_list]
        rnn_state = proto_to_ndarray(request.rnn_state)

        if rnn_state is None:
            action = self._get_action(obs_list)
            next_rnn_state = None
        else:
            action, next_rnn_state = self._get_action(obs_list, rnn_state)

        return learner_pb2.Action(action=ndarray_to_proto(action),
                                  rnn_state=ndarray_to_proto(next_rnn_state))

    # From actor
    def GetPolicyVariables(self, request, context):
        peer = context.peer()

        variables = self._get_policy_variables()
        if variables is None:
            return learner_pb2.NNVariables(succeeded=False)

        if id(variables) != self._policy_variables_id_cache:
            self._policy_variables_id_cache = id(variables)
            self._proto_policy_variables_cache = [ndarray_to_proto(v) for v in variables]

        actor_info = self._peer_set.get_info(peer)

        if self._policy_variables_id_cache == actor_info['policy_variables_id_cache']:
            return learner_pb2.NNVariables(succeeded=False)

        actor_info['policy_variables_id_cache'] = self._policy_variables_id_cache
        return learner_pb2.NNVariables(succeeded=True,
                                       variables=self._proto_policy_variables_cache)

    # From actor
    def Add(self, request, context):
        self._add_episode([proto_to_ndarray(n_obses) for n_obses in request.n_obses_list],
                          proto_to_ndarray(request.n_actions),
                          proto_to_ndarray(request.n_rewards),
                          [proto_to_ndarray(obs) for obs in request.next_obs_list],
                          proto_to_ndarray(request.n_dones),
                          proto_to_ndarray(request.n_mu_probs),
                          proto_to_ndarray(request.n_rnn_states))

        return Empty()

    # From evolver
    def GetNNVariables(self, request, context):
        variables = self._get_nn_variables()
        return learner_pb2.NNVariables(succeeded=True,
                                       variables=[ndarray_to_proto(v) for v in variables])

    # From evolver
    def UpdateNNVariables(self, request, context):
        variables = [proto_to_ndarray(v) for v in request.variables]
        self._udpate_nn_variables(variables)
        return Empty()

    def ForceClose(self, request, context):
        self._force_close()
        return Empty()


class EvolverStubController:
    _closed = False

    def __init__(self, evolver_host, evolver_port):
        self._logger = logging.getLogger('ds.learner.evolver_stub')

        self._evolver_channel = grpc.insecure_channel(f'{evolver_host}:{evolver_port}', [
            ('grpc.max_reconnect_backoff_ms', C.MAX_RECONNECT_BACKOFF_MS),
            ('grpc.max_send_message_length', C.MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', C.MAX_MESSAGE_LENGTH)
        ])
        self._evolver_stub = evolver_pb2_grpc.EvolverServiceStub(self._evolver_channel)
        self._logger.info(f'Starting evolver stub [{evolver_host}:{evolver_port}]')

        self._evolver_connected = False

        t_evolver = threading.Thread(target=self._start_evolver_persistence, daemon=True)
        t_evolver.start()

    @property
    def connected(self):
        return not self._closed and self._evolver_connected

    @rpc_error_inspector
    def register_to_evolver(self, learner_host, learner_port):
        self._logger.warning('Waiting for evolver connection')
        while not self.connected:
            time.sleep(C.RECONNECTION_TIME)
            continue

        response = None
        self._logger.info('Registering to evolver...')
        while response is None:
            response = self._evolver_stub.RegisterLearner(
                evolver_pb2.RegisterLearnerRequest(learner_host=learner_host,
                                                   learner_port=learner_port))

            if response:
                self._logger.info('Registered to evolver')

                return (response.id, response.name,
                        json.loads(response.reset_config_json),
                        json.loads(response.sac_config_json))
            else:
                response = None
                time.sleep(C.RECONNECTION_TIME)

    @rpc_error_inspector
    def post_reward(self, reward):
        self._evolver_stub.PostReward(
            evolver_pb2.PostRewardToEvolverRequest(reward=float(reward)))

    @rpc_error_inspector
    def get_nn_variables(self):
        response = self._evolver_stub.GetNNVariables(Empty())
        if response.succeeded:
            variables = [proto_to_ndarray(v) for v in response.variables]
            return variables
        else:
            return None

    def _start_evolver_persistence(self):
        def request_messages():
            while not self._closed:
                yield Ping(time=int(time.time() * 1000))
                time.sleep(C.PING_INTERVAL)
                if not self._evolver_connected:
                    break

        while not self._closed:
            try:
                reponse_iterator = self._evolver_stub.Persistence(request_messages())
                for response in reponse_iterator:
                    if not self._evolver_connected:
                        self._evolver_connected = True
                        self._logger.info('Evolver connected')
            except grpc.RpcError:
                if self._evolver_connected:
                    self._evolver_connected = False
                    self._logger.error('Evolver disconnected')
            finally:
                time.sleep(C.RECONNECTION_TIME)

    def close(self):
        self._evolver_channel.close()
        self._closed = True
