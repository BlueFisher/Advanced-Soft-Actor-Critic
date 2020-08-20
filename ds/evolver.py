import logging
import time
import random
import string
import os
from pathlib import Path
from concurrent import futures
import grpc

from .proto import evolver_pb2, evolver_pb2_grpc
from .proto.ndarray_pb2 import Empty
from .proto.numproto import ndarray_to_proto, proto_to_ndarray
from .proto.pingpong_pb2 import Ping, Pong
from .utils import PeerSet, rpc_error_inspector


import algorithm.config_helper as config_helper

MAX_THREAD_WORKERS = 64


class Evolver:
    def __init__(self, root_dir, config_dir, args):
        self._now = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))

        (self.config,
         self.net_config,
         model_abs_dir,
         config_abs_dir) = self._init_config(root_dir, config_dir, args)

        try:
            self._run()
        except KeyboardInterrupt:
            self.logger.warning('KeyboardInterrupt in _run')
            self.close()

    def _init_config(self, root_dir, config_dir, args):
        config_abs_dir = Path(root_dir).joinpath(config_dir)
        config_abs_path = config_abs_dir.joinpath('config_ds.yaml')
        default_config_file_path = Path(__file__).resolve().parent.joinpath('default_config.yaml')
        config = config_helper.initialize_config_from_yaml(default_config_file_path,
                                                           config_abs_path,
                                                           args.config)
        if args.name is not None:
            config['base_config']['name'] = args.name

        # Replace {time} from current time and random letters
        rand = ''.join(random.sample(string.ascii_letters, 4))
        config['base_config']['name'] = config['base_config']['name'].replace('{time}', self._now + rand)
        model_abs_dir = Path(root_dir).joinpath(f'models/{config["base_config"]["scene"]}/{config["base_config"]["name"]}')
        os.makedirs(model_abs_dir)

        logger_file = f'{model_abs_dir}/{args.logger_file}' if args.logger_file is not None else None
        self.logger = config_helper.set_logger('ds.learner', logger_file)

        config_helper.save_config(config, model_abs_dir, 'config_ds.yaml')

        config_helper.display_config(config, self.logger)

        return (config['base_config'],
                config['net_config'],
                model_abs_dir,
                config_abs_dir)

    def _run(self):
        servicer = EvolverService(self.config['name'])
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=MAX_THREAD_WORKERS))
        evolver_pb2_grpc.add_EvolverServiceServicer_to_server(servicer, self.server)
        self.server.add_insecure_port(f'[::]:{self.net_config["evolver_port"]}')
        self.server.start()
        self.logger.info(f'Evolver server is running on [{self.net_config["evolver_port"]}]...')

        self.server.wait_for_termination()

    def close(self):
        self.server.stop(0)

        self.logger.warn('Closed')


class EvolverService(evolver_pb2_grpc.EvolverServiceServicer):
    def __init__(self, name):
        self._peer_set = PeerSet(logging.getLogger('ds.evolver.service'))
        self.name = name

    def _record_peer(self, context):
        def _unregister_peer():
            self._peer_set.disconnect(context.peer())
        context.add_callback(_unregister_peer)
        self._peer_set.connect(context.peer())

    def Persistence(self, request_iterator, context):
        self._record_peer(context)
        for request in request_iterator:
            yield Pong(time=int(time.time() * 1000))

    def Register(self, request: evolver_pb2.RegisterRequest, context):
        self._peer_set.add_info(context.peer(), {
            'host': request.host,
            'port': request.port
        })

        return evolver_pb2.RegisterResponse(name=self.name)

    def PostRewards(self, request: evolver_pb2.PostRewardsToEvolverRequest, context):
        print(proto_to_ndarray(request.rewards))

        return Empty()
