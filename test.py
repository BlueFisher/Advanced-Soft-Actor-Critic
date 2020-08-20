import sys
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parent.joinpath('ds/proto')))

import grpc
import threading
import time

import ds.proto.evolver_pb2 as evolver_pb2
import ds.proto.evolver_pb2_grpc as evolver_pb2_grpc
from ds.proto.pingpong_pb2 import Ping, Pong

channel = grpc.insecure_channel(f'localhost:60002')
stub = evolver_pb2_grpc.EvolverServiceStub(channel)

connected = False


def _start_replay_persistence():
    global connected
    def request_messages():
        while True:
            yield Ping(time=int(time.time() * 1000))
            time.sleep(1)

    while True:
        try:
            reponse_iterator = stub.Persistence(request_messages())
            for response in reponse_iterator:
                if not connected:
                    connected = True
                    print('Replay connected')
        except grpc.RpcError:
            if connected:
                connected = False
                print('Replay disconnected')
        finally:
            time.sleep(1)


t = threading.Thread(target=_start_replay_persistence)
t.start()

t1 = threading.Thread(target=_start_replay_persistence)
t1.start()


for i in range(100):
    stub.Register(evolver_pb2.RegisterRequest(
        host='10',
        port=5
    ))
    time.sleep(1)

t.join()
t1.join()