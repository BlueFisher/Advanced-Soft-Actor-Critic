import logging
import grpc
from typing import Optional

import socket
from multiprocessing import Pipe
from concurrent.futures import ThreadPoolExecutor

from .communicator import Communicator
from mlagents.envs.communicator_objects.unity_to_external_pb2_grpc import (
    UnityToExternalServicer,
    add_UnityToExternalServicer_to_server,
)
from mlagents.envs.communicator_objects.unity_message_pb2 import UnityMessage
from mlagents.envs.communicator_objects.unity_input_pb2 import UnityInput
from mlagents.envs.communicator_objects.unity_output_pb2 import UnityOutput
from .exception import UnityTimeOutException, UnityWorkerInUseException

logger = logging.getLogger("mlagents.envs")


class UnityToExternalServicerImplementation(UnityToExternalServicer):
    def __init__(self):
        self.parent_conn, self.child_conn = Pipe()

    def Initialize(self, request, context):
        self.child_conn.send(request)
        return self.child_conn.recv()

    def Exchange(self, request, context):
        self.child_conn.send(request)
        return self.child_conn.recv()


class RpcCommunicator(Communicator):
    def __init__(self, worker_id=0, base_port=5005, timeout_wait=30):
        """
        Python side of the grpc communication. Python is the server and Unity the client


        :int base_port: Baseline port number to connect to Unity environment over. worker_id increments over this.
        :int worker_id: Number to add to communication port (5005) [0]. Used for asynchronous agent scenarios.
        """
        self.port = base_port + worker_id
        self.worker_id = worker_id
        self.timeout_wait = timeout_wait
        self.server = None
        self.unity_to_external = None
        self.is_open = False
        self.create_server()

    def create_server(self):
        """
        Creates the GRPC server.
        """
        self.check_port(self.port)

        try:
            # Establish communication grpc
            self.server = grpc.server(ThreadPoolExecutor(max_workers=10))
            self.unity_to_external = UnityToExternalServicerImplementation()
            add_UnityToExternalServicer_to_server(self.unity_to_external, self.server)
            # Using unspecified address, which means that grpc is communicating on all IPs
            # This is so that the docker container can connect.
            self.server.add_insecure_port("[::]:" + str(self.port))
            self.server.start()
            self.is_open = True
        except Exception:
            raise UnityWorkerInUseException(self.worker_id)

    def check_port(self, port):
        """
        Attempts to bind to the requested communicator port, checking if it is already in use.
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("localhost", port))
        except socket.error:
            raise UnityWorkerInUseException(self.worker_id)
        finally:
            s.close()

    def initialize(self, inputs: UnityInput) -> UnityOutput:
        if not self.unity_to_external.parent_conn.poll(self.timeout_wait):
            raise UnityTimeOutException(
                "The Unity environment took too long to respond. Make sure that :\n"
                "\t The environment does not need user interaction to launch\n"
                "\t The Academy's Broadcast Hub is configured correctly\n"
                "\t The Agents are linked to the appropriate Brains\n"
                "\t The environment and the Python interface have compatible versions."
            )
        aca_param = self.unity_to_external.parent_conn.recv().unity_output
        message = UnityMessage()
        message.header.status = 200
        message.unity_input.CopyFrom(inputs)
        self.unity_to_external.parent_conn.send(message)
        self.unity_to_external.parent_conn.recv()
        return aca_param

    def exchange(self, inputs: UnityInput) -> Optional[UnityOutput]:
        message = UnityMessage()
        message.header.status = 200
        message.unity_input.CopyFrom(inputs)
        self.unity_to_external.parent_conn.send(message)
        output = self.unity_to_external.parent_conn.recv()
        if output.header.status != 200:
            return None
        return output.unity_output

    def close(self):
        """
        Sends a shutdown signal to the unity environment, and closes the grpc connection.
        """
        if self.is_open:
            message_input = UnityMessage()
            message_input.header.status = 400
            self.unity_to_external.parent_conn.send(message_input)
            self.unity_to_external.parent_conn.close()
            self.server.stop(False)
            self.is_open = False
