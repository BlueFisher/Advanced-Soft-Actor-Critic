MAX_THREAD_WORKERS = 500  # Max workers number a grpc server can connect
MAX_RECONNECT_BACKOFF_MS = 500  # Max reconnection waiting time for stubs
MAX_MESSAGE_LENGTH = 1024 * 1024 * 1024  # Max data size grpc server and client transfer
RECONNECTION_TIME = 2  # If the connection is lost, the reconnection time
PING_INTERVAL = 5  # interval time between two pings

EVALUATION_INTERVAL = 10  # If learner is standalone, the interval time between evaluations
EVALUATION_WAITING_TIME = 1  # if learner is not training, the waiting time for the next evaluation
