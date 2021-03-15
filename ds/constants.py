MAX_THREAD_WORKERS = 500  # Max workers number a grpc server can connect
MAX_RECONNECT_BACKOFF_MS = 500  # Max reconnection waiting time for stubs
MAX_MESSAGE_LENGTH = 1024 * 1024 * 1024  # Max data size grpc server and client transfer
RECONNECTION_TIME = 2  # If the connection is lost, the reconnection time
PING_INTERVAL = 5  # Interval time between two pings
RPC_ERR_RETRY = 5  # Max retry times when rpc error encountered

EVALUATION_INTERVAL = 10  # If learner is standalone, the interval time between evaluations
EVALUATION_WAITING_TIME = 1  # If learner is not training, the waiting time for the next evaluation

UPDATE_DATA_BUFFER_MAXSIZE = 20
UPDATE_DATA_BUFFER_THREADS = 3

GET_SAMPLED_DATA_QUEUE_SIZE = 5
GET_SAMPLED_DATA_THREAD_SIZE = 5
UPDATE_TD_ERROR_QUEUE_SIZE = 5
UPDATE_TD_ERROR_THREAD_SIZE = 3
UPDATE_TRANSITION_QUEUE_SIZE = 5
UPDATE_TRANSITION_THREAD_SIZE = 3
