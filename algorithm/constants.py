GET_EPISODE_TD_ERROR_SEG = 1024

MAX_THREAD_WORKERS = 500  # Max workers number a grpc server can connect
MAX_RECONNECT_BACKOFF_MS = 500  # Max reconnection waiting time for stubs
MAX_MESSAGE_LENGTH = 1024 * 1024 * 1024  # Max data size grpc server and client transfer
RECONNECTION_TIME = 2  # If the connection is lost, the reconnection time
PING_INTERVAL = 5  # Interval time between two pings
RPC_ERR_RETRY = 5  # Max retry times when rpc error encountered

EPISODE_QUEUE_SIZE = 5
LEARNER_BATCH_DATA_BUFFER_SIZE = 10
LEARNER_PROCESS_EPISODE_THREAD_NUM = 2