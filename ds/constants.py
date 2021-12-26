MAX_THREAD_WORKERS = 500  # Max workers number a grpc server can connect
MAX_RECONNECT_BACKOFF_MS = 500  # Max reconnection waiting time for stubs
MAX_MESSAGE_LENGTH = 1024 * 1024 * 1024  # Max data size grpc server and client transfer
RECONNECTION_TIME = 2  # If the connection is lost, the reconnection time
PING_INTERVAL = 5  # Interval time between two pings
RPC_ERR_RETRY = 5  # Max retry times when rpc error encountered

ELAPSED_REPEAT = 50

EPISODE_QUEUE_TIMEOUT = 0.5
BATCH_QUEUE_TIMEOUT = 0.5
