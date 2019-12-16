import glob
from grpc_tools import protoc

for f in glob.glob('ds\\proto\\*.proto'):
    print(f)
    protoc.main((
        '',
        '-I./ds/proto',
        '--python_out=./ds/proto',
        '--grpc_python_out=./ds/proto',
        f,
    ))