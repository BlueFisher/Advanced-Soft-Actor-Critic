from io import BytesIO
import numpy as np
import ndarray_pb2
import ndarray_pb2_grpc


def ndarray_to_proto(nda: np.ndarray) -> ndarray_pb2.NDarray:
    nda_bytes = BytesIO()
    np.save(nda_bytes, nda, allow_pickle=False)

    return ndarray_pb2.NDarray(data=nda_bytes.getvalue())


def proto_to_ndarray(nda_proto: ndarray_pb2.NDarray) -> np.ndarray:
    nda_bytes = BytesIO(nda_proto.data)

    return np.load(nda_bytes, allow_pickle=False)
