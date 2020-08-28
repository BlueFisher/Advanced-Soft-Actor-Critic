from io import BytesIO

import numpy as np

from . import ndarray_pb2


def ndarray_to_proto(nda: np.ndarray) -> ndarray_pb2.NDarray:
    if nda is None:
        nda = np.empty(0)

    nda_bytes = BytesIO()
    np.save(nda_bytes, nda, allow_pickle=False)

    return ndarray_pb2.NDarray(data=nda_bytes.getvalue())


def proto_to_ndarray(nda_proto: ndarray_pb2.NDarray) -> np.ndarray:
    nda_bytes = BytesIO(nda_proto.data)

    nda = np.load(nda_bytes, allow_pickle=False)
    if nda.shape[0] == 0:
        nda = None

    return nda
