from .numproto import ndarray_to_proto, proto_to_ndarray
from .ma_variables_pb2 import MAVariables, Variables


def ma_variables_to_proto(ma_variables: dict | None):
    if ma_variables is None:
        return MAVariables()

    _ma_variables = {}
    for n, vs in ma_variables.items():
        if vs is not None:
            _ma_variables[n] = Variables(succeeded=True,
                                         variables=[ndarray_to_proto(v) for v in vs])
        else:
            _ma_variables[n] = Variables(succeeded=False)

    return MAVariables(ma_variables=_ma_variables)


def proto_to_ma_variables(proto):
    return {
        n: [proto_to_ndarray(v) for v in vs.variables] if vs.succeeded else None
        for n, vs in proto.ma_variables.items()
    }
