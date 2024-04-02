from .numproto import ndarray_to_proto, proto_to_ndarray
from .ma_variables_pb2 import MAVariables, Variables


def ma_variables_to_proto(ma_variables: dict):
    if ma_variables is None or len(ma_variables) == 0:
        return MAVariables(succeeded=False)

    return MAVariables(succeeded=True, ma_variables={
        n: Variables(variables=[ndarray_to_proto(v) for v in vs])
        for n, vs in ma_variables.items()
    })


def proto_to_ma_variables(proto):
    if not proto.succeeded:
        return None

    return {
        n: [proto_to_ndarray(v) for v in vs.variables]
        for n, vs in proto.ma_variables.items()
    }
