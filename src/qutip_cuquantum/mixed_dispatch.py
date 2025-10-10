import qutip.core.data as _data
from qutip import settings
from .state import zeros_like_cuState, CuState
from .operator import CuOperator

import cuquantum.densitymat as cudense
from cuquantum.densitymat import Operator

@_data.matmul.register(CuOperator, CuState, CuState)
def matmul_cuoperator_custate_custate(left, right, scale=1., out=None):
    if left.shape[1] == right.shape[0]:
        dual = False
    elif left.shape[1] == right.shape[0] * right.shape[1]:
        dual = True
    else:
        raise TypeError(...)

    oper = Operator(left.hilbert_space_dims, [left.to_OperatorTerm(dual=dual)])

    oper.prepare_action(settings.cuDensity["ctx"], right.base)
    if out is None:
        out = zeros_like_cuState(right)

    oper.compute_action(0, [], state_in=right.base, state_out=out.base)

    return out
