#cython: language_level=3

import cuquantum.densitymat as cudense
from cuquantum.densitymat import Operator

from qutip.core.cy.qobjevo cimport QobjEvo
from qutip.core.data cimport Data
from qutip.settings import settings
from qutip import Qobj

from .state import zeros_like_cuState, CuState
from .operator import CuOperator
from .callable import wrap_coeff, wrap_funcelement


# TODO: Being child class of QobjEvo needed? Or duck typing good enough?
cdef class CuQobjEvo(QobjEvo):
    """
    A QobjEvo using cuDensity's Operator instead of qutip's cython operations.
    Designed to use in solver only, as a static object

    It only support list based `QobjEvo`.
    """
    cdef:
        dict __dict__
        # object operator
        # tuple hilbert_space_dims
        # bint action_ready
        # bint expect_ready

    def __init__(self, qobjevo):
        qobjevo = qobjevo.to(CuOperator)
        as_list = qobjevo.to_list()
        self._dims = qobjevo._dims
        self.shape = qobjevo.shape
        self.action_ready = False
        self.expect_ready = False
        if qobjevo.issuper:
            self.hilbert_space_dims = tuple(self.dims[0][0])
        else:
            self.hilbert_space_dims = tuple(self.dims[0])

        self.operator = Operator(self.hilbert_space_dims)
        dual = qobjevo._dims.issuper

        for part in as_list:
            if isinstance(part, Qobj):
                self.operator.append(part.data.to_OperatorTerm(
                    dual, hilbert_dims=self.hilbert_space_dims
                ))
            elif (
                isinstance(part, list) and isinstance(part[0], Qobj)
            ):
                qobj = part[0]
                coeff = wrap_coeff(part[1])
                self.operator.append(qobj.data.to_OperatorTerm(
                    dual, hilbert_dims=self.hilbert_space_dims
                ), coeff)
            else:
                oper = wrap_funcelement(*part, dual, self.hilbert_space_dims)
                self.operator.append(oper)


    cpdef Data matmul_data(CuQobjEvo self, object t, Data state, Data out=None):
        if not isinstance(state, CuState):
            state = CuState(state, hilbert_dims=self.hilbert_space_dims)
        if not self.action_ready:
            self.operator.prepare_action(
                settings.cuDensity["ctx"],
                state.base
            )
            self.action_ready = True
        if out is None:
            out = zeros_like_cuState(state)
        self.operator.compute_action(
            t,
            None,
            state.base,
            out.base,
        )
        return out

    cpdef object expect_data(CuQobjEvo self, object t, Data state):
        if not isinstance(state, CuState):
            state = CuState(state, hilbert_dims=self.hilbert_space_dims)
        if not self.expect_ready:
            self.operator.prepare_expectation(
                settings.cuDensity["ctx"],
                state.base
            )
            self.expect_ready = True
        # Workaround for a bug in cudensity 0.2.0.
        settings.cuDensity["ctx"].release_workspace()
        return self.operator.compute_expectation(t, None, state.base)

    def arguments(self, args):
        raise NotImplementedError

    def linear_map(self, op_mapping, *, _skip_check=False):
        raise NotImplementedError

    def tidyup(self, atol=1e-12):
        raise NotImplementedError

    def to(self, data_type):
        raise NotImplementedError

    def dag(self):
        raise NotImplementedError

    def conj(self):
        raise NotImplementedError

    def trans(self):
        raise NotImplementedError

    @property
    def dtype(self):
        return Operator

    @property
    def num_elements(self):
        raise NotImplementedError

    @property
    def isconstant(self):
        raise NotImplementedError

    def __call__(self, t, _args=None, **kwargs):
        raise NotImplementedError

    def data(self, t):
        raise NotImplementedError


import qutip.core.data as _data


@_data.matmul.register(CuOperator, CuState, CuState)
def matmul_cuoperator_custate_custate(left, right, scale=1., out=None):
    if left.shape[1] == right.shape[0]:
        dual = False
    elif left.shape[1] == right.shape[0] * right.shape[1]:
        dual = True
    else:
        raise TypeError(...)

    if left._oper is None:
        left._oper = Operator(left.to_OperatorTerm(dual=dual))

    left._oper.prepare_action(settings.cuDensity["ctx"], right.base)
    if out is None:
        out = zeros_like_cuState(right)

    left._oper.compute_action(0, state=right.base, state_out=out.base)

    return out
