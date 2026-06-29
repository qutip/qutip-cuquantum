from typing import Any


import numpy as np
import cupy as cp
import pytest
import random
import numbers
from itertools import product
from utils import StateType, random_custate
cudense = pytest.importorskip("cuquantum.densitymat")

import qutip
from qutip_cuquantum.state import (
    CuState, iadd_cuState, add_cuState, mul_cuState, imul_cuState, l2_cuState,
    frobenius_cuState, trace_cuState, trace_oper_ket_cuState,
    inner_cuState, wrmn_error_cuState,
    transpose_cuState, adjoint_cuState, matmul_cuState, project_CuState
)

import qutip.core.data as _data
import qutip.tests.core.data.test_mathematics as test_tools
import qutip.tests.core.data.test_norm as test_norm



qutip.settings.cuDensity["ctx"] = cudense.WorkStream()



test_tools._ALL_CASES = {
    CuState: lambda shape: [lambda: random_custate(shape),],
}

test_tools._RANDOM = {
    CuState: lambda shape: [lambda: random_custate(shape),],
}

_unary_pure = [
    (pytest.param((2, StateType.KET), id="simple ket"),),
    (pytest.param((6, 6, StateType.KET), id="2 hilbert ket"),),
    (pytest.param((2, 2, 2, StateType.KET), id="complex ket"),),
]

_unary_mixed = [
    (pytest.param((3, StateType.DM), id="scalar dm"),),
    (pytest.param((2, 3, StateType.DM), id="2 hilbert dm"),),
    (pytest.param((2, 3, 4, StateType.DM), id="complex dm"),),
]


_compatible_hilbert = [
    (
        pytest.param((2, StateType.KET), id="simple ket"),
         pytest.param((2, StateType.KET), id="simple ket"),
    ),
    (
        pytest.param((2, 3, StateType.KET), id="2 hilbert ket"),
        pytest.param((2, 3, StateType.KET), id="weak ket"),
    ),
    (
        pytest.param((2, 2, 2, 3, StateType.KET), id="complex ket"),
        pytest.param((2, 2, 2, 3, StateType.KET), id="complex ket"),
    ),
    (
        pytest.param((2, 3, StateType.DM), id="2 hilbert dm"),
        pytest.param((2, 3, StateType.DM), id="2 hilbert dm"),
    ),
    (
        pytest.param((2, 3, 2, StateType.DM), id="3 hilbert dm"),
        pytest.param((2, 3, 2, StateType.DM), id="2 weak hilbert dm"),
    ),
    (
        pytest.param((2, 3, 4, StateType.DM), id="complex dm"),
        pytest.param((2, 3, 4, StateType.DM), id="complex dm"),
    ),
]


_imcompatible_hilbert = [
    (pytest.param((2, StateType.KET), id="simple ket"), pytest.param((2, StateType.DM), id="simple dm"),),
    (pytest.param((2, StateType.KET), id="2 ket"), pytest.param((3, StateType.KET), id="3 ket"),),
    (pytest.param((3, 2, StateType.KET), id="3, 2 ket"), pytest.param((2, 3, StateType.KET), id="2, 3 ket"),),
    (pytest.param((3, 2, StateType.DM), id="3, 2 dm"), pytest.param((2, 3, StateType.DM), id="2, 3 dm"),),
    (pytest.param((2, 4, StateType.DM), id="2, 4 dm"), pytest.param((4 ,2, StateType.DM), id="4, 2 dm"),),
]

# For matmul, we need shapes where left.shape[1] == right.shape[0]
_matmul_compatible = [
    (
        pytest.param((2, 3, StateType.DM), id="2,3 dm"),
        pytest.param((2, 3, StateType.DM), id="2,3 dm"),
    ),
    (
        pytest.param((2, 3, 2, StateType.DM), id="2,3,2 dm"),
        pytest.param((2, 3, 2, StateType.KET), id="2,3,2 ket"),
    ),
    (
        pytest.param((2, 3, StateType.KET), id="2,3 ket"),
        pytest.param((2, 3, StateType.BRA), id="2,3 bra"),
    ),
    (
        pytest.param((2, 3, StateType.BRA), id="2,3 bra"),
        pytest.param((2, 3, StateType.KET), id="2,3 ket"),
    ),
]

_matmul_incompatible = [
    (
        pytest.param((2, 3, 2, StateType.DM), id="2,3,2 dm"),
        pytest.param((2, 3, StateType.KET), id="2,3 ket"),
    ),
    (
        pytest.param((2, 3, 4, StateType.DM), id="2,3,4 dm"),
        pytest.param((2, 12, StateType.DM), id="2,12 dm"),
    ),
    (
        pytest.param((2, 3, StateType.KET), id="2,3 ket"),
        pytest.param((2, 3, StateType.DM), id="2,3 dm"),
    ),
    (
        pytest.param((2, 3, StateType.DM), id="2,3 dm"),
        pytest.param((2, 3, StateType.BRA), id="2,3 bra"),

    ),
]

_kron_hilbert = [
    (
        pytest.param((2, StateType.KET), id="simple ket"),
        pytest.param((3, StateType.KET), id="simple ket"),),
    (
        pytest.param((2, 3, StateType.KET), id="2 hilbert ket"),
        pytest.param((2, StateType.KET), id="simple ket"),),
    (
        pytest.param((2, 4, 3, StateType.KET), id="complex ket"),
        pytest.param((4, 6, StateType.KET), id="complex ket"),),
    (
        pytest.param((2, StateType.DM), id="simple dm"),
        pytest.param((2, 3, StateType.DM), id="2 hilbert dm"),),
    (
        pytest.param((2, 3, 2, StateType.DM), id="3 hilbert dm"),
        pytest.param((2, 6, StateType.DM), id="2 hilbert dm"),),
    (
        pytest.param((2, 3, 4, StateType.DM), id="complex dm"),
        pytest.param((2, 6, 2, StateType.DM), id="complex dm"),
    ),
]


class TestTrace(test_tools.TestTrace):
    specialisations = [
        pytest.param(trace_cuState, CuState, complex),
        # OperKet are not actually stacked in the tensor rep
        pytest.param(trace_oper_ket_cuState, CuState, complex),
    ]

    shapes = _unary_mixed
    bad_shapes = []


class TestAdd(test_tools.TestAdd):
    specialisations = [
        pytest.param(add_cuState, CuState, CuState, CuState),
        pytest.param(iadd_cuState, CuState, CuState, CuState),
    ]

    shapes = _compatible_hilbert
    bad_shapes = _imcompatible_hilbert


class TestWRMN_error(test_tools.TestWRMN_error):
    specialisations = [
        pytest.param(wrmn_error_cuState, CuState, CuState, float),
    ]

    shapes = _compatible_hilbert
    bad_shapes = _imcompatible_hilbert


class TestMul(test_tools.TestMul):
    specialisations = [
        pytest.param(mul_cuState, CuState, CuState),
        pytest.param(imul_cuState, CuState, CuState),
    ]

    shapes = _unary_pure + _unary_mixed
    bad_shapes = []


class TestFrobeniusNorm(test_norm.TestFrobeniusNorm):
    specialisations = [
        pytest.param(frobenius_cuState, CuState, float),
    ]

    shapes = _unary_pure + _unary_mixed
    bad_shapes = []


class TestL2Norm(test_norm.TestL2Norm):
    specialisations = [
        pytest.param(l2_cuState, CuState, float),
    ]

    shapes = _unary_pure
    bad_shapes = _unary_mixed


class TestInner(test_tools.TestInner):
    specialisations = [
        pytest.param(inner_cuState, CuState, CuState, complex),
    ]

    shapes = [(hilbert[0], hilbert[0]) for hilbert in _unary_pure]
    bad_shapes = []


class TestTranspose(test_tools.TestTranspose):
    specialisations = [
        pytest.param(transpose_cuState, CuState, CuState),
    ]

    shapes = _unary_pure + _unary_mixed
    bad_shapes = []


class TestAdjoint(test_tools.TestAdjoint):
    specialisations = [
        pytest.param(adjoint_cuState, CuState, CuState),
    ]

    shapes = _unary_pure + _unary_mixed
    bad_shapes = []


class TestMatmul(test_tools.TestMatmul):
    specialisations = [
        pytest.param(matmul_cuState, CuState, CuState, CuState),
    ]

    shapes = _matmul_compatible
    bad_shapes = _matmul_incompatible


class TestProject(test_tools.TestProject):
    specialisations = [
        pytest.param(project_CuState, CuState, CuState),
    ]

    shapes = _unary_pure
    bad_shapes = _unary_mixed


def test_isherm():
    A = qutip.basis(3, dtype="CuState")
    assert _data.isherm(A.data) == False
    B = qutip.rand_dm(3, dtype="CuState")
    assert _data.isherm(B.data)
    C = qutip.rand_stochastic(5, density=1) @ qutip.rand_dm(5, density=1)
    assert _data.isherm(C.to("CuState").data) == False


def test_conj():
    A = (qutip.basis(3, dtype="CuState") * 0.5j).data
    assert abs(frobenius_cuState(A - A.conj()) - 1.) < 1e-10
    B = (qutip.basis(3, dtype="CuState")).data
    assert abs(frobenius_cuState(B - B.conj())) < 1e-10
