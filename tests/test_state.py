import numpy as np
import cupy as cp
import pytest
import random
import numbers
from itertools import product
cudense = pytest.importorskip("cuquantum.densitymat")

import qutip
from qutip_cuquantum.state import (
    CuState, iadd_cuState, add_cuState, mul_cuState, imul_cuState,
    frobenius_cuState, trace_cuState, inner_cuState, wrmn_error_cuState
)

import qutip.core.data as _data
import qutip.tests.core.data.test_mathematics as test_tools
import qutip.tests.core.data.test_norm as test_norm


qutip.settings.cuDensity["ctx"] = cudense.WorkStream()


def random_pure_custate(hilbert):
    """Generate a random `CuPyDense` matrix with the given shape."""
    N = abs(np.prod(hilbert))
    out = (
        np.random.rand(N, 1) + 1j * np.random.rand(N, 1)
    ).astype(cp.complex128)
    out = qutip.core.data.Dense(out)
    return CuState(out, hilbert, copy=False)


def random_mixed_custate(hilbert):
    """Generate a random `CuPyDense` matrix with the given shape."""
    N = abs(np.prod(hilbert))
    out = (
        np.random.rand(N, N) + 1j * np.random.rand(N, N)
    ).astype(cp.complex128)
    out = qutip.core.data.Dense(out)
    return CuState(out, hilbert, copy=False)


def random_custate(shape):
    *hilbert, pure = shape
    if pure:
        return random_pure_custate(hilbert)
    else:
        return random_mixed_custate(hilbert)


test_tools._ALL_CASES = {
    CuState: lambda shape: [lambda: random_custate(shape),],
}

test_tools._RANDOM = {
    CuState: lambda shape: [lambda: random_custate(shape),],
}

_unary_pure = [
    (pytest.param((2, True), id="simple ket"),),
    (pytest.param((6, 6, True), id="2 hilbert ket"),),
    (pytest.param((2, 2, 2, True), id="complex ket"),),
]

_unary_mixed = [
    (pytest.param((3, False), id="scalar dm"),),
    (pytest.param((2, 3, False), id="2 hilbert dm"),),
    (pytest.param((2, 3, 4, False), id="complex dm"),),
]


_compatible_hilbert = [
    (
        pytest.param((2, True), id="simple ket"), 
         pytest.param((2, True), id="simple ket"),
    ),
    (
        pytest.param((2, 3, True), id="2 hilbert ket"), 
        pytest.param((2, 3, True), id="weak ket"),
    ),
    (
        pytest.param((2, 2, 2, 3, True), id="complex ket"), 
        pytest.param((2, 2, 2, 3, True), id="complex ket"),
    ),
    (
        pytest.param((2, 3, False), id="2 hilbert dm"),
        pytest.param((2, 3, False), id="2 hilbert dm"),
    ),
    (
        pytest.param((2, 3, 2, False), id="3 hilbert dm"),
        pytest.param((2, 3, 2, False), id="2 weak hilbert dm"),
    ),
    (
        pytest.param((2, 3, 4, False), id="complex dm"),
        pytest.param((2, 3, 4, False), id="complex dm"),
    ),
]


_imcompatible_hilbert = [
    (pytest.param((2, True), id="simple ket"), pytest.param((2, False), id="simple dm"),),
    (pytest.param((2, True), id="2 ket"), pytest.param((3, True), id="3 ket"),),
    (pytest.param((3, 2, True), id="3, 2 ket"), pytest.param((2, 3, True), id="2, 3 ket"),),
    (pytest.param((3, 2, False), id="3, 2 dm"), pytest.param((2, 3, False), id="2, 3 dm"),),
    (pytest.param((2, 4, False), id="2, 4 dm"), pytest.param((4 ,2, False), id="4, 2 dm"),),
]

_kron_hilbert = [
    (
        pytest.param((2, True), id="simple ket"),
        pytest.param((3, True), id="simple ket"),),
    (
        pytest.param((2, 3, True), id="2 hilbert ket"), 
        pytest.param((2, True), id="simple ket"),),
    (
        pytest.param((2, 4, 3, True), id="complex ket"), 
        pytest.param((4, 6, True), id="complex ket"),),
    (
        pytest.param((2, False), id="simple dm"), 
        pytest.param((2, 3, False), id="2 hilbert dm"),),
    (
        pytest.param((2, 3, 2, False), id="3 hilbert dm"), 
        pytest.param((2, 6, False), id="2 hilbert dm"),),
    (
        pytest.param((2, 3, 4, False), id="complex dm"), 
        pytest.param((2, 6, 2, False), id="complex dm"),
    ),
]


class TestTrace(test_tools.TestTrace):
    specialisations = [
        pytest.param(trace_cuState, CuState, CuState, complex),
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


class TestInner(test_tools.TestInner):
    specialisations = [
        pytest.param(inner_cuState, CuState, CuState, complex),
    ]

    shapes = [(hilbert[0], hilbert[0]) for hilbert in _unary_pure]
    bad_shapes = []
