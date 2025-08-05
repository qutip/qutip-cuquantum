import numpy as np
import cupy as cp
import pytest
import random
import numbers
from itertools import product
cudense = pytest.importorskip("cuquantum.densitymat")

import qutip
from qutip_cuquantum.state import (
    CuState, iadd_cuState, add_cuState, mul_cuState, imul_cuState
    frobenius_cuState, kron_cuState, trace_cuState, inner_cuState
)

import qutip.core.data as _data
import qutip.tests.core.data.test_mathematics as test_tools
import qutip.tests.core.data.test_norm as test_norm


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
    (2, True),
    (-6, True),
    (-2, 2, 2, True),
]

_unary_mixed = [
    (3, False),
    (2, 3, False),
    (2, 3, -4, False),
]


_compatible_hilbert = [
    ((2, True), (2, True)),
    ((2, 3, False), (2, 3, False)),
    ((2, 3, True), (-6, True)),
    ((2, 3, -4, False), (-6, 2, 2, False)),
    ((2, -4, True), (-4, 2, True)),
]


_imcompatible_hilbert = [
    ((2, True), (2, False)),
    ((2, True), (3, True)),
    ((2, 3, True), (-6, False)),
    ((2, 3, False), (3, 2, False)),
    ((-2, -4, True), (4, -2, True)),
]


class TestKron(test_tools.TestKron):
    specialisations = [
        pytest.param(kron_cuState, CuState, CuState, CuState),
    ]

    shapes = (
        list(product(_unary_pure, repeat=2))
        + list(product(_unary_mixed, repeat=2))
    )
    bad_shapes = []


class TestTrace(test_tools.TestTrace):
    specialisations = [
        pytest.param(trace_cuState, CuState, CuState, CuState),
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


class TestMul(test_tools.TestMul):
    specialisations = [
        pytest.param(mul_cuState, CuState, CuState),
        pytest.param(imul_cuState, CuState, CuState),
    ]

    shapes = _unary_pure + _unary_mixed
    bad_shapes = []


class TestFrobeniusNorm(test_norm.TestFrobeniusNorm):
    specialisations = [
        pytest.param(frobenius_cuState, CuState, numbers.Number),
    ]

    shapes = _unary_pure + _unary_mixed
    bad_shapes = []


class TestInner(test_tools.TestInner):
    specialisations = [
        pytest.param(inner_cuState, CuState, CuState, complex),
    ]

    shapes = _unary_pure
    bad_shapes = []
