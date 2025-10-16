import qutip.core.data as _data
import qutip.tests.core.data.test_mathematics as test_tools
from qutip.tests.core.data.conftest import (
    random_csr, random_dense, random_diag
)
import pytest
import random
import numpy as np
import cupy as cp

cudense = pytest.importorskip("cuquantum.densitymat")

from qutip_cuquantum.operator import CuOperator, ProdTerm, Term
from qutip_cuquantum.utils import Transform
from qutip_cuquantum.state import CuState
from qutip_cuquantum.mixed_dispatch import matmul_cuoperator_custate_custate
import qutip_cuquantum
cudm_ctx = cudense.WorkStream()
qutip_cuquantum.set_as_default(cudm_ctx)


def _rand_transform(gen):
    """
    Random transform between raw, dag, T, conj, with bias toward common cases.
    """
    return gen.choice(list(Transform), p=[0.4, 0.15, 0.15, 0.3])


def _rand_elementary_oper(size, gen):
    if gen.uniform() < 0.5 and size > 0:
        # 50% Dia format
        mat = random_diag((size, size), gen.uniform()*0.4, False, gen)
    elif gen.uniform() < 0.6:
        # 30% Dense format
        mat = random_dense((abs(size), abs(size)), gen.uniform() > 0.5, gen)
    else:
        # 20% CSR format (not fully supported, converted to dense eventually)
        mat = random_csr((abs(size), abs(size)), gen.uniform()*0.4, False, gen)

    if gen.uniform() < 0.5 and size > 0:
        # Use cuDensity format instead of qutip.
        array_type = np if gen.uniform() < 0.5 else cp
        if isinstance(mat, _data.Dia):
            dia_matrix = mat.as_scipy()
            offsets = list(dia_matrix.offsets)
            data = array_type.zeros(
                (dia_matrix.shape[0], len(offsets)),
                dtype=complex,
            )
            for i, offset in enumerate(offsets):
                end = None if offset == 0 else -abs(offset)
                data[:end, i] = array_type.asarray( dia_matrix.diagonal(offset) )
            mat = cudense.MultidiagonalOperator(data, offsets)

        else:
            mat = cudense.DenseOperator(array_type.array(mat.to_array()))

    return mat


def random_CuOperator(hilbert_dims, N_elementary, seed):
    """
    Generate a random `CuOperator` matrix with the given hilbert_dims.
    """
    generator = np.random.default_rng(seed)
    out = CuOperator(hilbert_dims=hilbert_dims)
    for N in N_elementary:
        term = Term([], generator.normal() + 1j * generator.normal())
        for _ in range(N):
            mode = np.random.randint(len(hilbert_dims))
            size = hilbert_dims[mode]
            oper = _rand_elementary_oper(size, generator)

            term.prod_terms.append(ProdTerm(oper, (mode,), _rand_transform(generator)))
        out.terms.append(term)
    return out


def cases_cuoperator(hilbert):
    """Generate a random `CuPyDense` matrix with the given shape."""

    def factory(N_elementary, seed):
        return lambda: random_CuOperator(hilbert, N_elementary, seed)

    cases = []

    cases.append(pytest.param(factory([], 0), id="zero"))
    cases.append(pytest.param(factory([0], 0), id="id"))
    seed = random.randint(0, 2**31)
    cases.append(pytest.param(factory([1], seed), id=f"simple_{seed}"))
    seed = random.randint(0, 2**31)
    cases.append(pytest.param(factory([3], seed), id=f"3_prods_{seed}"))
    seed = random.randint(0, 2**31)
    cases.append(pytest.param(factory([1, 1, 1], seed), id=f"3_terms_{seed}"))
    seed = random.randint(0, 2**31)
    cases.append(pytest.param(factory([1, 2, 3], seed), id=f"complex_{seed}"))

    return cases


def random_pure_custate(hilbert):
    """Generate a random `CuPyDense` matrix with the given shape."""
    N = abs(np.prod(hilbert))
    out = (
        np.random.rand(N, 1) + 1j * np.random.rand(N, 1)
    ).astype(cp.complex128)
    out = _data.Dense(out)
    return CuState(out, hilbert, copy=False)


def random_mixed_custate(hilbert):
    """Generate a random `CuPyDense` matrix with the given shape."""
    N = abs(np.prod(hilbert))
    out = (
        np.random.rand(N, N) + 1j * np.random.rand(N, N)
    ).astype(cp.complex128)
    out = _data.Dense(out)
    return CuState(out, hilbert, copy=False)


def random_custate(shape):
    *hilbert, pure = shape
    if pure:
        return random_pure_custate(hilbert)
    else:
        return random_mixed_custate(hilbert)


test_tools._ALL_CASES = {
    CuOperator: cases_cuoperator,
    CuState: lambda shape: [lambda: random_custate(shape),],
}

test_tools._RANDOM = {
    CuOperator: lambda hilbert: [lambda: random_CuOperator(hilbert, [2], 0)],
    CuState: lambda shape: [lambda: random_custate(shape),],
}

_compatible_hilbert = [
    (pytest.param((2,), id="single"), pytest.param((2, True), id="single")),
    (pytest.param((2, 3), id="double"), pytest.param((2, 3, False), id="2-ket")),
    (pytest.param((-6,), id="single_weak"), pytest.param((2, 3, True), id="2-dm")),
    (pytest.param((2, -4), id="double_weak"), pytest.param((2, 2, 2, False), id="3-ket")),
    (pytest.param((2, 2, 2), id="triple"), pytest.param((2, 2, 2, True), id="3-dm")),
]

_imcompatible_hilbert = [
    (pytest.param((2,), id="single"), pytest.param((3, False), id="different")),
    (pytest.param((2, 3), id="double"), pytest.param((6, False), id="merged")),
    (pytest.param((2, 3), id="double"), pytest.param((3, 2, False), id="inverted")),
    (pytest.param((2, -4), id="double_weak"), pytest.param((4, 2, False), id="double_weak")),
    (pytest.param((2, 3, -4), id="complex"), pytest.param((6, 2, 2, False), id="complex")),
]


class TestMatmul(test_tools.TestMatmul):
    specialisations = [
        pytest.param(matmul_cuoperator_custate_custate, CuOperator, CuState, CuState),
    ]

    shapes = _compatible_hilbert
    bad_shapes = _imcompatible_hilbert
