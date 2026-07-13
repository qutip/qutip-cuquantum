from typing import Any


import numpy as np
import scipy.linalg
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
from qutip_cuquantum.stochastic.batching import split_batch

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
    (pytest.param((2, 2, StateType.BATCH_KET), id="batched ket"),),
]

_unary_mixed = [
    (pytest.param((3, StateType.DM), id="scalar dm"),),
    (pytest.param((2, 3, StateType.DM), id="2 hilbert dm"),),
    (pytest.param((2, 3, 4, StateType.DM), id="complex dm"),),
    (pytest.param((2, 2, StateType.BATCH_DM), id="batched dm"),),
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
    (
        pytest.param((2, 3, StateType.BATCH_KET), id="batched ket"),
        pytest.param((2, 3, StateType.BATCH_KET), id="batched ket"),
    ),
    (
        pytest.param((2, 3, StateType.BATCH_DM), id="batched ket"),
        pytest.param((2, 3, StateType.BATCH_DM), id="batched ket"),
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
    (
        pytest.param((2, 3, StateType.DM), id="2,3 dm"),
        pytest.param((2, 3, StateType.BATCH_DM), id="2,3 dm"),
    ),
    (
        pytest.param((2, 3, 2, StateType.BATCH_DM), id="2,3,2 dm"),
        pytest.param((2, 3, 2, StateType.KET), id="2,3,2 ket"),
    ),
    (
        pytest.param((2, 3, StateType.BATCH_KET), id="2,3 ket"),
        pytest.param((2, 3, StateType.BATCH_BRA), id="2,3 bra"),
    ),
    (
        pytest.param((2, 3, StateType.BATCH_BRA), id="2,3 bra"),
        pytest.param((2, 3, StateType.BATCH_KET), id="2,3 ket"),
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
        pytest.param((3, StateType.KET), id="simple ket"),
    ),
    (
        pytest.param((2, 3, StateType.KET), id="2 hilbert ket"),
        pytest.param((2, StateType.KET), id="simple ket"),
    ),
    (
        pytest.param((2, 4, 3, StateType.KET), id="complex ket"),
        pytest.param((4, 6, StateType.KET), id="complex ket"),
    ),
    (
        pytest.param((2, StateType.DM), id="simple dm"),
        pytest.param((2, 3, StateType.DM), id="2 hilbert dm"),
    ),
    (
        pytest.param((2, 3, 2, StateType.DM), id="3 hilbert dm"),
        pytest.param((2, 6, StateType.DM), id="2 hilbert dm"),
    ),
    (
        pytest.param((2, 3, 4, StateType.DM), id="complex dm"),
        pytest.param((2, 6, 2, StateType.DM), id="complex dm"),
    ),
]


class TestTrace(test_tools.TestTrace):
    def op_numpy(self, matrix):
        if len(matrix.shape) == 3:
            return np.einsum("iik->k", matrix)
        return np.sum(np.diag(matrix))

    specialisations = [
        pytest.param(trace_cuState, CuState, complex),
        pytest.param(trace_oper_ket_cuState, CuState, object),
    ]

    shapes = _unary_mixed
    bad_shapes = []


class TestAdd(test_tools.TestAdd):
    @pytest.mark.parametrize('scale', [None, 0.2, 0.5j],
                             ids=['unscaled', 'scale[real]',
                                  'scale[complex]'])
    def test_mathematically_correct(self, op, data_l, data_r, out_type, scale):
        """
        Test that the binary operation is mathematically correct for all the
        known type specialisations, including with scaling.
        """
        left, right = data_l(), data_r()
        if scale is not None:
            expected = self.op_numpy(left.to_array(), right.to_array(), scale)
            test = op(left, right, scale)
        else:
            expected = self.op_numpy(left.to_array(), right.to_array())
            test = op(left, right)

        assert isinstance(test, out_type)
        if issubclass(out_type, CuState):
            np.testing.assert_allclose(test.to_array(), expected,
                                       atol=self.atol, rtol=self.rtol)
        else:
            np.testing.assert_allclose(test, expected, atol=self.atol,
                                       rtol=self.rtol)

    specialisations = [
        pytest.param(add_cuState, CuState, CuState, CuState),
        pytest.param(iadd_cuState, CuState, CuState, CuState),
    ]

    shapes = _compatible_hilbert
    bad_shapes = _imcompatible_hilbert


class TestWRMN_error(test_tools.TestWRMN_error):
    def op_numpy(self, left, right, atol, rtol):
        return np.linalg.norm(
            np.abs(left) / (atol + rtol * np.abs(right)), axis=(0, 1)
        ) / (left.shape[0] * left.shape[1])**0.5

    specialisations = [
        pytest.param(wrmn_error_cuState, CuState, CuState, float),
    ]

    shapes = _compatible_hilbert
    bad_shapes = _imcompatible_hilbert


class TestMul(test_tools.TestMul):

    @pytest.mark.parametrize('scalar', [
        pytest.param(0, id='zero'),
        pytest.param(4.5, id='real'),
        pytest.param(3j, id='complex'),
    ])
    def test_mathematically_correct(self, op, data_m, scalar, out_type):
        matrix = data_m()
        expected = self.op_numpy(matrix.to_array(), scalar)
        test = op(matrix, scalar)
        assert isinstance(test, out_type)
        if issubclass(out_type, CuState):
            assert test.shape == expected.shape[:2]
            np.testing.assert_allclose(test.to_array(), expected,
                                       atol=self.atol,
                                       rtol=self.rtol)
        else:
            np.testing.assert_allclose(test, expected, atol=self.atol,
                                       rtol=self.rtol)

    specialisations = [
        pytest.param(mul_cuState, CuState, CuState),
        pytest.param(imul_cuState, CuState, CuState),
    ]

    shapes = _unary_pure + _unary_mixed
    bad_shapes = []


class TestFrobeniusNorm(test_norm.TestFrobeniusNorm):
    def op_numpy(self, matrix):
        print(matrix.shape)
        if len(matrix.shape) == 3:
            return [
                scipy.linalg.norm(matrix[:, :, i], 'fro')
                for i in range(matrix.shape[2])
            ]
        return scipy.linalg.norm(matrix, 'fro')

    specialisations = [
        pytest.param(frobenius_cuState, CuState, object),
    ]

    shapes = _unary_pure + _unary_mixed
    bad_shapes = []


class TestL2Norm(test_norm.TestL2Norm):
    def op_numpy(self, matrix):
        if len(matrix.shape) == 3:
            return np.array([
                scipy.linalg.norm(matrix[:, :, i], 'fro')
                for i in range(matrix.shape[2])
            ])
        return scipy.linalg.norm(matrix, 'fro')

    specialisations = [
        pytest.param(l2_cuState, CuState, object),
    ]

    shapes = _unary_pure
    bad_shapes = _unary_mixed


class TestInner(test_tools.TestInner):

    def op_numpy(self, left, right, scalar_is_ket=False):
        if len(left.shape) == 3:
            if left.shape[1] == 1:
                if left.shape[0] != 1 or scalar_is_ket:
                    left = np.conj(left.transpose(1, 0, 2))
            return np.einsum("abd,bcd->d", left, right)
        return super().op_numpy(left, right, scalar_is_ket)

    specialisations = [
        pytest.param(inner_cuState, CuState, CuState, object),
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
