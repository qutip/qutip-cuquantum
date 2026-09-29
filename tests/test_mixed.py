import qutip.core.data as _data
import qutip.tests.core.data.test_mathematics as test_tools
import pytest
import numpy as np
import cupy as cp
from utils import StateType, random_custate, random_CuOperator, cases_cuoperator

cudense = pytest.importorskip("cuquantum.densitymat")

from qutip_cuquantum.operator import CuOperator
from qutip_cuquantum.state import CuState
from qutip_cuquantum.mixed_dispatch import matmul_cuoperator_custate_custate, matmul_custate_cuoperator_custate
import qutip_cuquantum
#cudm_ctx = cudense.WorkStream()
#qutip_cuquantum.set_as_default(cudm_ctx)


@pytest.fixture(scope="module", autouse=True)
def global_gpu_backend():
    cudm_ctx = cudense.WorkStream()
    with qutip_cuquantum.CuQuantumBackend(cudm_ctx):
        yield


test_tools._ALL_CASES = {
    CuOperator: cases_cuoperator,
    CuState: lambda shape: [lambda: random_custate(shape),],
}

test_tools._RANDOM = {
    CuOperator: lambda hilbert: [lambda: random_CuOperator(hilbert, [2], 0)],
    CuState: lambda shape: [lambda: random_custate(shape),],
}

_compatible_op_state = [
    (pytest.param((2,), id="single"), pytest.param((2, StateType.KET), id="single")),
    (pytest.param((2, 3), id="double"), pytest.param((2, 3, StateType.DM), id="2-dm")),
    (pytest.param((-6,), id="single_weak"), pytest.param((2, 3, StateType.KET), id="2-ket")),
    (pytest.param((2, -4), id="double_weak"), pytest.param((2, 2, 2, StateType.DM), id="3-dm")),
    (pytest.param((2, 2, 2), id="triple"), pytest.param((2, 2, 2, StateType.KET), id="3-ket")),
]

_imcompatible_op_state = [
    (pytest.param((2,), id="single"), pytest.param((3, StateType.DM), id="different")),
    (pytest.param((2, 3), id="double"), pytest.param((6, StateType.DM), id="merged")),
    (pytest.param((2, 3), id="double"), pytest.param((3, 2, StateType.DM), id="inverted")),
    (pytest.param((2, -4), id="double_weak"), pytest.param((4, 2, StateType.DM), id="double_weak")),
    (pytest.param((2, 3, -4), id="complex"), pytest.param((6, 2, 2, StateType.DM), id="complex")),
    (pytest.param((2,), id="dm"), pytest.param((2, StateType.BRA), id="bra")),
]


class TestOpStateMatmul(test_tools.TestMatmul):
    specialisations = [
        pytest.param(matmul_cuoperator_custate_custate, CuOperator, CuState, CuState),
    ]

    shapes = _compatible_op_state
    bad_shapes = _imcompatible_op_state

_compatible_state_op = [
    (pytest.param((2, StateType.BRA), id="single"), pytest.param( (2,), id="single")),
    (pytest.param((2, 3, StateType.BRA), id="2-bra"), pytest.param( (2, 3), id="double")),
    (pytest.param((2, 3, StateType.DM), id="2-dm"), pytest.param((-6,), id="single_weak")),
    (pytest.param((2, 2, 2, StateType.BRA), id="3-bra"), pytest.param((2, -4), id="double_weak")),
    (pytest.param((2, 2, 2, StateType.DM), id="3-dm"), pytest.param((2, 2, 2), id="triple")),
]

_imcompatible_state_op = [
    (pytest.param((2, StateType.KET), id="single"), pytest.param( (2,), id="single")),
]

class TestStateOpMatmul(test_tools.TestMatmul):
    specialisations = [
        pytest.param(matmul_custate_cuoperator_custate, CuState, CuOperator, CuState),
    ]

    shapes = _compatible_state_op
    bad_shapes = _imcompatible_state_op


def test_mixed_dispatch_dual_op_dm():
    op = random_CuOperator((2, 3, 2, 3), [5], 0)
    state = random_custate((2, 3, StateType.DM))
    actual = matmul_cuoperator_custate_custate(op, state).to_array().ravel('F')
    expected = np.matmul(op.to_array(), state.to_array().ravel('F'))
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-7)
