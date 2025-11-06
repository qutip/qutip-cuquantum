import numpy as np
import cupy as cp
import pytest
import random
cudense = pytest.importorskip("cuquantum.densitymat")

import qutip
from qutip_cuquantum.operator import CuOperator, ProdTerm, Term
from qutip_cuquantum.utils import Transform
from qutip_cuquantum.state import zeros_like_cuState, CuState

import qutip.core.data as _data
import qutip.tests.core.data.test_mathematics as test_tools
from qutip.tests.core.data.conftest import (
    random_csr, random_dense, random_diag
)


def _rand_transform(gen):
    """
    Random transform between raw, dag, T, conj, with bias toward common cases.
    """
    return gen.choice(list(Transform), p=[0.4, 0.15, 0.15, 0.3])


def _rand_elementary_oper(size, gen):
    if gen.uniform() < 0.5:
        # 50% Dia format
        mat = random_diag((size, size), gen.uniform()*0.4, False, gen)
    elif gen.uniform() < 0.6:
        # 30% Dense format
        mat = random_dense((size, size), gen.uniform() > 0.5, gen)
    else:
        # 20% CSR format (not fully supported, converted to dense eventually)
        mat = random_csr((size, size), gen.uniform()*0.4, False, gen)

    if gen.uniform() < 0.5:
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
            size = abs(hilbert_dims[mode])
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


test_tools._ALL_CASES = {
    CuOperator: cases_cuoperator,
}

test_tools._RANDOM = {
    CuOperator: lambda hilbert: [lambda: random_CuOperator(hilbert, [2], 0)],
}

_unary_hilbert = [
    (pytest.param((1,), id="scalar"),),
    (pytest.param((3,), id="single"),),
    (pytest.param((-6,), id="single_weak"),),
    (pytest.param((2, 3), id="double"),),
    (pytest.param((-2, -4), id="double_weak"),),
    (pytest.param((2, 3, -4), id="complex"),),
]

_compatible_hilbert = [
    (pytest.param((2,), id="single"), pytest.param((2,), id="single")),
    (pytest.param((2, 3), id="double"), pytest.param((2, 3), id="double")),
    (pytest.param((2, 3), id="double"), pytest.param((-6,), id="single_weak")),
    (pytest.param((2, -4), id="double_weak"), pytest.param((-4, 2), id="double_weak")),
    (pytest.param((2, 3, -4), id="complex"), pytest.param((-6, 2, 2), id="complex")),
]

_imcompatible_hilbert = [
    (pytest.param((2,), id="single"), pytest.param((3,), id="single")),
    (pytest.param((2, 3), id="double"), pytest.param((6,), id="single")),
    (pytest.param((2, 3), id="double"), pytest.param((3, 2), id="single_weak")),
    (pytest.param((2, -4), id="double_weak"), pytest.param((4, -2), id="double_weak")),
    (pytest.param((2, 3, -4), id="complex"), pytest.param((6, 2, 2), id="complex")),
]


@pytest.mark.parametrize("shape", [(1, 1), (100, 100)])
def test_zeros(shape):
    oper = _data.zeros[CuOperator](*shape)
    assert np.all(oper.to_array() == 0.)
    assert oper.shape == shape
    assert oper.to_array().shape == shape
    assert len(oper.terms) == 0


@pytest.mark.parametrize("N", [1, 100])
def test_id(N):
    oper = _data.identity[CuOperator](N)
    assert np.all(oper.to_array() == np.eye(N))
    shape = (N, N)
    assert oper.shape == shape
    assert oper.to_array().shape == shape
    assert len(oper.terms) == 1
    assert len(oper.terms[0].prod_terms) == 0


@pytest.mark.parametrize(["N_diag", "size"], [
    (1, 1),
    (1, 10),
    (3, 2),
    (2, 10),
    (5, 25),
])
def test_diag(N_diag, size):
    diags = []
    offsets = list(range(-size + 1, size))
    random.shuffle(offsets)
    offsets = offsets[:N_diag]
    for offset in offsets:
        N = size - abs(offset)
        diags.append(np.random.randn(N) + 1j * np.random.randn(N))

    oper_cuoper = _data.diag[CuOperator](diags, offsets)
    oper_dia = _data.diag["Dia"](diags, offsets)
    assert np.all(oper_cuoper.to_array() == oper_dia.to_array())


def test_permute():
    X = qutip.sigmax(dtype=CuOperator)
    Y = qutip.sigmay(dtype=CuOperator)
    Z = qutip.sigmaz(dtype=CuOperator)
    I = qutip.qeye(2, dtype=CuOperator)

    oper = X & Y & Z & I
    new = _data.permute.dimensions(oper.data, [2, 2, 2, 2], [1, 3, 0, 2])
    target = Y & I & X & Z

    assert np.all(new.to_array() == target.full())

    A = qutip.fock_dm(3, 0).to(CuOperator)
    B = qutip.fock_dm(5, 4).to(CuOperator)
    C = qutip.fock_dm(4, 2).to(CuOperator)
    rho = qutip.tensor(A, B, C)
    rho2 = rho.permute([2, 0, 1])
    assert np.all(rho2.full() == qutip.tensor(C, A, B).full())


_qeye = lambda N: lambda : qutip.qeye(N)
_destroy = lambda N: lambda : qutip.destroy(N)


def _cplx(N):
    def func():
        return (
            (qutip.create(N) & qutip.qeye(2))
            + ( qutip.qeye(N) & qutip.sigmax() )
            + ( (qutip.destroy(N) @ qutip.create(N)) & qutip.sigmay() )
        )

    return func


@pytest.mark.parametrize(["left", "right", "expected"], [
    pytest.param(_qeye(3), _qeye(3), True, id="qeye", marks=pytest.mark.xfail),
    pytest.param(_qeye(2), _qeye(3), False, id="qeye_bad_size"),
    pytest.param(_destroy(3), _destroy(4), False, id="destroy_bad_size"),
    pytest.param(_qeye(3), _destroy(3), False, id="different"),
    pytest.param(_destroy(4), _destroy(4), True, id="destroy", marks=pytest.mark.xfail),
    pytest.param(_cplx(3), _cplx(3), True, id="complex", marks=pytest.mark.xfail),
    pytest.param(_cplx(3), _qeye(4), False, id="complex_bad_size"),
])
def test_equal(left, right, expected):
    left = left()
    right = right()
    assert (left.data == right.data) is expected


class TestAdd(test_tools.TestAdd):
    specialisations = [
        pytest.param(lambda x, y, s=1.: x + y * s, CuOperator, CuOperator, CuOperator),
    ]

    shapes = _compatible_hilbert
    bad_shapes = _imcompatible_hilbert


class TestSub(test_tools.TestSub):
    specialisations = [
        pytest.param(lambda x, y: x - y, CuOperator, CuOperator, CuOperator),
    ]

    shapes = _compatible_hilbert
    bad_shapes = _imcompatible_hilbert


class TestMatmul(test_tools.TestMatmul):
    specialisations = [
        pytest.param(lambda x, y: x @ y, CuOperator, CuOperator, CuOperator),
    ]

    shapes = _compatible_hilbert
    bad_shapes = _imcompatible_hilbert


class TestMul(test_tools.TestMul):
    specialisations = [
        pytest.param(lambda op, scale: op * scale, CuOperator, CuOperator),
    ]

    shapes = _unary_hilbert
    bad_shapes = []


class TestNeg(test_tools.TestNeg):
    specialisations = [
        pytest.param(lambda op: -op, CuOperator, CuOperator),
    ]

    shapes = _unary_hilbert
    bad_shapes = []


class TestAdjoint(test_tools.TestAdjoint):
    specialisations = [
        pytest.param(lambda op: op.adjoint(), CuOperator, CuOperator),
    ]

    shapes = _unary_hilbert
    bad_shapes = []


class TestConj(test_tools.TestConj):
    specialisations = [
        pytest.param(lambda op: op.conj(), CuOperator, CuOperator),
    ]

    shapes = _unary_hilbert
    bad_shapes = []


class TestTranspose(test_tools.TestTranspose):
    specialisations = [
        pytest.param(lambda op: op.transpose(), CuOperator, CuOperator),
    ]

    shapes = _unary_hilbert
    bad_shapes = []


class TestKron(test_tools.TestKron):
    specialisations = [
        pytest.param(_data.kron, CuOperator, CuOperator, CuOperator),
    ]

    shapes = _compatible_hilbert + _imcompatible_hilbert
    bad_shapes = []


class TestIsEqual:

    _compatible_hilbert = [
        ((2,), (2,)),
        ((2, 3), (2, 3)),
        ((2, 3), (-6,)),
        ((2, -4), (-4, 2)),
        ((2, 3, -4), (-6, 2, 2)),
    ]

    _imcompatible_hilbert = [
        ((2,), (3,)),
        ((2, 3), (3, 2)),
        ((2, 3), (6,)),
        ((2, -4), (4, -2)),
        ((2, 3, -4), (6, 2, 2)),
    ]

    def op_numpy(self, left, right, atol, rtol):
        return np.allclose(left.to_array(), right.to_array(), rtol, atol)

    @pytest.mark.parametrize("hilbert", _compatible_hilbert)
    def test_same_shape(self, hilbert):
        atol = 1e-8
        rtol = 1e-6
        A = random_CuOperator(hilbert[0], [2], 23)
        B = random_CuOperator(hilbert[1], [1, 1], 32)
        assert _data.isequal(A, A, atol, rtol)
        assert _data.isequal(B, B, atol, rtol)
        assert (
            _data.isequal(A, B, atol, rtol) == self.op_numpy(A, B, atol, rtol)
        )

    @pytest.mark.parametrize("hilbert", _imcompatible_hilbert)
    def test_different_shape(self, hilbert):
        A = random_CuOperator(hilbert[0], [1], 21)
        B = random_CuOperator(hilbert[1], [1], 12)
        assert not _data.isequal(A, B, 1e10, 1e10)

    @pytest.mark.parametrize("rtol", [1e-6, 100])
    @pytest.mark.parametrize("hilbert", _compatible_hilbert)
    def test_rtol(self, hilbert, rtol):
        mat = random_CuOperator(hilbert[0], [1], 135)
        assert _data.isequal(mat + mat * (rtol / 10), mat, 1e-14, rtol)
        assert not _data.isequal(mat * (1 + rtol * 10), mat, 1e-14, rtol)

    @pytest.mark.parametrize("atol", [1e-14, 1e-6, 100])
    @pytest.mark.parametrize("hilbert", _compatible_hilbert)
    def test_atol(self, hilbert, atol):
        A = random_CuOperator(hilbert[0], [1, 1], 123)
        B = random_CuOperator(hilbert[1], [2], 321)
        assert _data.isequal(A, A + B * (atol / 10), atol, 0)
        assert not _data.isequal(A, A + B * (atol * 10), atol, 0)


def test_isherm():
    A = qutip.rand_dm(3, dtype="CuOperator")
    B = qutip.rand_dm(4, dtype="CuOperator")
    assert _data.isherm(A.data)
    assert _data.isherm(B.data)
    assert _data.isherm(((A & qutip.qeye(4)) + (qutip.qeye(3) & B)).data)
    assert _data.isherm(((A & qutip.qeye(4)) * (qutip.qeye(3) & B)).data)
    C = qutip.rand_stochastic(3, density=1, dtype="CuOperator")
    assert _data.isherm(C.data) == False
    assert _data.isherm((A @ C).data) == False
    assert _data.isherm(((C & qutip.qeye(4)) * (qutip.qeye(3) & B)).data) == False


@pytest.fixture(params=[
    pytest.param("dag", id='dag'),
    pytest.param("trans", id='trans'),
    pytest.param("conj", id='conj'),
    pytest.param("copy", id='direct'),
])
def transform(request):
    return request.param


def _compare_Operator(operator, qobj, N=3):
    """
    Check that ``operator == qobj`` for an cuquantum.densitymat.Operator
    by multiplying each by random states vector and comparing the output.
    """
    # TODO: OperatorTerm @ ID would return the operator in a format that we can
    # compare directly.
    for _ in range(N):
        if qobj._dims.issuper:
            dims = qobj.dims[1][0]
            state = qutip.rand_dm(dims, density=1., dtype="Dense")
            custate = CuState(state.data, hilbert_dims=dims)
        else:
            dims = qobj.dims[1]
            state = qutip.rand_ket(dims, density=1., dtype="Dense")
            custate = CuState(state.data, hilbert_dims=dims)

        assert state == qutip.Qobj(custate, dims=state.dims).to("dense")

        expected = qobj(state)

        out = zeros_like_cuState(custate)
        if _ == 0:
            operator.prepare_action(
                qutip.settings.cuDensity["ctx"],
                custate.base,
            )
        operator.compute_action(
            0., None, custate.base, out.base,
        )
        obtained = qutip.Qobj(out, dims=state.dims).to("dense")
        assert expected == obtained


class TestToOperatorTerm:
    ctx = cudense.WorkStream()
    qutip.settings.cuDensity["ctx"] = ctx
    _terms_core = None
    _terms_cu = None
    _op1 = None
    _op2 = None

    @property
    def terms_core(self):
        if self._terms_core is not None:
            return self._terms_core
        id2 = qutip.qeye(2, dtype="dia")
        a = qutip.destroy(2, dtype="dia")
        y = qutip.sigmay(dtype="csr")
        z = qutip.sigmaz(dtype="dia")
        op1 = qutip.rand_dm(2, density=1., dtype="Dense")
        op2 = qutip.rand_dm(2, density=1., dtype="Dense")
        self._op1 = op1
        self._op2 = op2

        self._terms_core = (
            (op1 @ op2) & id2 & id2,  # matmul before tensor
            (id2 & op1 & id2) @ (id2 & op2 & id2),  # matmul after tensor
            (id2 & op1 & id2) @ (id2 & id2 & op2),  # matmul diff mode
            (id2 & y & z) @ (id2 & id2 & y),  # matmul overlap
            ((op1 @ op2) & id2 & id2).dag(),  # matmul dag

            id2 & (a + z) & id2,  # sum before tensor
            (id2 & y & id2) + (id2 & y & id2),  # sum after tensor
            (id2 & y & id2) + (id2 & id2 & y),  # sum diff mode
            (id2 & y & z) + (id2 & id2 & y),  # sum overlap

            (op1 & id2 & op2),  # 2 modes 1
            (id2 & y & op2),  # 2 modes 2
            (y & z & op1),  # 3 modes 1
            (y & z & op1).dag(),  # 3 modes 1 dag
            (op1 & y & op2),  # 3 modes 2
        )
        return self._terms_core

    @property
    def terms_cu(self):
        if self._terms_cu is not None:
            return self._terms_cu

        a = qutip.destroy(2, dtype="dia")
        y = qutip.sigmay(dtype="csr")
        z = qutip.sigmaz(dtype="dia")

        id2 = qutip.qeye(2, dtype=CuOperator)
        y_op2 = (y & self._op2).to(CuOperator)
        y_z_op1 = (y & z & self._op1).to(CuOperator)
        a = qutip.destroy(2, dtype="dia").to(CuOperator)
        y = qutip.sigmay(dtype="csr").to(CuOperator)
        z = qutip.sigmaz(dtype="dia").to(CuOperator)
        op1 = self._op1.to(CuOperator)
        op2 = self._op2.to(CuOperator)

        self._terms_cu = (
            (op1 @ op2) & id2 & id2,  # matmul before tensor
            (id2 & op1 & id2) @ (id2 & op2 & id2),  # matmul after tensor
            (id2 & op1 & id2) @ (id2 & id2 & op2),  # matmul diff mode
            (id2 & y & z) @ (id2 & id2 & y),  # matmul overlap
            ((op1 @ op2) & id2 & id2).dag(),  # matmul dag

            id2 & (a + z) & id2,  # sum before tensor
            (id2 & y & id2) + (id2 & y & id2),  # sum after tensor
            (id2 & y & id2) + (id2 & id2 & y),  # sum diff mode
            (id2 & y & z) + (id2 & id2 & y),  # sum overlap

            (op1 & id2 & op2),  # 2 modes 1
            (id2 & y_op2),  # 2 modes 2
            (y_z_op1),  # 3 modes 1
            (y_z_op1).dag(),  # 3 modes 1 dag
            (op1 & y_op2),  # 3 modes 2
        )
        return self._terms_cu

    def test_dense_single(self, transform):
        op = qutip.rand_dm(4, density=1., dtype="Dense")
        reference = qutip.qeye(3, dtype="dia") & op & qutip.qeye(5, dtype="dia")
        reference = getattr(reference, transform)()
        cuoper = (
            qutip.qeye(3, dtype=CuOperator)
            & op.to(CuOperator)
            & qutip.qeye(5, dtype=CuOperator)
        )
        cuoper = getattr(cuoper, transform)()
        opterm = cuoper.data.to_OperatorTerm()
        oper = cudense.Operator([3, 4, 5], [opterm])
        _compare_Operator(oper, reference)

    def test_multi(self):
        op1 = qutip.rand_dm(3, density=1., dtype="Dense")
        op2 = qutip.rand_dm(4, density=1., dtype="Dense")
        op3 = qutip.destroy(4, dtype="dia")
        op4 = qutip.rand_dm(5, density=1., dtype="Dense")
        reference = (
            (op1.conj() & qutip.qeye(4, dtype="dia") & qutip.qeye(5, dtype="dia"))
            + (qutip.qeye(3, dtype="dia") & op2.trans() & qutip.qeye(5, dtype="dia"))
            + (qutip.qeye(3, dtype="dia") & (op2 @ op3) & qutip.qeye(5, dtype="dia"))
            + (qutip.qeye(3, dtype="dia") & op3 & op4.dag())
        )

        op1 = op1.to(CuOperator)
        op2 = op2.to(CuOperator)
        op3 = op3.to(CuOperator)
        op4 = op4.to(CuOperator)

        id3 = qutip.qeye(3, dtype=CuOperator)
        id4 = qutip.qeye(4, dtype=CuOperator)
        id5 = qutip.qeye(5, dtype=CuOperator)
        cuoper = (
            (op1 & id4 & id5).conj()
            + (id3 & op2 & id5).trans()
            + (id3 & op2 & id5) @ (id3 & op3 & id5)
            + (id3 & op3 & op4.dag())
        )
        opterm = cuoper.data.to_OperatorTerm()
        oper = cudense.Operator([3, 4, 5], [opterm])
        _compare_Operator(oper, reference)

    @pytest.mark.parametrize("term", range(14))
    def test_multi2(self, term):
        reference = self.terms_core[term]
        cuoper = self.terms_cu[term]
        cuoper.data._update_hilbert([2, 2, 2])
        opterm = cuoper.data.to_OperatorTerm(dual=False)
        oper = cudense.Operator([2, 2, 2], [opterm])
        _compare_Operator(oper, reference)

    def test_dense_pre(self, transform):
        op = qutip.rand_dm(4, density=1.,dtype="Dense")
        reference = qutip.qeye(3, dtype="dia") & op & qutip.qeye(5, dtype="dia")
        reference = getattr(reference, transform)()
        reference = qutip.spre(reference)
        cuoper = (
            qutip.qeye(3, dtype=CuOperator)
            & op.to(CuOperator)
            & qutip.qeye(5, dtype=CuOperator)
        )
        cuoper = getattr(cuoper, transform)()
        cuoper = qutip.spre(cuoper)
        opterm = cuoper.data.to_OperatorTerm(dual=True)
        oper = cudense.Operator([3, 4, 5], [opterm])
        _compare_Operator(oper, reference)

    def test_dense_dual(self, transform):
        op = qutip.rand_dm(4, density=1.,dtype="Dense")
        reference = qutip.qeye(3, dtype="dia") & op & qutip.qeye(5, dtype="dia")
        reference = getattr(reference, transform)()
        reference = qutip.spost(reference)
        cuoper = (
            qutip.qeye(3, dtype=CuOperator)
            & op.to(CuOperator)
            & qutip.qeye(5, dtype=CuOperator)
        )
        cuoper = getattr(cuoper, transform)()
        cuoper = qutip.spost(cuoper)
        opterm = cuoper.data.to_OperatorTerm(dual=True)
        oper = cudense.Operator([3, 4, 5], [opterm])
        _compare_Operator(oper, reference)

    def test_dense_mix(self):
        opl = qutip.rand_dm(2, density=1.,dtype="Dense")
        opr = qutip.rand_dm(2, density=1.,dtype="Dense")
        id2 = qutip.qeye(2, dtype="dia")
        reference = qutip.sprepost(id2 & opl & id2, id2 & opr & id2)

        cuoper = qutip.Qobj(CuOperator(
            qutip.sprepost(opl, opr).data,
            [1, 4],
            hilbert_dims=[2, 2, 2, 2, 2, 2]
        ))

        opterm = cuoper.data.to_OperatorTerm(dual=True)
        oper = cudense.Operator([2, 2, 2], [opterm])
        _compare_Operator(oper, reference)

    def test_dense_mix2(self):
        opl = qutip.rand_dm([2, 2], density=1.,dtype="Dense")
        opr = qutip.rand_dm([2, 2], density=1.,dtype="Dense")
        id2 = qutip.qeye(2, dtype="dia")
        reference = qutip.sprepost(id2 & opl, id2 & opr)

        cuoper = qutip.Qobj(CuOperator(
            qutip.sprepost(opl, opr).data,
            [1, 2, 4, 5],
            hilbert_dims=[2, 2, 2, 2, 2, 2]
        ))

        opterm = cuoper.data.to_OperatorTerm(dual=True)
        oper = cudense.Operator([2, 2, 2], [opterm])
        _compare_Operator(oper, reference)

    @pytest.mark.parametrize("term", range(14))
    def test_dense_complex(self, term):
        reference = (
            qutip.spre(self.terms_core[term])
            @ qutip.spost(self.terms_core[term])
        )
        cuoper = (
            qutip.spre(self.terms_cu[term])
            @ qutip.spost(self.terms_cu[term])
        )
        cuoper.data._update_hilbert([2, 2, 2, 2, 2, 2])
        opterm = cuoper.data.to_OperatorTerm(dual=True)
        oper = cudense.Operator([2, 2, 2], [opterm])
        _compare_Operator(oper, reference)
