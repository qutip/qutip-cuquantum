from cuquantum.densitymat import (
    MultidiagonalOperator,
    DenseOperator,
    OperatorTerm,
    tensor_product,
)
from cuquantum.densitymat._internal.callbacks import ScalarCallbackCoefficient

from typing import NamedTuple, Any
import itertools
import numpy as np
import cupy as cp

from qutip.core.data import Data
from qutip.core import data as _data
from qutip.settings import settings

from .utils import (
    Transform, conj_transform, trans_transform, adjoint_transform,
    _compare_hilbert
)

__all__ = ["CuOperator"]

# TODO: sanity check on mode as int instead of tuple
# Input fixed in tests, but did not fail early


def _transpose_cu_operator(oper):
    if isinstance(oper, MultidiagonalOperator):
        out = MultidiagonalOperator(
            oper.data,
            [-offset for offset in oper.offsets],
            callback=oper.callback,
        )
    elif isinstance(oper, DenseOperator):
        N = oper.num_modes
        batch_dims_oper = len(oper.data.shape) % 2
        batch_dims_callback = len(oper.callback.callback(0, None).shape) % 2
        perm = tuple(range(N, 2*N)) + tuple(range(N))
        new_callback = None

        if oper.callback is not None:
            perm_callback = perm
            if batch_dims_callback:
                perm_callback += (2 * N,)

            @oper.callback.__class__
            def new_callback(t, _):
                # TODO: copy needed?
                return (
                    oper.callback.callback(t, _)
                    .transpose(perm_callback)
                    .copy(order="F")
                )

        if batch_dims_oper:
            perm += (2 * N,)
        out = DenseOperator(
            oper.data.transpose(perm).copy(order="F"),
            callback=new_callback
        )
    else:
        raise NotImplementedError

    return out


def _oper_to_array(oper, transform):
    if isinstance(oper, _data.Data):
        arr = oper.to_array()

    elif isinstance(oper, DenseOperator):
        N = np.prod(oper.mode_dims)
        arr = oper.data[..., 0].reshape((N, N))
        if isinstance(arr, cp.ndarray):
            arr = arr.get() # Convert CuPy to NumPy

    elif isinstance(oper, MultidiagonalOperator):
        # for diag, offset in zip(oper.data[:, :, 0].T, oper.offsets):
        arr = sum(
            np.diag(diag[:-abs(offset) or None], offset)
            for diag, offset
            in zip(oper.data[:, :, 0].T, oper.offsets)
        )
    else:
        raise NotImplementedError

    if transform == Transform.DIRECT:
        out = arr
    elif transform == Transform.CONJ:
        out = arr.conj()
    elif transform == Transform.TRANSPOSE:
        out = arr.transpose()
    elif transform == Transform.ADJOINT:
        out = arr.transpose().conj()

    return out


def _oper_to_ElementaryOperator(
    oper,
    hilbert_idx,
    hilbert_dims,
    transform,
    copy=False
):
    N = len(hilbert_idx)
    shape = tuple(hilbert_dims[idx] for idx in hilbert_idx)

    if isinstance(oper, (DenseOperator, MultidiagonalOperator)):
        if N != 1 and isinstance(oper, MultidiagonalOperator):
            raise ValueError(
                "MultidiagonalOperator on multiple hilbert spaces"
            )
        if list(oper.shape[:len(oper.shape) // 2]) != list(shape):
            raise ValueError(
                f"Operator shape does not match hilbert spaces: "
                f"{list(oper.shape[:len(oper.shape) // 2])}, {shape}"
            )

        if transform == Transform.DIRECT:
            out = oper
        elif transform == Transform.ADJOINT:
            out = oper.dag()
        elif transform == Transform.CONJ:
            out = _transpose_cu_operator(oper).dag()
        elif transform == Transform.TRANSPOSE:
            out = _transpose_cu_operator(oper)

    else:
        if transform == Transform.DIRECT:
            pass
        elif transform == Transform.ADJOINT:
            oper = oper.adjoint()
        elif transform == Transform.CONJ:
            oper = oper.conj()
        elif transform == Transform.TRANSPOSE:
            oper = oper.transpose()

        if isinstance(oper, _data.Dia) and N == 1:
            dia_matrix = oper.as_scipy()
            offsets = list(dia_matrix.offsets)
            data = np.zeros((dia_matrix.shape[0], len(offsets)), dtype=complex)
            for i, offset in enumerate(offsets):
                end = None if offset == 0 else -abs(offset)
                data[:end, i] = dia_matrix.diagonal(offset)
            out = MultidiagonalOperator(data, offsets)

        else:
            out = DenseOperator(oper.to_array().reshape(shape + shape))

    return out


###############################################################################
###############################################################################

class ProdTerm(NamedTuple):
    operator: Any
    hilbert: tuple[int]
    transform: Transform


class Term(NamedTuple):
    prod_terms: list[ProdTerm]
    factor: complex


def _has_dual(arg):
    for term_duals in arg.duals:
        for oper_dual in term_duals:
            for mode_dual in oper_dual:
                if mode_dual:
                    return True
    return False


class CuOperator(Data):
    """
    Pseudo symbolic data layer that follow a structure close to
    cuDensity OperatorTerm.

    Only support square operators.

    The operator is composed by a sum of terms, which reach term composed of
    multiple product:

    op = sum_i term[i].factor * prod(term[i].prod_terms)

    prod(term[i].prod_terms) =
        expand_operator(oper[N-1]^trans[N-1], hilbert[N-1], hilbert_dims) @
        ...
        expand_operator(oper[1]^trans[1], hilbert[1], hilbert_dims) @
        expand_operator(oper[0]^trans[0], hilbert[0], hilbert_dims)

    This object always has (partial) knowledge of the full hilbert space.
    There is no dual representation, super operator will have the hilbert space
    doubled, with the operation being applied to the right first and
    transposed.
    """
    terms: list
    hilbert_dims: tuple

    def __init__(
        self,
        arg=None,
        mode=None,
        shape=None,
        copy=True,
        hilbert_dims=None,
    ):
        self.terms = []
        self.hilbert_dims = ()
        self._oper = None
        oper_shape = None
        if isinstance(mode, int):
            mode = (mode,)

        if arg is None:
            if hilbert_dims is None and shape is None:
                raise ValueError(
                    "Either hilbert_dims or shape must be provided "
                    "for empty CuOperator."
                )
            if hilbert_dims is not None:
                self.hilbert_dims = hilbert_dims
                N = abs(np.prod(hilbert_dims))
                oper_shape = (N, N)
            else:
                self.hilbert_dims = (-shape[0],)
                oper_shape = shape

        elif isinstance(arg, MultidiagonalOperator):
            oper_shape = arg.shape
            self.terms.append(
                Term([ProdTerm(arg, mode or (0,), Transform.DIRECT)], 1.+0j)
            )
            if hilbert_dims is None:
                self.hilbert_dims = (arg.shape[0],)
            else:
                # TODO: raise error if not matching?
                self.hilbert_dims = hilbert_dims

        elif isinstance(arg, DenseOperator):
            oper_shape = np.prod(arg.mode_dims), np.prod(arg.mode_dims)
            if mode is None:
                mode = tuple(i for i in range(arg.num_modes))
            self.terms.append(
                Term([ProdTerm(arg, mode, Transform.DIRECT)], 1.+0j)
            )
            if hilbert_dims is None:
                self.hilbert_dims = arg.mode_dims
            else:
                # TODO: raise error if not matching?
                self.hilbert_dims = hilbert_dims

        elif isinstance(arg, OperatorTerm):
            if hilbert_dims is None:
                hilbert_dims = arg.hilbert_space_dims
            if hilbert_dims is None:
                raise ValueError(
                    "hilbert_dims must be provided for OperatorTerm"
                    " or derivable from it."
                )

            has_dual = _has_dual(arg)
            N = len(hilbert_dims)
            iter = zip(arg.terms, arg.modes, arg.duals, arg._coefficients)
            for terms_, modes, duals, coeff in iter:
                if not isinstance(coeff, ScalarCallbackCoefficient):
                    raise NotImplementedError(
                        "OperatorTerm with Coeffcient are not supported."
                    )
                terms = Term([], factor=coeff._static_coeff)

                for term, mode, dual in zip(terms_, modes, duals):
                    # term = term.copy() if copy else term
                    if has_dual and not dual:
                        mode = tuple(i + N for i in mode)
                    if has_dual and dual:
                        terms.prod_terms.append(
                            ProdTerm(term, mode, Transform.TRANSPOSE)
                        )
                    else:
                        terms.prod_terms.append(
                            ProdTerm(term, mode, Transform.DIRECT)
                        )
                self.terms.append(terms)

            if has_dual:
                self.hilbert_dims = hilbert_dims + hilbert_dims
            else:
                self.hilbert_dims = hilbert_dims
            hilbert_dims = None
            N = abs(np.prod(self.hilbert_dims))
            oper_shape = (N, N)

        elif isinstance(arg, Data) and not isinstance(arg, CuOperator):
            arg = arg.copy() if copy else arg
            self.terms.append(
                Term([ProdTerm(arg, mode or (0,), Transform.DIRECT)], 1.+0j)
            )
            if hilbert_dims is None:
                self.hilbert_dims = (-arg.shape[0],)
                oper_shape = arg.shape
            else:
                self.hilbert_dims = hilbert_dims
                oper_shape = (abs(np.prod(hilbert_dims)),) * 2

        else:
            raise TypeError(f"{type(arg)} not supported.")

        if shape and shape != oper_shape:
            raise ValueError(
                f"Provided shape {shape} does "
                f"not match operator shape {oper_shape}."
            )
        if oper_shape[0] != oper_shape[1]:
            raise ValueError("Operator must be square")
        if abs(np.prod(self.hilbert_dims)) != oper_shape[0]:
            raise ValueError(
                f"Total product of hilbert_dims "
                f"{abs(np.prod(self.hilbert_dims))} "
                f"does not match operator dimension {oper_shape[0]}."
            )

        super().__init__(oper_shape)

        if hilbert_dims is not None:
            self._update_hilbert(hilbert_dims)

    def _update_hilbert(self, new):
        matched = _compare_hilbert(self.hilbert_dims, new, return_shifts=True)
        if not matched:
            raise ValueError(f"{self.hilbert_dims} updated to {new}")

        new_hilbert, shifts, _ = matched
        self.hilbert_dims = tuple(new_hilbert)
        new_terms = []
        for term in self.terms:
            copy_term = Term([], factor=term.factor)
            for pterm in term.prod_terms:
                new_mode = []
                for i in pterm.hilbert:
                    new_mode += shifts[i]
                copy_term.prod_terms.append(ProdTerm(
                    pterm.operator,
                    tuple(new_mode),
                    pterm.transform,
                ))
            new_terms.append(copy_term)
        self.terms = new_terms

    @property
    def hilbert_space_dims(self):
        return tuple(abs(i) for i in self.hilbert_dims)

    def copy(self, shallow=False):
        new = CuOperator(shape=self.shape, hilbert_dims=self.hilbert_dims)

        for term in self.terms:
            copy_term = Term([], factor=term.factor)

            for pterm in term.prod_terms:
                if isinstance(pterm.operator, _data.Data) and not shallow:
                    # DenseOperator do not have a copy method...
                    oper = pterm.operator.copy()
                else:
                    oper = pterm.operator
                copy_term.prod_terms.append(ProdTerm(
                    oper,
                    pterm.hilbert,
                    pterm.transform,
                ))
            new.terms.append(copy_term)
        return new

    def to_array(self):
        hilbert = self.hilbert_space_dims
        out = np.zeros(self.shape, dtype=complex)

        for term in self.terms:
            termmat = np.eye(self.shape[0], dtype=complex) * term.factor
            for prod_term in term.prod_terms:
                mat = _oper_to_array(prod_term.operator, prod_term.transform)

                if isinstance(mat, cp.ndarray):
                    # if cupy array, get numpy
                    mat = mat.get()

                if len(mat.shape) > 2:
                    raise NotImplementedError("to_array parts must be 2d")

                idxs = list(range(len(hilbert)))

                sizes = []
                for i in prod_term.hilbert:
                    sizes.append(hilbert[i])
                for i in reversed(sorted(prod_term.hilbert)):
                    del idxs[i]

                for i in idxs:
                    N = hilbert[i]
                    mat = np.kron(mat, np.eye(N))
                    sizes.append(N)

                mat = _data.permute.dimensions(
                    _data.Dense(mat),
                    sizes,
                    np.argsort(list(prod_term.hilbert) + idxs),
                    dtype=_data.Dense,
                ).to_array()

                termmat = mat @ termmat
            out += termmat
        return out

    def conj(self):
        new = CuOperator(shape=self.shape, hilbert_dims=self.hilbert_dims)
        for term in self.terms:
            copy_term = Term([], factor=term.factor.conjugate())
            for pterm in term.prod_terms:
                copy_term.prod_terms.append(ProdTerm(
                    pterm.operator,
                    pterm.hilbert,
                    conj_transform[pterm.transform],
                ))
            new.terms.append(copy_term)
        return new

    def transpose(self):
        new = CuOperator(shape=self.shape, hilbert_dims=self.hilbert_dims)
        for term in self.terms:
            copy_term = Term([], factor=term.factor)
            for pterm in term.prod_terms[::-1]:
                copy_term.prod_terms.append(ProdTerm(
                    pterm.operator,
                    pterm.hilbert,
                    trans_transform[pterm.transform],
                ))
            new.terms.append(copy_term)
        return new

    def adjoint(self):
        new = CuOperator(shape=self.shape, hilbert_dims=self.hilbert_dims)
        for term in self.terms:
            copy_term = Term([], factor=term.factor.conjugate())
            for pterm in term.prod_terms[::-1]:
                copy_term.prod_terms.append(ProdTerm(
                    pterm.operator,
                    pterm.hilbert,
                    adjoint_transform[pterm.transform],
                ))
            new.terms.append(copy_term)
        return new

    def __neg__(self):
        return self * -1

    def __add__(self, other):
        if not isinstance(other, CuOperator):
            if isinstance(other, Data):
                return _data.add(self, other)
            return NotImplemented

        if not _compare_hilbert(self.hilbert_dims, other.hilbert_dims):
            raise ValueError("Incompatible Hilbert spaces")
        new = self.copy()
        new._update_hilbert(other.hilbert_dims)
        other = other.copy()
        other._update_hilbert(new.hilbert_dims)
        new.terms += other.terms

        return new

    def __sub__(self, other):
        if not isinstance(other, CuOperator):
            if isinstance(other, Data):
                return _data.sub(self, other)
            return NotImplemented

        if not _compare_hilbert(self.hilbert_dims, other.hilbert_dims):
            raise ValueError()
        return self + -other

    def __mul__(self, other):
        new = CuOperator(shape=self.shape, hilbert_dims=self.hilbert_dims)
        for term in self.terms:
            copy_term = Term([], factor=term.factor * other)
            for pterm in term.prod_terms:
                copy_term.prod_terms.append(ProdTerm(
                    pterm.operator,
                    pterm.hilbert,
                    pterm.transform,
                ))
            new.terms.append(copy_term)
        return new

    def __truediv__(self, other):
        return self * (1 / other)

    def __matmul__(self, other):
        if not isinstance(other, CuOperator):
            if isinstance(other, Data):
                return _data.matmul(self, other)
            return NotImplemented

        if not _compare_hilbert(self.hilbert_dims, other.hilbert_dims):
            raise ValueError("Incompatible Hilbert spaces.")
        left = self.copy()
        left._update_hilbert(other.hilbert_dims)
        right = other.copy()
        right._update_hilbert(self.hilbert_dims)

        new = CuOperator(shape=self.shape, hilbert_dims=left.hilbert_dims)

        for term_left, term_right in itertools.product(left.terms, right.terms):
            new.terms.append(
                Term(
                    term_right.prod_terms + term_left.prod_terms,
                    term_left.factor * term_right.factor,
                )
            )

        return new

    def to_OperatorTerm(self, dual=False, copy=True, hilbert_dims=None):
        # TODO: Dim input instead of dual?
        if hilbert_dims is not None:
            self = self.copy()
            if dual:
                hilbert_dims = hilbert_dims + hilbert_dims
            self._update_hilbert(hilbert_dims)

        out = OperatorTerm(dtype="complex128")
        if not dual:
            for term in self.terms:
                cuterm = tensor_product(dtype="complex128")
                for pterm in term.prod_terms:
                    oper = _oper_to_ElementaryOperator(
                        pterm.operator,
                        pterm.hilbert,
                        self.hilbert_space_dims,
                        pterm.transform,
                        copy
                    )
                    # Inverted order confirmed by nvidia
                    cuterm = cuterm * tensor_product((oper, pterm.hilbert))
                out = out + (cuterm * term.factor)

        else:
            N_hilbert = len(self.hilbert_dims) // 2
            # TODO: make this tests weak compare?
            assert self.hilbert_dims[:N_hilbert] == self.hilbert_dims[N_hilbert:]
            for term in self.terms:
                cuterm = tensor_product(dtype="complex128")
                for pterm in term.prod_terms:
                    if all(i < N_hilbert for i in pterm.hilbert):
                        oper = _oper_to_ElementaryOperator(
                            pterm.operator,
                            pterm.hilbert,
                            self.hilbert_space_dims,
                            trans_transform[pterm.transform],
                            copy
                        )
                        # Inverted order confirmed by nvidia
                        cuterm = cuterm * tensor_product(
                            (oper, pterm.hilbert, (True,))
                        )

                    elif any(i < N_hilbert for i in pterm.hilbert):
                        raise NotImplementedError(
                            "Operators acting on both original and "
                            "dual spaces are not supported."
                        )

                    else:
                        oper = _oper_to_ElementaryOperator(
                            pterm.operator,
                            pterm.hilbert,
                            self.hilbert_space_dims,
                            pterm.transform,
                            copy
                        )
                        cuterm = cuterm * tensor_product(
                            (oper, tuple(i - N_hilbert for i in pterm.hilbert))
                        )

                out = out + (cuterm * term.factor)

        return out


def CuOperator_from_Dia(mat):
    if mat.shape[0] != mat.shape[1]:
        # TODO: This break the function
        raise ValueError("Rectangular CuOperator are not supported")
    return CuOperator(mat)


def CuOperator_from_Dense(mat):
    if mat.shape[0] != mat.shape[1]:
        # TODO: This break the function
        raise ValueError("Rectangular CuOperator are not supported")
    return CuOperator(mat)


def Dense_from_CuOperator(mat):
    print("converting to Dense")
    return _data.Dense(mat.to_array())


_data.to.add_conversions(
    [
        (CuOperator, _data.Dense, CuOperator_from_Dense),
        (CuOperator, _data.Dia, CuOperator_from_Dia),
        (_data.Dense, CuOperator, Dense_from_CuOperator, 10),
    ]
)
_data.to.register_aliases(
    ["densitymat_OperatorTerm", "CuOperator"],
    CuOperator
)


@_data.identity.register(CuOperator)
def identity_CuOperator(dimension, scale=1):
    new = CuOperator(shape=(dimension, dimension))
    new.terms.append(Term([], complex(scale)))
    return new


@_data.zeros.register(CuOperator)
def zeros_CuOperator(rows, cols):
    if rows != cols: raise ValueError("Zero operator must be square.")
    return CuOperator(shape=(rows, cols))


@_data.diag.register(CuOperator)
def diags_CuOperator(diagonals, offsets=None, shape=None):
    return CuOperator(_data.dia.diags(diagonals, offsets, shape))


@_data.kron.register(CuOperator)
def kron_CuOperator(left, right):
    N = len(left.hilbert_dims)
    S = left.shape[0] * right.shape[0]
    new_hilbert = left.hilbert_dims + right.hilbert_dims

    left_ext = CuOperator(shape=(S, S), hilbert_dims=new_hilbert)
    for term in left.terms:
        copy_term = Term([], factor=term.factor)
        for pterm in term.prod_terms:
            copy_term.prod_terms.append(ProdTerm(
                pterm.operator,
                pterm.hilbert,
                pterm.transform,
            ))
        left_ext.terms.append(copy_term)

    right_shifted = CuOperator(shape=(S, S), hilbert_dims=new_hilbert)
    for term in right.terms:
        copy_term = Term([], factor=term.factor)
        for pterm in term.prod_terms:
            copy_term.prod_terms.append(ProdTerm(
                pterm.operator,
                tuple(i + N for i in pterm.hilbert),
                pterm.transform,
            ))
        right_shifted.terms.append(copy_term)

    return left_ext @ right_shifted


@_data.extract.register(CuOperator)
def extract_CuOperator(mat, format=None, copy=True):
    # TODO: Better name, other input for dual?
    if format not in [None, "OperatorTerm", "DualOperatorTerm"]:
        raise ValueError(...)

    dual = format == "DualOperatorTerm"
    return mat.to_OperatorTerm(dual=dual, copy=copy)


@_data.permute.dimensions.register(CuOperator)
def dimensions_CuOperator(matrix, hilbert, order):
    if not _compare_hilbert(matrix.hilbert_dims, hilbert):
        # Assumes hilbert is a full representation
        raise ValueError(
            f"Input hilbert dimensions {hilbert} are not "
            f"compatible with matrix's hilbert_dims {matrix.hilbert_dims}."
        )

    new = CuOperator(shape=matrix.shape)
    permutation = np.argsort(order)
    new.hilbert_dims = tuple(matrix.hilbert_dims[i] for i in order)

    for term in matrix.terms:
        copy_term = Term([], factor=term.factor)
        for pterm in term.prod_terms:
            copy_term.prod_terms.append(ProdTerm(
                pterm.operator.copy(),
                tuple(int(permutation[mode]) for mode in pterm.hilbert),
                pterm.transform,
            ))
        new.terms.append(copy_term)
    return new


@_data.isequal.register(CuOperator)
def isequal_CuOperator(left, right, atol=-1, rtol=-1):
    if not _compare_hilbert(left.hilbert_dims, right.hilbert_dims):
        return False
    if atol < 0:
        atol = settings.core["atol"]
    if rtol < 0:
        rtol = settings.core["rtol"]
    # TODO: Implement real one
    if left.shape[0] > 1000:
        return left is right
    return np.allclose(left.to_array(), right.to_array(), rtol, atol)


@_data.isherm.register(CuOperator)
def isherm(operator, tol=None):
    if operator.shape[0] != operator.shape[1]:
        return False
    # TODO: Ideally we should check simple case without merging.
    oper = operator.to_array()
    if tol < 0:
        tol = settings.core["atol"]
    return cp.allclose(oper, oper.T.conj(), atol=tol)
###############################################################################
###############################################################################


_data.adjoint.add_specialisations([
    (CuOperator, CuOperator, CuOperator.adjoint),
])

_data.transpose.add_specialisations([
    (CuOperator, CuOperator, CuOperator.transpose),
])

_data.conj.add_specialisations([
    (CuOperator, CuOperator, CuOperator.conj),
])

_data.mul.add_specialisations([
    (CuOperator, CuOperator, CuOperator.__mul__),
])

_data.neg.add_specialisations([
    (CuOperator, CuOperator, CuOperator.__neg__),
])

_data.matmul.add_specialisations([(
    CuOperator, CuOperator, CuOperator,
    lambda left, right, scale=1.: left @ right * scale
)])

_data.add.add_specialisations([(
    CuOperator, CuOperator, CuOperator,
    lambda left, right, scale=1.: left + right * scale
)])

_data.sub.add_specialisations([
    (CuOperator, CuOperator, CuOperator, CuOperator.__sub__),
])
