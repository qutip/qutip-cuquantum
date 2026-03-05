from typing import Any
from cuquantum.densitymat import DensePureState, DenseMixedState

import numpy as np
import cupy as cp

from qutip.core.data import Data
from qutip.core import data as _data
from qutip import settings

try:
    from qutip_cupy import CuPyDense
except ImportError:
    CuPyDense = None

try:
    import mpi4py.MPI as MPI
except ImportError:
    MPI = None


class CuState(Data):
    def __init__(self, arg, hilbert_dims=None, shape=None, copy=True):
        ctx = settings.cuDensity["ctx"]
        if hilbert_dims is not None and any(d < 0 for d in hilbert_dims):
            raise ValueError("weak hilbert dims not supported for CuState")

        if isinstance(arg, (DensePureState, DenseMixedState)):
            if shape is not None:
                pass
            elif isinstance(arg, DensePureState):
                shape = (np.prod(arg.hilbert_space_dims), 1)
            elif isinstance(arg, DenseMixedState):
                shape = (np.prod(arg.hilbert_space_dims),) * 2
            if hilbert_dims is None:
                hilbert_dims = arg.hilbert_space_dims
            base = arg

        elif isinstance(arg, CuState):
            if shape is not None:
                assert arg.shape == shape
            else:
                shape = arg.shape
            if hilbert_dims is not None:
                assert arg.base.hilbert_space_dims == hilbert_dims
            if copy:
                arg = arg.copy()
            base = arg.base

        elif (CuPyDense is not None and isinstance(arg, CuPyDense)) or isinstance(arg, cp.ndarray):
            if CuPyDense is not None and isinstance(arg, CuPyDense):
                arg = arg._cp

            if arg.ndim == 1:
                arg = arg.reshape(-1, 1)
            elif arg.ndim > 2:
                raise ValueError("Only 1D or 2D arrays are supported")

            if shape is None:
                shape = arg.shape
            if hilbert_dims is None:
                if(arg.shape[0] == 1):
                    hilbert_dims = (arg.shape[1],)
                else:
                    hilbert_dims = (arg.shape[0],)

            if arg.shape[0] != 1 and arg.shape[1] != 1:
                is_hilbert_dim_matching = (arg.shape[0] == np.prod(hilbert_dims) and arg.shape[1] == np.prod(hilbert_dims))
                if not is_hilbert_dim_matching:
                    raise ValueError(f"Shape {arg.shape} does not match hilbert_dims {hilbert_dims} for mixed state")
                base = DenseMixedState(ctx, hilbert_dims, 1, "complex128")
                sizes, offsets = base.local_info
                sls = tuple(slice(s, s+n) for s, n in zip(offsets, sizes))[:-1]
                N = np.prod(sizes)
                if len(arg) == N:
                    base.attach_storage(cp.array(
                        arg
                        .reshape(hilbert_dims * 2)[sls]
                        .ravel(order="F"),
                        dtype="complex128",
                        copy=copy
                    ))
                else:
                    base.allocate_storage()
                    base.storage[:N] = (
                        arg
                        .reshape(hilbert_dims * 2)[sls]
                        .ravel(order="F")
                    )

            else:
                is_hilbert_dim_matching = ((arg.shape[1] == 1 and arg.shape[0] == np.prod(hilbert_dims)) or
                                           (arg.shape[0] == 1 and arg.shape[1] == np.prod(hilbert_dims)))
                if not is_hilbert_dim_matching:
                    raise ValueError(f"Shape {arg.shape} does not match hilbert_dims {hilbert_dims} for pure state")

                base = DensePureState(ctx, hilbert_dims, 1, "complex128")
                sizes, offsets = base.local_info
                sls = tuple(slice(s, s+n) for s, n in zip(offsets, sizes))[:-1]
                N = np.prod(sizes)
                if len(arg) == N:
                    base.attach_storage(cp.array(
                        arg
                        .reshape(hilbert_dims)[sls]
                        .ravel(order="F"),
                        dtype="complex128",
                        copy=copy
                    ))
                else:
                    base.allocate_storage()
                    base.storage[:N] = (
                        arg
                        .reshape(hilbert_dims)[sls]
                        .ravel(order="F")
                    )

        elif isinstance(arg, Data):
            # TODO: Allocate CSR/Dia without intermediate dense representation
            arg = _data.to(_data.Dense, arg)

            if shape is None:
                shape = arg.shape
            if hilbert_dims is None:
                hilbert_dims = arg.shape[:1]

            if arg.shape[0] != np.prod(hilbert_dims) or arg.shape[1] != 1:
                # TODO: Add sanity check for hilbert_dims
                base = DenseMixedState(ctx, hilbert_dims, 1, "complex128")
                sizes, offsets = base.local_info
                sls = tuple(slice(s, s+n) for s, n in zip(offsets, sizes))[:-1]
                N = np.prod(sizes)
                arr_np = (
                    arg.to_array().reshape(hilbert_dims * 2)[sls].ravel("F")
                )
                base.allocate_storage()
                base.storage[:N] = cp.array(arr_np)

            else:
                base = DensePureState(ctx, hilbert_dims, 1, "complex128")
                sizes, offsets = base.local_info
                sls = tuple(slice(s, s+n) for s, n in zip(offsets, sizes))[:-1]
                N = np.prod(sizes)
                arr_np = arg.to_array().reshape(hilbert_dims)[sls].ravel("F")
                base.allocate_storage()
                base.storage[:N] = cp.array(arr_np)

        else:
            raise NotImplementedError(type(arg))

        self.base = base
        super().__init__(shape=shape)

    def copy(self):
        return CuState(
            self.base.clone(cp.array(self.base.storage, copy=True)),
            shape=self.shape
        )

    def to_array(self, as_tensor=False):
        return self.to_cupy(as_tensor).get()

    def to_cupy(self, as_tensor=False):
        if type(self.base) is DenseMixedState:
            tensor_shape = self.base.hilbert_space_dims * 2
        else:
            tensor_shape = self.base.hilbert_space_dims

        local_tensor = self.base.view()[..., 0]
        if self.base.local_info[0][:-1] != tensor_shape:
            if MPI is None:
                raise ImportError("mpi4py is not imported. Distributed tensor assembly requires mpi4py.")
            comm = MPI.COMM_WORLD
            tensor = cp.empty(tensor_shape, dtype=cp.complex128)
            sizes, offsets = self.base.local_info
            local_sls = tuple(slice(s, s+n) for s, n in zip(offsets, sizes))[:-1]
            all_sls = comm.allgather(local_sls)
            all_tensor = comm.allgather(local_tensor)
            for rank in range(comm.Get_size()):
                tensor[all_sls[rank]] = all_tensor[rank]
        else:
            tensor = local_tensor
        if not as_tensor:
            tensor = tensor.reshape(*self.shape, order="C")
        return tensor

    def __neg__(self):
        return self * -1

    def __add__(self, other):
        if not isinstance(other, CuState):
            if isinstance(other, Data):
                return _data.add(self, other)
            return NotImplemented
        if(self.shape != other.shape):
            raise ValueError("Incompatible shapes")
        new = self.copy()
        new.base.inplace_accumulate(other.base, 1.)
        return new

    def __sub__(self, other):
        if not isinstance(other, CuState):
            if isinstance(other, Data):
                return _data.sub(self, other)
            return NotImplemented

        if(self.shape != other.shape):
            raise ValueError("Incompatible shapes")
        new = self.copy()
        new.base.inplace_accumulate(other.base, -1.)
        return new

    def __mul__(self, other):
        new = self.copy()
        new.base.inplace_scale(other)
        return new

    def __div__(self, other):
        return self * (1 / other)

    def conj(self):
        return CuState(
            self.base.clone(cp.array(self.base.storage.conj(), copy=False)),
            shape=self.shape
        )

    def transpose(self):
        arr = self.to_cupy().transpose()
        return CuState(arr, hilbert_dims=self.base.hilbert_space_dims, shape=(self.shape[1], self.shape[0]))


    def adjoint(self):
        arr = self.to_cupy().transpose().conj()
        return CuState(arr, hilbert_dims=self.base.hilbert_space_dims, shape=(self.shape[1], self.shape[0]))


def CuState_from_Dense(mat):
    return CuState(mat)


def Dense_from_CuState(mat):
    print("Dense_from_CuState")
    return _data.Dense(mat.to_array())


_data.to.add_conversions(
    [
        (CuState, _data.Dense, CuState_from_Dense),
        (_data.Dense, CuState, Dense_from_CuState),
    ]
)

_data.to.register_aliases(["CuState"], CuState)


@_data.trace.register(CuState)
def trace_cuState(mat):
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(...)

    return complex(mat.base.trace()[0])


@_data.inner.register(CuState)
def inner_cuState(left, right, scalar_is_ket=False):
    if left.shape == (1, 1) and not scalar_is_ket:
        inner = left.base.storage[0] * right.base.storage[0]
    else:
        inner = left.base.inner_product(right.base)[0]
    return complex(inner)


@_data.mul.register(CuState)
def mul_cuState(mat, val):
    return mat * val


@_data.add.register(CuState)
def add_cuState(left, right, scale=1.):
    if left.base.hilbert_space_dims != right.base.hilbert_space_dims:
        raise ValueError(
            f"Incompatible hilbert space: {left.base.hilbert_space_dims} "
            f"and {right.base.hilbert_space_dims}."
        )
    out = left.copy()
    out.base.inplace_accumulate(right.base, scale)
    return out


@_data.imul.register(CuState)
def imul_cuState(mat, val):
    mat.base.inplace_scale(val)
    return mat


@_data.iadd.register(CuState)
def iadd_cuState(left, right, scale=1.):
    left.base.inplace_accumulate(right.base, scale)
    return left


@_data.norm.frobenius.register(CuState)
def frobenius_cuState(mat):
    return float(mat.base.norm()[0])**0.5


@_data.norm.l2.register(CuState)
def l2_cuState(matrix):
    if matrix.shape[0] != 1 and matrix.shape[1] != 1:
        raise ValueError("L2 norm is only defined on vectors")
    return frobenius_cuState(matrix)


@_data.ode.wrmn_error.register(CuState)
def wrmn_error_cuState(diff, state, atol, rtol):
    if diff.base.hilbert_space_dims != state.base.hilbert_space_dims:
        raise ValueError(
            f"Incompatible hilbert space: {diff.base.hilbert_space_dims} "
            f"and {state.base.hilbert_space_dims}."
        )
    diff.base.storage[:] = cp.abs(diff.base.storage)
    diff.base.storage[:] = diff.base.storage / (atol + rtol * cp.abs(state.base.storage))
    return float(diff.base.norm()[0] / (diff.shape[0] * diff.shape[1]))**0.5


@_data.reshape.register(CuState)
def reshape_stack(matrix):
    raise NotImplementedError(...)
    return matrix


@_data.column_stack.register(CuState)
def column_stack(matrix):
    return CuState(matrix.base, shape=(matrix.shape[0] * matrix.shape[1], 1))


@_data.column_unstack.register(CuState)
def column_unstack(matrix, rows):
    return CuState(matrix.base, shape=(matrix.shape[0] / rows, rows))


@_data.isherm.register(CuState)
def isherm(state, tol=-1):
    if state.shape[0] != state.shape[1]:
        return False
    if settings.cuDensity["ctx"].get_num_ranks() != 1:
        # MPI, Not Implemented yet.
        return None
    cupy_view = state.to_cupy()
    if tol < 0:
        tol = settings.core["atol"]
    return cp.allclose(cupy_view, cupy_view.T.conj(), atol=tol)


def zeros_like_cuState(state):
    return CuState(state.base.clone(cp.zeros_like(state.base.storage, order="F")))

@_data.conj.register(CuState)
def conj_cuState(state):
    return state.conj()

@_data.transpose.register(CuState)
def transpose_cuState(state):
    return state.transpose()

@_data.adjoint.register(CuState)
def adjoint_cuState(state):
    return state.adjoint()

@_data.sub.register(CuState)
def sub_cuState(left, right):
    return add_cuState(left, right, -1)

@_data.iszero.register(CuState)
def iszero_cuState(state):
    return not cp.any(state.base.storage)


@_data.matmul.register(CuState)
def matmul_cuState(left, right):
    if(left.shape[1] != right.shape[0]):
        raise ValueError("Incompatible shapes")

    if left.base.hilbert_space_dims != right.base.hilbert_space_dims:
        raise ValueError(
            f"Incompatible hilbert space: {left.base.hilbert_space_dims} "
            f"and {right.base.hilbert_space_dims}."
        )

    output_shape = (left.shape[0], right.shape[1])
    ctx = settings.cuDensity["ctx"]
    if(left.shape[0] == 1 and right.shape[1] == 1):
        # Scalar case
        hilbert_dims = (1,)
    else:
        hilbert_dims = left.base.hilbert_space_dims

    left_array = left.to_cupy()
    right_array = right.to_cupy()
    arr = left_array @ right_array

    return CuState(arr, hilbert_dims=hilbert_dims, shape=output_shape)
