from enum import Enum
from cuquantum.densitymat import Operator, DensePureState

__all__ = [
    "Transform",
    "conj_transform",
    "trans_transform",
    "adjoint_transform",
    "_compare_hilbert"
]


class Transform(Enum):
    DIRECT = 0
    CONJ = 1
    TRANSPOSE = 2
    ADJOINT = 3


conj_transform = {
    Transform.DIRECT : Transform.CONJ,
    Transform.CONJ : Transform.DIRECT,
    Transform.TRANSPOSE : Transform.ADJOINT,
    Transform.ADJOINT : Transform.TRANSPOSE,
}

trans_transform = {
    Transform.DIRECT : Transform.TRANSPOSE,
    Transform.CONJ : Transform.ADJOINT,
    Transform.TRANSPOSE : Transform.DIRECT,
    Transform.ADJOINT : Transform.CONJ,
}

adjoint_transform = {
    Transform.DIRECT : Transform.ADJOINT,
    Transform.CONJ : Transform.TRANSPOSE,
    Transform.TRANSPOSE : Transform.CONJ,
    Transform.ADJOINT : Transform.DIRECT,
}


def _compare_hilbert(left, right, return_shifts=False):

    def _set(hilbert, idx, shift):
        if len(hilbert) > idx:
            shift[idx] = []
            return hilbert[idx]
        return None

    ptr_l = 0
    ptr_r = 0
    out_hilbert = []
    shifts_left = {}
    shifts_right = {}

    size_l = _set(left, ptr_l, shifts_left)
    size_r = _set(right, ptr_r, shifts_right)

    while ptr_l < len(left) and ptr_r < len(right):
        shifts_left[ptr_l].append(len(out_hilbert))
        shifts_right[ptr_r].append(len(out_hilbert))

        if abs(size_l) == abs(size_r):
            out_hilbert.append(max(size_l, size_r))
            ptr_l += 1
            ptr_r += 1
            size_l = _set(left, ptr_l, shifts_left)
            size_r = _set(right, ptr_r, shifts_right)

        elif -size_l > abs(size_r):
            if size_l % size_r != 0:
                return False
            out_hilbert.append(size_r)
            size_l = -abs(size_l // size_r)
            ptr_r += 1
            size_r = _set(right, ptr_r, shifts_right)

        elif abs(size_l) < -size_r:
            if size_r % size_l != 0:
                return False
            out_hilbert.append(size_l)
            size_r = -abs(size_r // size_l)
            ptr_l += 1
            size_l = _set(left, ptr_l, shifts_left)

        else:
            return False

        if size_l is None and size_r is None:
            break
        elif size_l is None or size_r is None:
            return False

    if return_shifts:
        return tuple(out_hilbert), shifts_left, shifts_right
    else:
        return tuple(out_hilbert)


def Oper_to_cupy(oper, ctx):
    dims = oper.hilbert_space_dims
    N = np.prod(dims)
    id_ = DensePureState(ctx, dims, N, "complex128")
    id_.allocate_storage()
    id_.storage[::N+1] = 1
    out = DensePureState(ctx, dims, N, "complex128")
    out.allocate_storage()
    
    oper.prepare_action(ctx, id_)
    oper.compute_action(0., None, id_, out)
    return out.view()


def Operterm_to_cupy(term, hdims, ctx):
    oper = Operator(hdims, (term,))
    return Oper_to_cupy(oper, ctx)



