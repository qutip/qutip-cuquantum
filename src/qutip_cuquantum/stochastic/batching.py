import cupy as cp
from cuquantum.densitymat import WorkStream
from ..state import CuState
from qutip import settings
from qutip.core.cy.coefficient import FunctionCoefficient, Coefficient

__all__ = ["get_max_batch_size"]

def get_max_batch_size(
    state: CuState,
    headroom: float = 0.10,
    num_copies: int = 1,
) -> float:
    """
    Estimate the optimal number of state to batch in one run of a
    non-deterministic solver.

    Parameters
    ----------
    state:
        Sample state with batch size of 1.

    headroom:
        GPU mem to leave unused as a fraction.

    num_copies:
        Number of copies of the states required at once. Includes the
        derivatives needed by the SDE / ODE solver, previous states stored in
        results, etc.
    """
    device = cp.cuda.Device()
    free_mem, total_mem = device.mem_info
    usable_mem = free_mem * (1 - headroom)

    size = state.base.storage_size
    bytes = size * 16  # Complex128

    max_batch_size = usable_mem / bytes / num_copies
    return max_batch_size


def batch_copy(state: CuState, batch_size: int):
    """
    Make a batched state filled with ``batch_size`` copies of ``state``.
    """
    # TODO: No mpi support
    ctx = settings.cuDensity["ctx"]
    cls = state.base.__class__
    hilbert_space_dims = state.base.hilbert_space_dims
    batched = cls(ctx, hilbert_space_dims, batch_size, "complex128")
    batched.allocate_storage()

    if not all(offset == 0 for offset in batched.local_info[1]):
        raise NotImplementedError

    # sizes, offsets = batched.local_info
    # sls = tuple(slice(s, s+n) for s, n in zip(offsets, sizes))[:-1]
    # N = np.prod(sizes)

    for i in range(batch_size):
        batched.view()[..., i] = state.base.view()[..., 0]

    return CuState(batched, copy=False, shape=state.shape)


def split_batch(state: CuState):
    """
    Return a list of unbatched states
    """
    # TODO: No mpi support
    if not all(offset == 0 for offset in state.local_info[1]):
        raise NotImplementedError

    ctx = settings.cuDensity["ctx"]
    cls = state.base.__class__
    hilbert_space_dims = state.base.hilbert_space_dims
    num_batch = state.base.batch_size
    elements = [
        cls(ctx, hilbert_space_dims, 1, "complex128")
        for _ in range(num_batch)
    ]
    for i, elem in enumerate(elements):
        elem.allocate_storage()
        elem.view()[..., 0] = state.view()[..., i]

    return [
        CuState(elem, copy=False, shape=state.shape)
        for elem in elements
    ]


class BatchCoefficient(FunctionCoefficient):
    def __init__(self, *args_, **kwargs):
        super().__init__(*args_, **kwargs)
        self._conj : bool = False
        self.other_mul : Coefficient = None
        self.other_add : Coefficient = None

    def __call__(self, t, **kw):
        return self._call(t, **kw)

    def _call(self, t, **kw):
        batch = super().__call__(t, **kw)
        if self._conj:
            batch = batch.conj()
        if self.other_mul:
            batch = batch * self.other_mul(t, **kw)
        if self.other_add:
            batch = batch + self.other_add(t, **kw)
        return batch

    def __add__(self, other):
        if not isinstance(other, Coefficient):
            return NotImplemented
        out = self.copy()
        out._conj = self._conj
        out.other_mul = self.other_mul
        if self.other_add:
            out.other_add = self.other_add + other
        else:
            out.other_add = other
        return out

    def __mul__(self, other):
        if not isinstance(other, Coefficient):
            return NotImplemented
        out = self.copy()
        out._conj = self._conj
        if self.other_mul:
            out.other_mul = self.other_mul * other
        else:
            out.other_mul = other
        if self.other_add:
            out.other_add = self.other_add * other
        return out

    def conj(self, other):
        if not isinstance(other, Coefficient):
            return NotImplemented
        out = self.copy()
        out._conj = not self._conj
        if self.other_mul:
            out.other_mul = self.other_mul.conj()
        if self.other_add:
            out.other_add = self.other_add.conj()
        return out

    def copy(self):
        _f_parameters, _f_pythonic, args, func, _ = self.__reduce__()[2]
        out = BatchCoefficient(
            func, args,
            _f_pythonic=_f_pythonic,
            _f_parameters=_f_parameters,
        )
        out.conj = self.conj
        out.other_mul = self.other_mul
        out.other_add = self.other_add
        return out
