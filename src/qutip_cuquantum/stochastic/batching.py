import cupy as cp
from cuquantum.densitymat import WorkStream
from ..state import CuState
from qutip import settings


def get_batch_size(
    ctx : WorkStream,
    state: CuState,
    headroom: float = 0.10,
    num_copies: int = 1,
) -> float:
    """
    Estimate the optimal number of state to batch in one run of a
    non-deterministic solver.

    Parameters
    ----------
    ctx:
        The cuDensityMat Workstream context.

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

    # sizes, offsets = batched.local_info
    # sls = tuple(slice(s, s+n) for s, n in zip(offsets, sizes))[:-1]
    # N = np.prod(sizes)

    for i in range(batch_size):
        batched.view()[..., i] = state.base.view()[..., 0]

    return CuState(batched, copy=False, shape=state.shape)
