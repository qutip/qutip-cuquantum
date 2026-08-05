from qutip.solver.heom import HEOMSolver
from qutip.solver.solver_base import Solver
from time import time
import numpy as np
from qutip import Qobj, QobjEvo, liouvillian
from qutip.solver.heom.bofin_solvers import HierarchyADOs, HierarchyADOsState
from qutip.solver.integrator.qutip_integrator import IntegratorVern7
from qutip import settings

from qutip.solver.heom.bofin_solvers import HierarchyADOs
from ..operator import CuOperator
from ..state import CuState
from .rhs import CuHEOMRhs
import cupy as cp


def _matrix_to_storage(rho, hilbert_dims):
    """
    Flatten a ``(D, D)`` system density matrix into cuDensity's storage order.

    A ``DenseMixedState`` over ``hilbert_dims = (d1, ..., dn)`` stores the
    F-ordered ``(d1, ..., dn, d1, ..., dn)`` tensor, which only coincides with
    the F-ordered ``(D, D)`` matrix when there is a single mode.
    """
    return rho.reshape(tuple(hilbert_dims) * 2).ravel("F")


def _storage_to_matrix(storage, hilbert_dims):
    """Inverse of :func:`_matrix_to_storage`, returning a ``(D, D)`` array."""
    D = int(np.prod(hilbert_dims))
    tensor = storage.reshape(tuple(hilbert_dims) * 2, order="F")
    return tensor.reshape(D, D)


class CuHierarchyADOsState(HierarchyADOsState):
    def __init__(self, ados, ado_state, hilbert_dims, sys_dim):
        self.hilbert_dims = tuple(hilbert_dims)
        self.D = int(np.prod(self.hilbert_dims))
        self.sys_dim = sys_dim
        self._ados = ados
        self._ado_state = ado_state
        self.rho = self.extract(0)

    def __getattr__(self, name):
        if name == "_ados":
            # avoid infinite recursion
            raise AttributeError(name)
        return getattr(self._ados, name)

    def extract(self, idx_or_label):
        if isinstance(idx_or_label, int):
            idx = idx_or_label
        else:
            idx = self._ados.idx(idx_or_label)
        D2 = self.D ** 2
        storage = self._ado_state.base.storage[idx * D2:(idx + 1) * D2].get()
        return Qobj(
            _storage_to_matrix(storage, self.hilbert_dims), dims=self.sys_dim
        )


class CuHEOMSolver(HEOMSolver):
    solver_options = {
        "progress_bar": "text",
        "progress_kwargs": {"chunk_size": 10},
        "store_final_state": False,
        "store_states": None,
        "normalize_output": False,
        "method": "vern7",
        "store_ados": False,
        "state_data_type": "CuState",
    }

    def __init__(self, H, bath, max_depth, *, odd_parity=False, options=None):
        _time_start = time()
        self.odd_parity = bool(odd_parity)

        if not isinstance(H, (Qobj, QobjEvo)):
            raise TypeError("The Hamiltonian (H) must be a Qobj or QobjEvo")

        H = QobjEvo(H.to(CuOperator))
        self.L_sys = liouvillian(H) if H.type == "oper" else H

        self._sys_shape = int(np.sqrt(self.L_sys.shape[0]))
        self._sup_shape = self.L_sys.shape[0]
        self._hilbert_dims = tuple(self.L_sys.dims[0][0])

        self.ados = HierarchyADOs(
            self._combine_bath_exponents(bath), max_depth,
        )
        self._n_ados = len(self.ados.labels)
        self._n_exponents = len(self.ados.exponents)

        if any(exp.fermionic for exp in self.ados.exponents):
            raise NotImplementedError("Fermionic exponents not supported.")

        self._init_ados_time = time() - _time_start
        _time_start = time()

        rhs = self._calculate_rhs()

        self._init_superop_cache_time = 0.0          # nothing to cache
        self._init_rhs_time = time() - _time_start

        Solver.__init__(self, rhs, options=options)  # NB: skip HEOMSolver.__init__


    def _get_integrator(self):
        # Only the explicit Runge-Kutta integrators drive the system purely
        # through ``matmul_data`` while keeping the state in its own data
        # layer, which is what lets the hierarchy stay on the GPU. The others
        # either need the generator as a matrix, which CuHEOMRhs cannot build,
        # or hand the state back as a dense host array. ``IntegratorVern7`` is
        # the base of vern9 and tsit5, so it identifies the whole family.
        # Checked before instantiating, because some integrators fail inside
        # their own setup first and would mask this message.
        method = self._options["method"]
        integrator = self.avail_integrators().get(method, method)
        if not (
            isinstance(integrator, type)
            and issubclass(integrator, IntegratorVern7)
        ):
            supported = sorted(
                name
                for name, cls in self.avail_integrators().items()
                if issubclass(cls, IntegratorVern7)
            )
            raise ValueError(
                f"CuHEOMSolver requires an explicit Runge-Kutta integrator, "
                f"but method={method!r} is not one of {supported}."
            )
        return super()._get_integrator()

    def steady_state(self, use_mkl=True, mkl_max_iter_refine=100, mkl_weighted_matching=False):
        raise NotImplementedError("steady_state is not supported for CuHEOMSolver.")

    def _calculate_rhs(self):
        ctx = settings.cuDensity["ctx"]
        if ctx is None:
            raise RuntimeError(
                "CuHEOMSolver requires a cuQuantum WorkStream. "
                "Call qutip_cuquantum.set_as_default(ctx) first."
            )
        return CuHEOMRhs(ctx, self.L_sys, self.ados)

    def _prepare_state(self, state):
        if(isinstance(state, Qobj)):
            rho0 = state
            D2 = self._sys_shape ** 2
            if rho0._dims != self._sys_dims :
                raise ValueError(
                    f"Initial state rho has dims {rho0.dims}"
                    f" but the system dims are {self._sys_dims}"
                )
            arr = cp.zeros([D2 * self._n_ados], dtype=cp.complex128)
            arr[:D2] = cp.asarray(
                _matrix_to_storage(rho0.full(), self._hilbert_dims),
                dtype=cp.complex128,
            )
            return CuState(arr)
        else:
            if(isinstance(state, CuHierarchyADOsState)):
                return state._ado_state
            else:
                raise TypeError(
                    f"Initial ADOs passed have type {type(state)}"
                    " but a CuHierarchyADOsState instance"
                    " was expected"
                )



    def _restore_state(self, state, *, copy=True):
        if(copy):
            state = state.copy()
        return CuHierarchyADOsState(
            self.ados, state, self._hilbert_dims, self._sys_dims
        )

