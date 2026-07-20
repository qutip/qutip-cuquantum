from qutip.solver.heom import HEOMSolver
from qutip.solver.solver_base import Solver
from time import time
import numpy as np
from qutip import Qobj, QobjEvo, liouvillian
from qutip.solver.heom.bofin_solvers import HierarchyADOs, HierarchyADOsState
from qutip import settings

from qutip.solver.heom.bofin_solvers import HierarchyADOs
from ..operator import CuOperator
from ..state import CuState
from .rhs import CuHEOMRhs
import cupy as cp


class CuHierarchyADOsState(HierarchyADOsState):
    def __init__(self, ados, ado_state, sys_shape, sys_dim):
        self.D = sys_shape
        self.sys_dim = sys_dim
        self._ados = ados
        self._ado_state = ado_state
        rho_arr = ado_state.base.storage[:self.D**2].reshape((self.D, self.D), order="F").get()
        self.rho = Qobj(rho_arr, dims=sys_dim)

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
        arr = self._ado_state.base.storage[idx*self.D**2:(idx+1)*self.D**2].reshape((self.D, self.D), order="F").get()
        return Qobj(arr, dims=self.sys_dim)

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

        # if isinstance(H, (QobjEvo)):
        #     if not H.isconstant :
        #         raise NotImplementedError("Time-dependent H is not supported yet.")
        #     else:
        #         H = H(0)

        H = QobjEvo(H.to(CuOperator))
        self.L_sys = liouvillian(H) if H.type == "oper" else H
        

        
        self._sys_shape = int(np.sqrt(self.L_sys.shape[0]))
        self._sup_shape = self.L_sys.shape[0]

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
            n = self._sys_shape
            rho_dims = self._sys_dims            
            if rho0._dims != rho_dims:
                raise ValueError(
                    f"Initial state rho has dims {rho0.dims}"
                    f" but the system dims are {rho_dims}"
                )
            arr = cp.zeros([n ** 2 * self._n_ados], dtype=cp.complex128)
            arr[:n ** 2] = cp.asarray(rho0.full().ravel('F'), dtype=cp.complex128)
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
        return CuHierarchyADOsState(self.ados, state, self._sys_shape, self._sys_dims)

