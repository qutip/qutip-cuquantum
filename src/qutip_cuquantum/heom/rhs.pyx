#cython: language_level=3


from qutip.core.cy.qobjevo cimport QobjEvo
from qutip.core.data cimport Data

import numpy as np
from qutip.core.dimensions import Dimensions
from qutip.core.superoperator import spre, spost
from qutip.solver.heom.bofin_baths import BathExponent
from ..qobjevo import CuQobjEvo
from ..operator import CuOperator
from ..state import CuState, zeros_like_cuState
from qutip.core import data as _data
import cupy as cp
import cuquantum.densitymat as cudm

cdef class CuHEOMRhs(QobjEvo):
    cdef:
        dict __dict__    

    def __init__(self, ctx, Lsys, ados):
        # Initialize the QobjEvo base-class cdef state ourselves: we
        # intentionally bypass ``QobjEvo.__init__`` (CuHEOMRhs is not built
        # from elements/coefficients), but the base class -- and downstream
        # code in ``Solver.__init__`` -- assumes these containers exist.
        self.elements = []
        self._feedback_functions = {}
        self._solver_only_feedback = {}

        self._ctx = ctx
        N = len(ados.labels) * Lsys.shape[0]
        self._dims = Dimensions([[N], [N]])
        self.shape = (N, N)
        # ------------------------------------------------------------------
        # 1. System Hamiltonian / Liouvillian
        # ------------------------------------------------------------------
        self.dim = int(np.sqrt(Lsys.shape[0]))
        self.hilbert_dims = (self.dim,)

        # ------------------------------------------------------------------
        # 2. Bath / hierarchy
        # ------------------------------------------------------------------
        n_exponents = len(ados.exponents)
        for exp in ados.exponents:
            if exp.fermionic:
                raise NotImplementedError(
                    "Fermionic exponents are not supported."
                )

        sys_shape = ados.exponents[0].Q.shape[0]
        D2 = sys_shape ** 2
        n_ados = len(ados.labels)

        self._ados = ados
        self._n_exponents = n_exponents
        self._n_ados = n_ados
        self._sup_shape = D2

        # ------------------------------------------------------------------
        # 3. Per-(k, ADO) routing tables: indices and weights
        # ------------------------------------------------------------------
        prev_map = np.zeros((n_exponents, n_ados), dtype=int)
        next_map = np.zeros((n_exponents, n_ados), dtype=int)
        prev_w_minus = np.zeros((n_exponents, n_ados), dtype=complex)
        prev_w_plus = np.zeros((n_exponents, n_ados), dtype=complex)
        for he_n in ados.labels:
            n = ados.idx(he_n)
            for k in range(n_exponents):
                t = ados.exponents[k].type
                if t == BathExponent.types.R:
                    prev_w_minus[k, n] = -1j * ados.ck[k] * he_n[k]
                elif t == BathExponent.types.I:
                    prev_w_plus[k, n] = ados.ck[k] * he_n[k]
                elif t == BathExponent.types.RI:
                    prev_w_minus[k, n] = -1j * ados.ck[k] * he_n[k]
                    prev_w_plus[k, n] = ados.ck2[k] * he_n[k]
                else:
                    raise ValueError(f"Unsupported exponent type {t}")

                prev_he = ados.prev(he_n, k)
                prev_map[k, n] = (
                    ados.idx(prev_he) if prev_he is not None else n_ados
                )
                next_he = ados.next(he_n, k)
                next_map[k, n] = (
                    ados.idx(next_he) if next_he is not None else n_ados
                )

        self._prev_map_gpu = cp.asarray(prev_map, dtype=cp.intp)
        self._next_map_gpu = cp.asarray(next_map, dtype=cp.intp)
        self._prev_w_minus_gpu = cp.asarray(
            prev_w_minus, dtype=cp.complex128,
        )
        self._prev_w_plus_gpu = cp.asarray(
            prev_w_plus, dtype=cp.complex128,
        )
        self._ones_neg_j = cp.full(n_ados, -1j, dtype=cp.complex128)

        # ------------------------------------------------------------------
        # 4. Extended input state with permanent zero sentinel column
        # ------------------------------------------------------------------
        self._extended_state_in = cudm.DenseMixedState(
            ctx, self.hilbert_dims, n_ados + 1, "complex128",
        )
        self._extended_state_in.allocate_storage()
        self._extended_state_in.storage[:] = 0

        self._tmpl_state = cudm.DenseMixedState(
            ctx, self.hilbert_dims, n_ados, "complex128",
        )
        self._tmpl_state.allocate_storage()

        # ------------------------------------------------------------------
        # 5. Diagonal operator: L_sys + per-ADO scalar
        # ------------------------------------------------------------------
        diagonal = np.zeros(n_ados)
        for he_n in ados.labels:
            n = ados.idx(he_n)
            diagonal[n] = -sum(
                he_n[i] * ados.vk[i] for i in range(len(ados.vk))
            )

        self._diag_op = CuQobjEvo(Lsys).operator
        self._diag_op.append(cudm.tensor_product(
            coeff=diagonal, batch_size=n_ados, dtype="complex128",
        ))

        # ------------------------------------------------------------------
        # 6. Per-Q bucketing + per-bucket coupling operators
        # ------------------------------------------------------------------
        # Bucket exponents by content-hashed Q. Every exponent of a
        # given bath shares the same Q reference (and therefore the same
        # M = [Q, .] / P = {Q, .} superoperators), so for Drude-Lorentz
        # Matsubara the bucket count is simply the number of baths B,
        # independent of Nk.
        ks_by_q = {}
        for k in range(n_exponents):
            Q = ados.exponents[k].Q
            key = (
                tuple(tuple(d) for d in Q.dims),
                Q.full().tobytes(),
            )
            ks_by_q.setdefault(key, []).append(k)

        unique_keys = list(ks_by_q.keys())

        # ``_minus_ops`` and ``_plus_ops`` are positional, indexed by
        # bucket order in ``unique_keys`` / ``plus_buckets``. Only
        # buckets that contain at least one I or RI exponent get a P_Q
        # op (R-only buckets contribute nothing to the {Q, .} side).
        self._minus_ops = []
        self._plus_ops = []
        plus_buckets = []
        for key in unique_keys:
            ks = ks_by_q[key]
            rep_k = ks[0]
            Q_cu = ados.exponents[rep_k].Q.to(CuOperator)
            sQ = spre(Q_cu).data
            spQ = spost(Q_cu).data
            mt = _data.sub(sQ, spQ).to_OperatorTerm(
                dual=True, hilbert_dims=self.hilbert_dims,
            )
            self._minus_ops.append(
                cudm.Operator(self.hilbert_dims, (mt,))
            )
            has_iri = any(
                ados.exponents[k].type in (
                    BathExponent.types.I, BathExponent.types.RI,
                )
                for k in ks
            )
            if has_iri:
                pt = _data.add(sQ, spQ).to_OperatorTerm(
                    dual=True, hilbert_dims=self.hilbert_dims,
                )
                self._plus_ops.append(
                    cudm.Operator(self.hilbert_dims, (pt,))
                )
                plus_buckets.append(key)

        # ------------------------------------------------------------------
        # 7. Consolidated gather buffer + per-bucket state views
        # ------------------------------------------------------------------
        # One slot per bucket per side (minus for every unique Q;
        # plus only for unique Q's with at least one I/RI exponent).
        # Each slot's ``specs`` list multiplexes ALL per-exponent
        # contributions from that bucket onto the same gather state, so
        # the cudm matvec count equals the slot count.
        n_gather = len(unique_keys) + len(plus_buckets)
        chunk_n = D2 * n_ados
        self._gathers_buf = cp.zeros(
            n_gather * chunk_n, dtype="complex128", order="F",
        )

        def make_state_view(slot):
            """Return a DenseMixedState view of slot ``slot`` in ``_gathers_buf``."""
            sub = self._gathers_buf[
                slot * chunk_n : (slot + 1) * chunk_n
            ]
            return self._tmpl_state.clone(sub)

        self._gather_states = []
        self._gather_arrs = []
        self._gather_specs = []
        op_list = [self._diag_op]

        slot = 0

        # ----- 7a. minus slots: one per unique Q ----------------------
        # For every k in the bucket, accumulate the next-coupling
        # contribution (constant ``-1j`` weight, broadcast over n_ados)
        # plus, for R/RI types, the prev-coupling-minus contribution
        # (per-ADO ``prev_w_minus[k]`` weight). I-type k contribute only
        # to the next-coupling on the minus side (their prev-side
        # contribution lives on the plus slot).
        for key, op in zip(unique_keys, self._minus_ops):
            st = make_state_view(slot)
            slot += 1
            arr = st.storage[:chunk_n].reshape(D2, n_ados, order="F")
            specs = []
            for k in ks_by_q[key]:
                specs.append((self._next_map_gpu[k], self._ones_neg_j))
                t = ados.exponents[k].type
                if t in (BathExponent.types.R, BathExponent.types.RI):
                    specs.append(
                        (self._prev_map_gpu[k], self._prev_w_minus_gpu[k])
                    )
            self._gather_states.append(st)
            self._gather_arrs.append(arr)
            self._gather_specs.append(specs)
            op_list.append(op)

        # ----- 7b. plus slots: one per unique Q with any I/RI ---------
        # Only I/RI exponents in the bucket contribute prev-coupling on
        # the plus side. R-type k have ``prev_w_plus[k] = 0`` by
        # construction; we skip them here entirely (zero-weight specs
        # would be inert but waste a cupy kernel).
        for key, op in zip(plus_buckets, self._plus_ops):
            st = make_state_view(slot)
            slot += 1
            arr = st.storage[:chunk_n].reshape(D2, n_ados, order="F")
            specs = []
            for k in ks_by_q[key]:
                t = ados.exponents[k].type
                if t in (BathExponent.types.I, BathExponent.types.RI):
                    specs.append(
                        (self._prev_map_gpu[k], self._prev_w_plus_gpu[k])
                    )
            self._gather_states.append(st)
            self._gather_arrs.append(arr)
            self._gather_specs.append(specs)
            op_list.append(op)

        # ------------------------------------------------------------------
        # 8. Build and warm-prepare the OperatorAction
        # ------------------------------------------------------------------
        # Same wiring as Decomposed4: diagonal slot pairs with the
        # caller's ``input_state`` at compute time; every other slot
        # pairs with its precomputed gather state.
        self._operator_action = cudm.OperatorAction(ctx, op_list)
        self._operator_action.prepare(
            ctx, [self._tmpl_state] + self._gather_states,
        )

    def _compute(self, t, input_state, output_state):
        D2 = self._sup_shape
        n_ados = self._n_ados

        # --- (1) Stage input into extended buffer --------------------------
        self._extended_state_in.storage[: D2 * n_ados] = (
            input_state.storage[: D2 * n_ados]
        )
        ext_in = self._extended_state_in.storage[
            : D2 * (n_ados + 1)
        ].reshape(D2, n_ados + 1, order="F")

        # --- (2, 3) Zero all gather slots, then accumulate contributions ---
        self._gathers_buf[:] = 0
        for arr, specs in zip(self._gather_arrs, self._gather_specs):
            for idx, w in specs:
                arr[:] += w[None, :] * ext_in[:, idx]

        # --- (4) One fused cudm launch -------------------------------------
        self._operator_action.compute(
            t, None,
            [input_state] + self._gather_states,
            output_state,
        )

    # def matmul_data(self, t, state, out=None, scale=1.0):
    cpdef Data matmul_data(CuHEOMRhs self, object t, Data state, Data out=None, double complex scale=1.0):        

        if not isinstance(state, CuState):
            state = CuState(state)
        if out is None:
            out = zeros_like_cuState(state)
            
        if(scale != 1.0):
            state = state * scale

        input_cudm_state = self._tmpl_state.clone(state.base.storage)
        output_cudm_state = self._tmpl_state.clone(out.base.storage)

        self._compute(t, input_cudm_state, output_cudm_state)
        return out

    # def _register_feedback(self, solvers_feeds, solver):
    #     # CuHEOMRhs does not support feedback args, and the base
    #     # implementation iterates over the cdef ``_solver_only_feedback``
    #     # dict, which is left uninitialized when we bypass
    #     # ``QobjEvo.__init__`` via ``_restore``. Override to a no-op.
    #     return

    def arguments(self, args):
        raise NotImplementedError

    def linear_map(self, op_mapping, *, _skip_check=False):
        raise NotImplementedError

    def tidyup(self, atol=1e-12):
        raise NotImplementedError

    def to(self, data_type):
        raise NotImplementedError

    def dag(self):
        raise NotImplementedError

    def conj(self):
        raise NotImplementedError

    def trans(self):
        raise NotImplementedError

    @property
    def dtype(self):
        return cudm.OperatorAction

    @property
    def num_elements(self):
        raise NotImplementedError

    @property
    def isconstant(self):
        # v1 scope: constant H, constant bath couplings. ``CuHEOMSolver``
        # already raises NotImplementedError on time-dependent H, and the
        # bath gathers are built with frozen Cupy weight arrays.
        return True

    def __call__(self, t, _args=None, **kwargs):
        raise NotImplementedError

    def data(self, t):
        raise NotImplementedError

    def __repr__(self):
        return "CuHEOMRhs"