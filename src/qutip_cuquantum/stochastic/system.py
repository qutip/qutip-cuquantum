import qutip.core.data as _data
from qutip import liouvillian, spre, spost
from ..state import trace_oper_ket_cuState

add = _data.add
imul = _data.imul
zeros_like = _data.zeros_like
iadd = _data.iadd


__all__ = ["PyStochasticOpenSystem"]


class PyStochasticOpenSystem:
    """
        RHS for open quantum stochastic system (smesolve)

        drift = liouvillian(H, sc_ops + c_ops)(rho)

        diffusion = c_i @ rho + rho @ c_i/dag - tr(c_i @ rho) rho
    """
    def __init__(self, H, sc_ops, c_ops=(), derr_dt=1e-6):
        if H.issuper:
            self.L = H + liouvillian(None, sc_ops)
        else:
            self.L = liouvillian(H, sc_ops)
        if c_ops:
            self.L = self.L + liouvillian(None, c_ops)

        self.c_ops = [spre(op) + spost(op.dag()) for op in sc_ops]
        self.num_collapse = len(self.c_ops)
        self.state_size = self.L.shape[1]
        self._is_set = 0
        self.N_root = int(self.state_size**0.5)
        self.dt = derr_dt

    def drift(self, t, state, out=None):
        return self.L.matmul_data(t, state, out=out)

    def diffusion(self, t, state, out=None):
        if out is None:
            out = [None] * self.num_collapse

        for i in range(self.num_collapse):
            c_op = self.c_ops[i]
            vec = c_op.matmul_data(t, state, out[i])
            expect = _data.trace_oper_ket(vec)
            out[i] = _data.iadd(vec, state, -expect)
        return out

    def expect(self, t, state):
        expect = []
        for i in range(self.num_collapse):
            c_op = self.c_ops[i]
            expect.append(c_op.expect_data(t, state))
        return expect

    def set_state(self, t, state):
        self.t = t
        self.state = state
        self._a_set = False
        self._b_set = False
        self._Lb_set = False
        self._L0b_set = False
        self._La_set = False
        self._LLb_set = False
        self._L0a_set = False

        if not self._is_set:
            n = self.num_collapse
            l = self.state_size
            self._is_set = 1
            self._a = zeros_like(state)
            self._b = [zeros_like(state) for _ in range(n)]
            self.expect_Cv = [0] * n
            self.expect_Cb = [[0] * n for _ in range(n)]
            self._Lb = [
                [zeros_like(state) for _ in range(n)]
                for _ in range(n)
            ]

    def a(self):
        if not self._is_set:
            raise RuntimeError(
                "Derrivatives set for ito taylor expansion need "
                "to receive the state with `set_state`."
            )
        if not self._a_set:
            self._compute_a()
        return self._a

    def _compute_a(self):
        if not self._is_set:
            raise RuntimeError(
                "Derrivatives set for ito taylor expansion need "
                "to receive the state with `set_state`."
            )
        imul(self._a, 0)
        self.L.matmul_data(self.t, self.state, self._a)
        self._a_set = True

    def bi(self, i):
        if not self._is_set:
            raise RuntimeError(
                "Derrivatives set for ito taylor expansion need "
                "to receive the state with `set_state`."
            )
        if not self._b_set:
            self._compute_b()
        return self._b[i]

    def expect_i(self, i):
        if not self._is_set:
            raise RuntimeError(
                "Derrivatives set for ito taylor expansion need "
                "to receive the state with `set_state`."
            )
        if not self._b_set:
            self._compute_b()
        return self.expect_Cv[i]

    def _compute_b(self):
        if not self._is_set:
            raise RuntimeError(
                "Derrivatives set for ito taylor expansion need "
                "to receive the state with `set_state`."
            )
        state=self.state
        for i in range(self.num_collapse):
            c_op = self.c_ops[i]
            b_vec = self._b[i]
            imul(b_vec, 0)
            c_op.matmul_data(self.t, state, b_vec)
            self.expect_Cv[i] = trace_oper_ket_cuState(b_vec)
            iadd(b_vec, state, -self.expect_Cv[i])
        self._b_set = True

    def Libj(self, i, j):
        if not self._is_set:
            raise RuntimeError(
                "Derrivatives set for ito taylor expansion need "
                "to receive the state with `set_state`."
            )
        if not self._Lb_set:
            self._compute_Lb()
        # We only support commutative diffusion
        if i > j:
            j, i = i, j
        return self._Lb[i][j]

    def _compute_Lb(self):
        state = self.state
        if not self._b_set:
            self._compute_b()

        for i in range(self.num_collapse):
            c_op = self.c_ops[i]
            for j in range(i, self.num_collapse):
                b_vec = self._b[j]
                Lb_vec = self._Lb[i][j]
                imul(Lb_vec, 0)
                c_op.matmul_data(self.t, b_vec, Lb_vec)
                self.expect_Cb[i][j] = trace_oper_ket_cuState(Lb_vec)
                iadd(Lb_vec, b_vec, -self.expect_Cv[i])
                iadd(Lb_vec, state, -self.expect_Cb[i][j])
        self._Lb_set = True

    def Lia(self, i):
        raise NotImplementedError

    def L0bi(self, i):
        raise NotImplementedError

    def LiLjbk(self, i, j, k):
        raise NotImplementedError

    def L0a(self):
        raise NotImplementedError
