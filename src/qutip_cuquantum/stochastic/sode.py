import numpy as np

import qutip.core.data as _data
from qutip.core.data import imul, iadd
from qutip.solver.sode.sode import _Explicit_Simple_Integrator
from qutip.solver.sode._noise import Wiener, PreSetWiener
from qutip.solver.integrator.integrator import Integrator
from .smesolve import SMESolver
from .batching import batch_copy
from ..state import CuState

from .system import PyStochasticOpenSystem


class Explicit_Simple_Integrator_Batched(_Explicit_Simple_Integrator):

    integrator_options = {
        "dt": 0.001,
        "tol": 1e-10,
        "batch": 1,
    }
    def __init__(self, rhs, options):
        self._options = self.integrator_options.copy()
        self.options = options
        self.rhs = rhs

    def set_state(self, t, state0, generator):
        """
        Set the state of the SODE solver.

        Parameters
        ----------
        t : float
            Initial time

        state0 : qutip.Data
            Initial state.

        generator : numpy.random.generator
            Random number generator.
        """
        self.t = t
        self.batch = self.options["batch"]

        stepper_opt = {
            key: self.options[key]
            for key in self._stepper_options
            if key in self.options
        }

        if isinstance(generator, PreSetWiener):
            self.wiener = generator
            if (
                generator.is_measurement
                and "measurement_noise" not in self._stepper_options
            ):
                raise NotImplementedError(
                    f"{type(self).__name__} does not support running"
                    " the evolution from measurements."
                )
            stepper_opt["measurement_noise"] = generator.is_measurement
        elif isinstance(generator, Wiener):
            self.wiener = generator
        else:
            num_collapse = len(self.rhs.sc_ops)
            self.wiener = Wiener(
                t, self.options["dt"], generator,
                (self.N_dw, num_collapse, self.batch)
            )
        self.rhs._register_feedback(self.wiener)
        if self.rhs.issuper:
            rhs = self.rhs(self.options)
            #rhs = PyStochasticOpenSystem(
            #    self.rhs.H, self.rhs.sc_ops, self.rhs.c_ops,
            #    options.get("derr_dt", 1e-6)
            #)
        else:
            raise NotImplementedError("Only open system are implemented yet.")
        self.step_func = self.stepper(rhs, **stepper_opt).run

        if self.batch == 1:
            self.state = CuState(state0, rhs.L.hilbert_space_dims)
        else:
            state0 = CuState(state0, rhs.L.hilbert_space_dims)
            self.state = batch_copy(state0, self.batch)

        self._is_set = True


    def integrate(self, t, copy=True):
        delta_t = t - self.t
        dt = self.options["dt"]
        if delta_t < 0:
            raise ValueError("Integration time, can't be negative.")
        elif delta_t < 0.5 * dt:
            warnings.warn(
                f"Step under minimum step ({dt}), skipped.",
                RuntimeWarning
            )
            return self.t, self.state, np.zeros(self.N_dw)

        N, extra = np.divmod(delta_t, dt)
        N = int(N)
        if extra > 0.5 * dt:
            # Not a whole number of steps, round to higher
            N += 1
        dW = self.wiener.dW(self.t, N)

        self.state = self.step_func(self.t, self.state, dt, dW, N)
        self.t += dt * N

        return self.t, self.state, np.sum(dW[:, 0, :], axis=0)


class Euler:
    def __init__(self, system, measurement_noise=False):
        self.system = system
        self.measurement_noise = measurement_noise

    def run(self, t, state, dt, dW, num_step):
        for i in range(num_step):
            state = self.step(t + i * dt, state, dt, dW[i, :, :])
        return state

    def step(self, t, state, dt, dW):
        """Integration scheme:

        Basic Euler order 0.5
        dV = d1 dt + d2_i dW_i
        Numerical Solution of Stochastic Differential Equations
        By Peter E. Kloeden, Eckhard Platen
        """
        system = self.system

        a = system.drift(t, state)
        b = system.diffusion(t, state)

        if self.measurement_noise:
            expect = system.expect(t, state)
            for i in range(system.num_collapse):
                dW[0, i] -= expect[i].real * dt

        new_state = _data.add(state, a, dt)
        for i in range(system.num_collapse):
            new_state = _data.add(new_state, b[i], dW[0, i])
        return new_state


class EulerSODE(Explicit_Simple_Integrator_Batched):
    """
    A simple generalization of the Euler method for ordinary
    differential equations to stochastic differential equations.  Only
    solver which could take non-commuting ``sc_ops``.

    - Order: 0.5
    """
    integrator_options = {
        "dt": 0.001,
        "tol": 1e-10,
        "batch": 1,
    }
    stepper = Euler
    N_dw = 1
    _stepper_options = ["measurement_noise"]


class Platen(Euler):

    def step(self, t, state, dt, dW):
        """Platen rhs function for both master eq and schrodinger eq.

        dV = -iH* (V+Vt)/2 * dt + (d1(V)+d1(Vt))/2 * dt
             + (2*d2_i(V)+d2_i(V+)+d2_i(V-))/4 * dW_i
             + (d2_i(V+)-d2_i(V-))/4 * (dW_i**2 -dt) * dt**(-.5)

        Vt = V -iH*V*dt + d1*dt + d2_i*dW_i
        V+/- = V -iH*V*dt + d1*dt +/- d2_i*dt**.5
        The Theory of Open Quantum Systems
        Chapter 7 Eq. (7.47), H.-P Breuer, F. Petruccione
        """
        system = self.system
        num_ops = system.num_collapse
        sqrt_dt = np.sqrt(dt)
        sqrt_dt_inv = 0.25 / sqrt_dt

        d1 = _data.add(state, system.drift(t, state), dt)
        d2 = system.diffusion(t, state)

        if self.measurement_noise:
            expect = system.expect(t, state)
            for i in range(system.num_collapse):
                dW[0, i] -= expect[i].real * dt

        out = _data.mul(d1, 0.5)
        Vt = d1.copy()
        Vp = []
        Vm = []
        for i in range(num_ops):
            Vp.append(_data.add(d1, d2[i], sqrt_dt))
            Vm.append(_data.add(d1, d2[i], -sqrt_dt))
            Vt = _data.add(Vt, d2[i], dW[0, i])

        d1 = system.drift(t, Vt)
        out = _data.add(out, d1, 0.5 * dt)
        out = _data.add(out, state, 0.5)
        for i in range(num_ops):
            d2p = system.diffusion(t, Vp[i])
            d2m = system.diffusion(t, Vm[i])
            dw = dW[0, i] * 0.25
            out = _data.add(out, d2[i], 2 * dw)

            for j in range(num_ops):
                if i == j:
                    dw2 = sqrt_dt_inv * (dW[0, i] * dW[0, j] - dt)
                    dw2p = dw2 + dw
                    dw2m = -dw2 + dw
                else:
                    dw2p = sqrt_dt_inv * dW[0, i] * dW[0, j]
                    dw2m = -dw2p
                out = _data.add(out, d2p[j], dw2p)
                out = _data.add(out, d2m[j], dw2m)

        return out


class PlatenSODE(Explicit_Simple_Integrator_Batched):
    """
    Explicit scheme, creates the Milstein using finite differences
    instead of analytic derivatives. Also contains some higher order
    terms, thus converges better than Milstein while staying strong
    order 1.0.  Does not require derivatives. See eq. (7.47) of chapter 7 of
    H.-P. Breuer and F. Petruccione, *The Theory of Open Quantum Systems*.

    - Order: strong 1, weak 2
    """
    integrator_options = {
        "dt": 0.001,
        "tol": 1e-10,
        "batch": 1,
    }
    stepper = Platen
    N_dw = 1
    _stepper_options = ["measurement_noise"]


class Explicit15(Euler):

    def __init__(self, system):
        self.system = system

    def step(self, t, state, dt, dW):
        """Chapter 11.2 Eq.

        (2.13)
        Numerical Solution of Stochastic Differential Equations
        By Peter E. Kloeden, Eckhard Platen
        """
        system = self.system
        num_ops = system.num_collapse
        sqrt_dt = np.sqrt(dt)
        sqrt_dt_inv = 1.0 / sqrt_dt
        batch_size = 1
        if isinstance(state, CuState):
            batch_size = state.base.batch_size

        dw = np.empty((num_ops, batch_size))
        dz = np.empty((num_ops, batch_size))
        dwp = np.zeros((num_ops, batch_size))
        dwm = np.zeros((num_ops, batch_size))
        for i in range(num_ops):
            dw[i] = dW[0, i]
            dz[i] = 0.5 * (dW[0, i] + 1.0 / np.sqrt(3) * dW[1, i])

        d1 = system.drift(t, state)
        d2 = system.diffusion(t, state)
        dd2 = system.diffusion(t + dt, state)
        # Euler part
        out = _data.add(state, d1, dt)
        for i in range(num_ops):
            out = _data.add(out, d2[i], dw[i])

        V = _data.add(state, d1, dt / num_ops)

        v2p = []
        v2m = []
        for i in range(num_ops):
            v2p.append(_data.add(V, d2[i], sqrt_dt))
            v2m.append(_data.add(V, d2[i], -sqrt_dt))

        p2p = []
        p2m = []
        for i in range(num_ops):
            d2p = system.diffusion(t, v2p[i])
            d2m = system.diffusion(t, v2m[i])
            ddw = (dw[i] * dw[i] - dt) * 0.25 * sqrt_dt_inv  # 1.0
            out = _data.add(out, d2p[i], ddw)
            out = _data.add(out, d2m[i], -ddw)
            temp_p2p = []
            temp_p2m = []
            for j in range(num_ops):
                temp_p2p.append(_data.add(v2p[i], d2p[j], sqrt_dt))
                temp_p2m.append(_data.add(v2p[i], d2p[j], -sqrt_dt))
            p2p.append(temp_p2p)
            p2m.append(temp_p2m)

        out = _data.add(out, d1, -0.5 * (num_ops) * dt)

        for i in range(num_ops):
            ddz = dz[i] * 0.5 / sqrt_dt  # 1.5
            ddd = 0.25 * (dw[i] * dw[i] / 3 - dt) * dw[i] / dt  # 1.5
            for j in range(num_ops):
                dwp[j] = 0
                dwm[j] = 0

            d1p = system.drift(t + dt / num_ops, v2p[i])
            d1m = system.drift(t + dt / num_ops, v2m[i])

            d2p = system.diffusion(t, v2p[i])
            d2m = system.diffusion(t, v2m[i])
            d2pp = system.diffusion(t, p2p[i][i])
            d2mm = system.diffusion(t, p2m[i][i])

            out = _data.add(out, d1p, (0.25 + ddz) * dt)
            out = _data.add(out, d1m, (0.25 - ddz) * dt)

            out = _data.add(out, dd2[i], dw[i] - dz[i])
            out = _data.add(out, d2[i], dz[i] - dw[i])

            out = _data.add(out, d2pp[i], ddd)
            out = _data.add(out, d2mm[i], -ddd)
            dwp[i] += -ddd
            dwm[i] += ddd

            for j in range(num_ops):
                ddw = 0.5 * (dw[j] - dz[j])  # O(1.5)
                dwp[j] += ddw
                dwm[j] += ddw
                out = _data.add(out, d2[j], -2 * ddw)

                if j > i:
                    ddw = 0.5 * (dw[i] * dw[j]) / sqrt_dt  # O(1.0)
                    dwp[j] += ddw
                    dwm[j] += -ddw

                    ddw = (
                        0.25 * (dw[j] * dw[j] - dt) * dw[i] / dt
                    )  # O(1.5)
                    d2pp = system.diffusion(t, p2p[j][i])
                    d2mm = system.diffusion(t, p2m[j][i])
                    out = _data.add(out, d2pp[j], ddw)
                    out = _data.add(out, d2mm[j], -ddw)
                    dwp[j] += -ddw
                    dwm[j] += ddw

                    for k in range(j + 1, num_ops):
                        ddw = (
                            0.5 * dw[i] * dw[j] * dw[k] / dt
                        )  # O(1.5)
                        out = _data.add(out, d2pp[k], ddw)
                        out = _data.add(out, d2mm[k], -ddw)
                        dwp[k] += -ddw
                        dwm[k] += ddw

                if j < i:
                    ddw = (
                        0.25 * (dw[j] * dw[j] - dt) * dw[i] / dt
                    )  # O(1.5)
                    d2pp = system.diffusion(t, p2p[j][i])
                    d2mm = system.diffusion(t, p2m[j][i])

                    out = _data.add(out, d2pp[j], ddw)
                    out = _data.add(out, d2mm[j], -ddw)
                    dwp[j] += -ddw
                    dwm[j] += ddw

            for j in range(num_ops):
                out = _data.add(out, d2p[j], dwp[j])
                out = _data.add(out, d2m[j], dwm[j])

        return out


class Explicit1_5_SODE(Explicit_Simple_Integrator_Batched):
    """
    Explicit order 1.5 strong schemes.  Reproduce the order 1.5 strong
    Taylor scheme using finite difference instead of derivatives.
    Slower than ``taylor15`` but usable when derrivatives cannot be
    analytically obtained.
    See eq. (2.13) of chapter 11.2 of Peter E. Kloeden and Exkhard Platen,
    *Numerical Solution of Stochastic Differential Equations.*

    - Order: strong 1.5
    """
    stepper = Explicit15
    N_dw = 2


class Milstein:
    def __init__(self, system, measurement_noise=False):
        self.system = system
        self.measurement_noise = measurement_noise

    def run(self, t, state, dt, dW, ntraj):
        out = _data.zeros_like(state)
        state = state.copy()

        for i in range(ntraj):
            out = self.step(t + i * dt, state, dt, dW[i, :, :], out)
            state, out = out, state
        return state

    def step(self, t, state, dt, dW, out):
        """Chapter 10.3 Eq.

        (3.12)
        Numerical Solution of Stochastic Differential Equations
        By Peter E. Kloeden, Eckhard Platen

        dV = -iH*V*dt + d1*dt + d2_i*dW_i
        + 0.5*d2_i' d2_j*(dW_i*dw_j -dt*delta_ij)
        """
        system = self.system
        num_ops = system.num_collapse

        system.set_state(t, state)

        out = imul(out, 0.0)
        out = iadd(out, state, 1.0)
        out = iadd(out, system.a(), dt)

        if self.measurement_noise:
            expect = system.expect(t, state)
            for i in range(system.num_collapse):
                dW[0, i] -= system.expect_i(i).real * dt

        for i in range(num_ops):
            out = iadd(out, system.bi(i), dW[0, i])

        for i in range(num_ops):
            for j in range(i, num_ops):
                if i == j:
                    dw = (dW[0, i] * dW[0, j] - dt) * 0.5
                else:
                    dw = dW[0, i] * dW[0, j]
                out = iadd(out, system.Libj(i, j), dw)
        return out


class Milstein_SODE(Explicit_Simple_Integrator_Batched):
    """
    An order 1.0 strong Taylor scheme.  Better approximate numerical
    solution to stochastic differential equations.  See eq. (3.12) of
    chapter 10.3 of Peter E. Kloeden and Exkhard Platen,
    *Numerical Solution of Stochastic Differential Equations*..

    - Order strong 1.0
    """
    integrator_options = {
        "dt": 0.001,
        "tol": 1e-10,
        "batch": 1,
    }
    stepper = Milstein
    N_dw = 1
    _stepper_options = ["measurement_noise"]


class PredCorr:
    def __init__(self, system, alpha=0.0, eta=0.5, measurement_noise=False):
        self.system = system
        self.alpha = alpha
        self.eta = eta
        self.measurement_noise = measurement_noise

    def run(self, t, state, dt, dW, ntraj):
        out = _data.zeros_like(state)
        self.euler = _data.zeros_like(state)
        state = state.copy()

        for i in range(ntraj):
            out = self.step(t + i * dt, state, dt, dW[i, :, :], out)
            state, out = out, state
        return state

    def step(self, t, state, dt, dW, out):
        """Chapter 15.5 Eq.

        (5.4)
        Numerical Solution of Stochastic Differential Equations
        By Peter E. Kloeden, Eckhard Platen
        """
        system = self.system
        num_ops = system.num_collapse
        eta = self.eta
        alpha = self.alpha
        euler = self.euler

        system.set_state(t, state)

        if self.measurement_noise:
            expect = system.expect(t, state)
            for i in range(system.num_collapse):
                dW[0, i] -= system.expect_i(i).real * dt

        out = imul(out, 0.0)
        out = iadd(out, state, 1)
        out = iadd(out, system.a(), dt * (1 - alpha))

        euler = imul(euler, 0.0)
        euler = iadd(euler, state, 1)
        euler = iadd(euler, system.a(), dt)

        for i in range(num_ops):
            euler = iadd(euler, system.bi(i), dW[0, i])
            out = iadd(out, system.bi(i), dW[0, i] * eta)
            out = iadd(out, system.Libj(i, i), dt * (alpha - 1) * 0.5)

        system.set_state(t + dt, euler)
        for i in range(num_ops):
            out = iadd(out, system.bi(i), dW[0, i] * (1 - eta))

        if alpha:
            out = iadd(out, system.a(), dt * alpha)
            for i in range(num_ops):
                out = iadd(out, system.Libj(i, i), -dt * alpha * 0.5)

        return out


class PredCorr_SODE(Explicit_Simple_Integrator_Batched):
    """
    Generalization of the trapezoidal method to stochastic differential
    equations. More stable than explicit methods.  See eq. (5.4) of
    chapter 15.5 of Peter E. Kloeden and Exkhard Platen,
    *Numerical Solution of Stochastic Differential Equations*.

    - Order strong 0.5, weak 1.0
    - Codes to only correct the stochastic part (:math:`\\alpha=0`,
      :math:`\\eta=1/2`): ``'pred-corr'``, ``'predictor-corrector'`` or
      ``'pc-euler'``
    - Codes to correct both the stochastic and deterministic parts
      (:math:`\\alpha=1/2`, :math:`\\eta=1/2`): ``'pc-euler-imp'``,
      ``'pc-euler-2'`` or ``'pred-corr-2'``
    """

    integrator_options = {
        "dt": 0.001,
        "tol": 1e-10,
        "alpha": 0.0,
        "eta": 0.5,
        "batch": 1,
    }
    stepper = PredCorr
    N_dw = 1
    _stepper_options = ["alpha", "eta", "measurement_noise"]

    @property
    def options(self):
        """
        Supported options by Explicit Stochastic Integrators:

        dt : float, default: 0.001
            Internal time step.

        tol : float, default: 1e-10
            Tolerance for the time steps.

        alpha : float, default: 0.
            Implicit factor to the drift.
            eff_drift ~= drift(t) * (1-alpha) + drift(t+dt) * alpha

        eta : float, default: 0.5
            Implicit factor to the diffusion.
            eff_diffusion ~= diffusion(t) * (1-eta) + diffusion(t+dt) * eta
        """
        return self._options

    @options.setter
    def options(self, new_options):
        Integrator.options.fset(self, new_options)


class RouchonSODE(SIntegrator):
    """
    Stochastic integration method keeping the positivity of the density matrix.
    See eq. (4) Pierre Rouchon and Jason F. Ralpha,
    *Efficient Quantum Filtering for Quantum Feedback Control*,
    `arXiv:1410.5345 [quant-ph] <https://arxiv.org/abs/1410.5345>`_,
    Phys. Rev. A 91, 012118, (2015).

    - Order: strong 1

    Notes
    -----
    This method should be used with very small ``dt``. Unlike other
    methods that will return unphysical state (negative eigenvalues, Nans)
    when the time step is too large, this method will return state that
    seems normal.
    """
    integrator_options = {
        "dt": 0.0001,
        "tol": 1e-7,
        "batch": 1,
    }

    def __init__(self, rhs, options):
        self._options = self.integrator_options.copy()
        self.options = options
        self.rhs = rhs
        self._make_operators()

    def _make_operators(self):
        rhs = self.rhs
        self.H = rhs.H
        if self.H.issuper:
            raise TypeError("The rouchon stochastic integration method can't"
                            " use a premade Liouvillian.")
        self._issuper = rhs.issuper

        dtype = type(self.H(0).data)
        self.c_ops = rhs.c_ops
        self.sc_ops = rhs.sc_ops
        self.cpcds = [op + op.dag() for op in self.sc_ops]
        for op in self.cpcds:
            op.compress()
        self.M = (
            - 1j * self.H
            - sum(op.dag() @ op for op in self.c_ops) * 0.5
            - sum(op.dag() @ op for op in self.sc_ops) * 0.5
        )
        self.M.compress()

        self.num_collapses = len(self.sc_ops)
        self.scc = [
            [self.sc_ops[i] @ self.sc_ops[j] for i in range(j+1)]
            for j in range(self.num_collapses)
        ]

        self.id = _data.identity[dtype](self.H.shape[0])

    def set_state(self, t, state0, generator):
        """
        Set the state of the SODE solver.

        Parameters
        ----------
        t : float
            Initial time

        state0 : qutip.Data
            Initial state.

        generator : numpy.random.generator
            Random number generator.
        """
        self.t = t
        self.state = state0
        if isinstance(generator, Wiener):
            self.wiener = generator
        else:
            self.wiener = Wiener(
                t, self.options["dt"], generator,
                (1, self.num_collapses,)
            )
        self.rhs._register_feedback(self.wiener)
        self._make_operators()
        self._is_set = True

    def integrate(self, t, copy=True):
        delta_t = (t - self.t)
        dt = self.options["dt"]
        if delta_t < 0:
            raise ValueError("Stochastic integration need increasing times")
        elif delta_t < 0.5 * dt:
            warnings.warn(
                f"Step under minimum step ({dt}), skipped.",
                RuntimeWarning
            )
            return self.t, self.state, np.zeros(len(self.sc_ops))

        N, extra = np.divmod(delta_t, dt)
        N = int(N)
        if extra > 0.5 * dt:
            # Not a whole number of steps, round to higher
            N += 1
        dW = self.wiener.dW(self.t, N)[:, 0, :]

        # if self._issuper:
        #     self.state = unstack_columns(self.state)
        for dw in dW:
            self.state = self._step(self.t, self.state, dt, dw)
            self.t += dt
        # if self._issuper:
        #     self.state = stack_columns(self.state)

        return self.t, self.state, np.sum(dW, axis=0)

    def _step(self, t, state, dt, dW):
        dy = [
            op.expect_data(t, state) * dt + dw
            for op, dw in zip(self.cpcds, dW)
        ]
        M = _data.add(self.id, self.M._call(t), dt)
        for i in range(self.num_collapses):
            M = _data.add(M, self.sc_ops[i]._call(t), dy[i])
            M = _data.add(M, self.scc[i][i]._call(t), (dy[i]**2-dt)/2)
            for j in range(i):
                M = _data.add(M, self.scc[i][j]._call(t), dy[i]*dy[j])
        out = _data.matmul(M, state)

        # M = (1 + -iH + c_i*c_i^t * dt + c_i dy_i +  c_i c_j dyy_ij)
        # M @ rho @ M

        if self._issuper:
            Mdag = M.adjoint()
            out = _data.matmul(out, Mdag)
            for cop in self.c_ops:
                op = cop._call(t)
                out += op @ state @ op.adjoint() * dt
            out = out / _data.trace(out)
        else:
            out = out / _data.norm.l2(out)
        return out

    @property
    def options(self):
        """
        Supported options by Rouchon Stochastic Integrators:

        dt : float, default: 0.001
            Internal time step.

        tol : float, default: 1e-7
            Relative tolerance.
        """
        return self._options

    @options.setter
    def options(self, new_options):
        Integrator.options.fset(self, new_options)

    def reset(self, hard=False):
        if self._is_set:
            state = self.get_state()
        if hard:
            raise NotImplementedError(
                "Changing stochastic integrator "
                "options is not supported."
            )
        if self._is_set:
            self.set_state(*state)


# SMESolver.add_integrator(RouchonSODE, "rouchon")
SMESolver.add_integrator(EulerSODE, "euler")
SMESolver.add_integrator(PlatenSODE, "platen")
SMESolver.add_integrator(Explicit1_5_SODE, "explicit1.5")
SMESolver.add_integrator(Milstein_SODE, "milstein")
SMESolver.add_integrator(PredCorr_SODE, "pred_corr")
