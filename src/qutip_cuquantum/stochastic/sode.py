import numpy as np

import qutip.core.data as _data
from qutip.core.data import imul, iadd
from qutip.solver.sode.sode import _Explicit_Simple_Integrator
from qutip.solver.sode._noise import Wiener, PreSetWiener
from qutip.solver.integrator.integrator import Integrator
from .smesolve import SMESolver
from .batching import batch_copy, BatchCoefficient
from ..state import CuState
from ..qobjevo import CuQobjEvo
import warnings
from .system import PyStochasticOpenSystem
from qutip.core.cy.coefficient import FunctionCoefficient
import qutip


class BatchWiener:
    """
    Wiener process.
    """
    def __init__(self, t0, dt, generators, shape):
        self.t0 = t0
        self.dt = dt
        self.shape = shape
        self.generators = generators
        self.noise = np.zeros((0,) + shape + (len(generators),), dtype=float)
        self.last_W = np.zeros(shape[-1], dtype=float)
        self.idx_last_0 = 0

    def _extend(self, idx):
        N_new_vals = idx - self.noise.shape[0]
        shape = (N_new_vals,) + self.shape
        dW = np.stack([
            generator.normal(0, np.sqrt(self.dt), size=shape)
            for generator in self.generators
        ], axis=-1)
        self.noise = np.concatenate((self.noise, dW), axis=0)

    def dW(self, t, N):
        # Find the index of t.
        # Rounded to the closest step, but only multiple of dt are expected.
        idx0 = round((t - self.t0) / self.dt)
        if idx0 + N - 1 >= self.noise.shape[0]:
            self._extend(idx0 + N)
        return self.noise[idx0:idx0 + N, :, :, :]

    def __call__(self, t):
        """
        Return the Wiener process at the closest ``dt`` step to ``t``.
        """
        # The Wiener process is not used directly in the evolution, so it's
        # less optimized than the ``dW`` method.

        # Find the index of t.
        # Rounded to the closest step, but only multiple of dt are expected.
        idx = round((t - self.t0) / self.dt)
        if idx >= self.noise.shape[0]:
            self._extend(idx + 1)

        if self.idx_last_0 > idx:
            # Before last call, reseting
            self.idx_last_0 = 0
            self.last_W = np.zeros(self.shape[-1], dtype=float)

        self.last_W = self.last_W + np.sum(
            self.noise[self.idx_last_0:idx+1, 0, :], axis=0
        )

        self.idx_last_0 = idx
        return self.last_W



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
        self.batch = len(generator)

        stepper_opt = {
            key: self.options[key]
            for key in self._stepper_options
            if key in self.options
        }

        if isinstance(generator, PreSetWiener):
            # Not reachable, run_from_experiment not implemented
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
            # Not reachable
            self.wiener = generator
        else:
            num_collapse = len(self.rhs.sc_ops)
            self.wiener = BatchWiener(
                t, self.options["dt"], generator,
                (self.N_dw, num_collapse)
            )
        self.rhs._register_feedback(self.wiener)
        if self.rhs.issuper:
            rhs = self.rhs(self.options)
        else:
            raise NotImplementedError("Only open system are implemented yet.")
        stepper = self.stepper(rhs, **stepper_opt)
        self.step_func = stepper.run

        if self.batch == 1:
            self.state = CuState(state0, rhs.L.hilbert_space_dims)
        else:
            state0 = CuState(state0, rhs.L.hilbert_space_dims)
            self.state = batch_copy(state0, self.batch)

        if hasattr(stepper, "_prepare"):
            stepper._prepare(self.state)

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

        return self.t, self.state, np.sum(dW[:, 0, :, :], axis=0)


class Euler:
    def __init__(self, system, measurement_noise=False):
        self.system = system
        self.measurement_noise = measurement_noise

    def _prepare(self, state):
        self.out = _data.zeros_like(state)
        self.a = _data.zeros_like(state)
        self.b = [
            _data.zeros_like(state)
            for _ in range(self.system.num_collapse)
        ]

    def run(self, t, state, dt, dW, num_step):
        out = self.out
        for i in range(num_step):
            new_state = self.step(t + i * dt, state, dt, dW[i, :, :, :], out)
            state, out = new_state, state
            self.out = out
        return state

    def step(self, t, state, dt, dW, out):
        """Integration scheme:

        Basic Euler order 0.5
        dV = d1 dt + d2_i dW_i
        Numerical Solution of Stochastic Differential Equations
        By Peter E. Kloeden, Eckhard Platen
        """
        system = self.system

        self.a = imul(self.a, 0)
        a = system.drift(t, state, self.a)
        for i in range(system.num_collapse):
            self.b = [imul(b_, 0) for b_ in self.b]
        b = system.diffusion(t, state, self.b)

        if self.measurement_noise:
            expect = system.expect(t, state)
            for i in range(system.num_collapse):
                dW[0, i] -= expect[i].real * dt

        out.base.view()[:] = state.base.view()

        out = _data.iadd(out, a, dt)
        for i in range(system.num_collapse):
            out = _data.iadd(out, b[i], dW[0, i, :])
        return out


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

    def _prepare(self, state):
        self.out = _data.zeros_like(state)
        self.d1 = _data.zeros_like(state)
        self.tmp = _data.zeros_like(state)
        self.Vt = self.tmp
        self.d2 = [
            _data.zeros_like(state)
            for _ in range(self.system.num_collapse)
        ]
        self.Vp = [
            _data.zeros_like(state)
            for _ in range(self.system.num_collapse)
        ]
        self.Vm = [
            _data.zeros_like(state)
            for _ in range(self.system.num_collapse)
        ]

    def step(self, t, state, dt, dW, out):
        """Platen rhs function for both master eq and schrodinger eq.

        dV = (d1(V)+d1(Vt))/2 * dt
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

        out.base.view()[:] = state.base.view()
        self.d1.base.view()[:] = state.base.view()
        self.tmp.base.view()[:] = 0.

        self.tmp = system.drift(t, state, self.tmp)
        self.d1 = _data.iadd(self.d1, self.tmp, dt)

        self.d2 = [imul(b_, 0) for b_ in self.d2]
        d2 = system.diffusion(t, state, self.d2)

        if self.measurement_noise:
            expect = system.expect(t, state)
            for i in range(system.num_collapse):
                dW[0, i] -= expect[i].real * dt

        out.base.view()[:] = self.d1.base.view()
        out = _data.imul(out, 0.5)
        out = _data.iadd(out, state, 0.5)
        self.Vt.base.view()[:] = self.d1.base.view()

        for i in range(num_ops):
            self.Vp[i].base.view()[:] = self.d1.base.view()
            self.Vm[i].base.view()[:] = self.d1.base.view()
            self.Vp[i] = _data.iadd(self.Vp[i], d2[i], sqrt_dt)
            self.Vm[i] = _data.iadd(self.Vm[i], d2[i], -sqrt_dt)
            self.Vt = _data.iadd(self.Vt, d2[i], dW[0, i])

        self.d1.base.view()[:] = 0.
        self.d1 = system.drift(t, self.Vt, self.d1)
        out = _data.iadd(out, self.d1, 0.5 * dt)

        for i in range(num_ops):
            out = _data.iadd(out, d2[i], 0.5 * dW[0, i])

        for i in range(num_ops):
            dw = dW[0, i] * 0.25

            self.d2 = [imul(b_, 0) for b_ in self.d2]
            d2p = system.diffusion(t, self.Vp[i], self.d2)

            for j in range(num_ops):
                if i == j:
                    dw2 = sqrt_dt_inv * (dW[0, i] * dW[0, j] - dt)
                    dw2p = dw2 + dw
                else:
                    dw2p = sqrt_dt_inv * dW[0, i] * dW[0, j]
                out = _data.iadd(out, d2p[j], dw2p)

            self.d2 = [imul(b_, 0) for b_ in self.d2]
            d2m = system.diffusion(t, self.Vm[i], self.d2)

            for j in range(num_ops):
                if i == j:
                    dw2 = sqrt_dt_inv * (dW[0, i] * dW[0, j] - dt)
                    dw2m = -dw2 + dw
                else:
                    dw2m = -sqrt_dt_inv * dW[0, i] * dW[0, j]
                out = _data.iadd(out, d2m[j], dw2m)

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

    def _prepare(self, state):
        self.out = _data.zeros_like(state)
        self.d1 = _data.zeros_like(state)
        self.d1_alt = _data.zeros_like(state)
        self.tmp = _data.zeros_like(state)
        self.V = _data.zeros_like(state)
        self.d2 = [
            _data.zeros_like(state)
            for _ in range(self.system.num_collapse)
        ]
        self.d2_buf = [
            _data.zeros_like(state)
            for _ in range(self.system.num_collapse)
        ]
        self.dd2 = [
            _data.zeros_like(state)
            for _ in range(self.system.num_collapse)
        ]
        self.v2p = [
            _data.zeros_like(state)
            for _ in range(self.system.num_collapse)
        ]
        self.v2m = [
            _data.zeros_like(state)
            for _ in range(self.system.num_collapse)
        ]
        self.p2p = [
            [
                _data.zeros_like(state)
                for _ in range(self.system.num_collapse)
            ]
            for _ in range(self.system.num_collapse)
        ]
        self.p2m = [
            [
                _data.zeros_like(state)
                for _ in range(self.system.num_collapse)
            ]
            for _ in range(self.system.num_collapse)
        ]

    def step(self, t, state, dt, dW, out):
        """Chapter 11.2 Eq. (2.13)
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

        self.d1.base.view()[:] = 0.
        d1 = system.drift(t, state, self.d1)

        out.base.view()[:] = state.base.view()
        out = _data.iadd(out, d1, dt)

        for b_ in self.d2:
            imul(b_, 0)
        d2 = system.diffusion(t, state, self.d2)

        for b_ in self.dd2:
            imul(b_, 0)
        dd2 = system.diffusion(t + dt, state, self.dd2)

        for i in range(num_ops):
            out = _data.iadd(out, d2[i], dw[i])

        self.V.base.view()[:] = state.base.view()
        V = _data.iadd(self.V, d1, dt / num_ops)

        for i in range(num_ops):
            self.v2p[i].base.view()[:] = V.base.view()
            self.v2m[i].base.view()[:] = V.base.view()
            self.v2p[i] = _data.iadd(self.v2p[i], d2[i], sqrt_dt)
            self.v2m[i] = _data.iadd(self.v2m[i], d2[i], -sqrt_dt)

        for i in range(num_ops):
            ddw_base = (dw[i] * dw[i] - dt) * 0.25 * sqrt_dt_inv
            for b_ in self.d2_buf:
                imul(b_, 0)
            d2p = system.diffusion(t, self.v2p[i], self.d2_buf)
            out = _data.iadd(out, d2p[i], ddw_base)

            for j in range(num_ops):
                self.p2p[i][j].base.view()[:] = self.v2p[i].base.view()
                self.p2m[i][j].base.view()[:] = self.v2p[i].base.view()
                self.p2p[i][j] = _data.iadd(self.p2p[i][j], d2p[j], sqrt_dt)
                self.p2m[i][j] = _data.iadd(self.p2m[i][j], d2p[j], -sqrt_dt)

            for b_ in self.d2_buf:
                imul(b_, 0)

            d2m = system.diffusion(t, self.v2m[i], self.d2_buf)
            out = _data.iadd(out, d2m[i], -ddw_base)

        del d2p, d2m

        out = _data.iadd(out, self.d1, -0.5 * num_ops * dt)

        for i in range(num_ops):
            ddz = dz[i] * 0.5 / sqrt_dt
            ddd = 0.25 * (dw[i] * dw[i] / 3 - dt) * dw[i] / dt

            for j in range(num_ops):
                dwp[j] = 0
                dwm[j] = 0

            self.d1_alt.base.view()[:] = 0.
            d1p = system.drift(t + dt / num_ops, self.v2p[i], self.d1_alt)
            out = _data.iadd(out, d1p, (0.25 + ddz) * dt)
            del d1p

            self.d1_alt.base.view()[:] = 0.
            d1m = system.drift(t + dt / num_ops, self.v2m[i], self.d1_alt)
            out = _data.iadd(out, d1m, (0.25 - ddz) * dt)
            del d1m

            out = _data.iadd(out, dd2[i], dw[i] - dz[i])
            out = _data.iadd(out, d2[i], dz[i] - dw[i])

            for b_ in self.d2_buf:
                imul(b_, 0)
            d2pp = system.diffusion(t, self.p2p[i][i], self.d2_buf)
            out = _data.iadd(out, d2pp[i], ddd)
            del d2pp

            for b_ in self.d2_buf:
                imul(b_, 0)
            d2mm = system.diffusion(t, self.p2m[i][i], self.d2_buf)
            out = _data.iadd(out, d2mm[i], -ddd)
            del d2mm

            dwp[i] += -ddd
            dwm[i] += ddd

            for j in range(num_ops):
                ddw_j = 0.5 * (dw[j] - dz[j])
                dwp[j] += ddw_j
                dwm[j] += ddw_j
                out = _data.iadd(out, d2[j], -2 * ddw_j)

                if j > i:
                    ddw_cross = 0.5 * (dw[i] * dw[j]) / sqrt_dt
                    dwp[j] += ddw_cross
                    dwm[j] += -ddw_cross

                    ddw_order15 = 0.25 * (dw[j] * dw[j] - dt) * dw[i] / dt

                    for b_ in self.d2_buf:
                        imul(b_, 0)
                    d2pp = system.diffusion(t, self.p2p[j][i], self.d2_buf)
                    out = _data.iadd(out, d2pp[j], ddw_order15)

                    for k in range(j + 1, num_ops):
                        ddw_k = 0.5 * dw[i] * dw[j] * dw[k] / dt
                        out = _data.iadd(out, d2pp[k], ddw_k)
                    del d2pp

                    for b_ in self.d2_buf:
                        imul(b_, 0)
                    d2mm = system.diffusion(t, self.p2m[j][i], self.d2_buf)
                    out = _data.iadd(out, d2mm[j], -ddw_order15)

                    for k in range(j + 1, num_ops):
                        ddw_k = 0.5 * dw[i] * dw[j] * dw[k] / dt
                        out = _data.iadd(out, d2mm[k], -ddw_k)
                    del d2mm

                    # Now safely apply updates to noise array coefficients
                    dwp[j] += -ddw_order15
                    dwm[j] += ddw_order15

                    for k in range(j + 1, num_ops):
                        ddw_k = 0.5 * dw[i] * dw[j] * dw[k] / dt
                        dwp[k] += -ddw_k
                        dwm[k] += ddw_k

                if j < i:
                    ddw_order15 = 0.25 * (dw[j] * dw[j] - dt) * dw[i] / dt

                    for b_ in self.d2_buf:
                        imul(b_, 0)
                    d2pp = system.diffusion(t, self.p2p[j][i], self.d2_buf)
                    out = _data.iadd(out, d2pp[j], ddw_order15)
                    del d2pp

                    for b_ in self.d2_buf:
                        imul(b_, 0)
                    d2mm = system.diffusion(t, self.p2m[j][i], self.d2_buf)
                    out = _data.iadd(out, d2mm[j], -ddw_order15)
                    del d2mm

                    dwp[j] += -ddw_order15
                    dwm[j] += ddw_order15

            for b_ in self.d2_buf:
                imul(b_, 0)
            d2p = system.diffusion(t, self.v2p[i], self.d2_buf)
            for j in range(num_ops):
                out = _data.iadd(out, d2p[j], dwp[j])
            del d2p

            for b_ in self.d2_buf:
                imul(b_, 0)
            d2m = system.diffusion(t, self.v2m[i], self.d2_buf)
            for j in range(num_ops):
                out = _data.iadd(out, d2m[j], dwm[j])
            del d2m

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


class RouchonSODE(Explicit_Simple_Integrator_Batched):
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
        if not rhs.issuper:
            raise NotImplementedError

    def _make_operators(self):
        rhs = self.rhs
        H = rhs.H
        batch = self.batch
        c_ops = rhs.c_ops
        sc_ops = rhs.sc_ops
        N = len(sc_ops)
        self.num_collapses = N
        dt = self.options["dt"]
        dy = np.zeros((N, batch), dtype=float)
        dw = np.zeros((N, N, batch), dtype=float)
        self.dy = dy
        self.dw = dw

        if H.issuper:
            raise TypeError("The rouchon stochastic integration method can't"
                            " use a premade Liouvillian.")
        self._issuper = rhs.issuper

        def _y(t, _i, cu_args):
            if cu_args.shape[1] != dy.shape[1]:
                return np.zeros(cu_args.shape[1], dtype=float)
            return dy[_i]

        def _w(t, _i, _j, cu_args):
            if cu_args.shape[1] != dy.shape[1]:
                return np.zeros(cu_args.shape[1], dtype=float)
            return dw[_i, _j]

        M = 1 - 1j * dt * H
        ML = 0
        MR = 0
        for op in c_ops:
            M -= op.dag() @ op * (0.5 * dt)
        for i, op in enumerate(sc_ops):
            M -= op.dag() @ op * (0.5 * dt)
            coeff = BatchCoefficient(_y, args={"_i": i, "cu_args": None})

            for part in op.to_list():
                if isinstance(part, qutip.Qobj):
                    ML += qutip.QobjEvo([part, coeff])
                    MR += qutip.QobjEvo([part.dag(), coeff])
                elif isinstance(part, list):
                    qobj, p_coeff = part
                    ML += qutip.QobjEvo([qobj, coeff * p_coeff])
                    MR += qutip.QobjEvo([qobj.dag(), coeff * p_coeff.conj()])
                else:
                    raise NotImplementedError

            for j in range(i+1):
                coeff = BatchCoefficient(
                    _w, args={"_i": i, "_j": j, "cu_args": None}
                )
                oper = sc_ops[j] @ op
                for part in oper.to_list():
                    if isinstance(part, qutip.Qobj):
                        ML += qutip.QobjEvo([part, coeff])
                        MR += qutip.QobjEvo([part.dag(), coeff])
                    elif isinstance(part, list):
                        qobj, p_coeff = part
                        ML += qutip.QobjEvo([qobj, coeff * p_coeff])
                        MR += qutip.QobjEvo([qobj.dag(), coeff * p_coeff.conj()])
                    else:
                        raise NotImplementedError

        self.C = 0
        for op in c_ops:
            self.C += qutip.sprepost(op, op.dag()) * dt
        if c_ops:
            self.C = CuQobjEvo(self.C)

        self.M_l = (
            CuQobjEvo(qutip.spre(ML), batch)
            + CuQobjEvo(qutip.spre(M), 1)
        )
        self.M_r = (
            CuQobjEvo(qutip.spost(MR), batch)
            + CuQobjEvo(qutip.spost(M.dag()), 1)
        )
        self.cpcds = [CuQobjEvo((op + op.dag()) * dt) for op in sc_ops]

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
        self.batch = len(generator)
        self.num_collapses = len(self.rhs.sc_ops)
        if isinstance(generator, Wiener):
            raise NotImplementedError
            self.wiener = generator
        else:
            self.wiener = BatchWiener(
                t, self.options["dt"], generator,
                (1, self.num_collapses)
            )
        self.rhs._register_feedback(self.wiener)
        self._make_operators()
        self._is_set = True

        if self.batch == 1:
            self.state = CuState(state0, self.M_l.hilbert_space_dims)
        else:
            state0 = CuState(state0, self.M_l.hilbert_space_dims)
            self.state = batch_copy(state0, self.batch)

        self._tmp = _data.zeros_like(self.state)
        self._out = _data.zeros_like(self.state)

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
        dW = self.wiener.dW(self.t, N)[:, 0, :, :]

        for dw in dW:
            new_state = self._step(self.t, self.state, dt, dw)
            self.state, self._out = new_state, self.state
            self.t += dt

        return self.t, self.state, np.sum(dW, axis=0)

    def _step(self, t, state, dt, dW):
        dy = np.array([
            op.expect_data(t, state).real + dw
            for op, dw in zip(self.cpcds, dW)
        ])

        N = self.num_collapses

        self.dy[:, :] = dy
        self.dw[:, :, :] = (
            self.dy[:, None, :] * self.dy[None, :, :]
            - np.eye(N)[:, :, None] * dt
        )
        for i in range(N):
            self.dw[i, i] /= 2

        self._tmp = _data.imul(self._tmp, 0)
        self._out = _data.imul(self._out, 0)

        self._tmp = self.M_l.matmul_data(t, state, self._tmp)
        self._out = self.M_r.matmul_data(t, self._tmp, self._out)
        if self.C:
            self._out = self.C.matmul_data(t, state, self._out)

        self._out = _data.imul(self._out, 1/_data.trace_oper_ket(self._out))
        return self._out

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


SMESolver.add_integrator(RouchonSODE, "rouchon")
SMESolver.add_integrator(EulerSODE, "euler")
SMESolver.add_integrator(PlatenSODE, "platen")
SMESolver.add_integrator(Explicit1_5_SODE, "explicit1.5")
SMESolver.add_integrator(Milstein_SODE, "milstein")
SMESolver.add_integrator(PredCorr_SODE, "pred_corr")
