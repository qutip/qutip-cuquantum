import pytest
import numpy as np
import qutip
import matplotlib.pyplot as plt
from cuquantum.densitymat import WorkStream

from qutip_cuquantum import CuQuantumBackend
from qutip_cuquantum.stochastic.smesolve import SMESolver as CuSMESolver


ctx = WorkStream()


def setup_operators(N, N_sc=1, H_td=False, sc_td=False, c_ops=False):
    """Generate the operators and initial states."""
    a = qutip.destroy(N)
    I = qutip.qeye(N)
    n = qutip.num(N)

    H = n & n
    if H_td:
        h = (a.dag() & a) + (a & a.dag())
        H += qutip.QobjEvo(h, lambda t: 2-t)

    a0 = a & I * 0.5
    a1 = I & a * 0.2
    a2 = (a.dag() & a.dag()) + (a & a)
    if sc_td:
        a0 = qutip.QobjEvo([a0, lambda t: 0.5 * t])
    sc_ops = [a0, a1, a2][:N_sc]

    e_ops = [I & I, n & I, I & n, (a+a.dag()) & I, I & (a+a.dag())]

    if not c_ops:
        c_ops = []
    else:
        c_ops = [I & a.dag()]

    psi0 = qutip.basis(N, N-1) & qutip.basis(N, N-2)

    return H, sc_ops, c_ops, e_ops, psi0


def compare_evolution(
    solver1, solver2,
    psi0, e_ops,
    atol, rtol,
    compare_measurement=False
):
    tlist = np.linspace(0, 0.1, 21)
    seed = np.random.randint(2**30)

    out_1 = solver1.run(psi0, tlist, e_ops=e_ops, ntraj=1, seeds=seed)
    out_2 = solver2.run(psi0, tlist, e_ops=e_ops, ntraj=1, seeds=seed)

    for idx in range(len(e_ops)):
        np.testing.assert_allclose(
            out_1.average_expect[idx].real,
            out_2.average_expect[idx].real,
            rtol=rtol,
            atol=atol,
        )

    if compare_measurement:
        np.testing.assert_allclose(
            out_1.measurement[0].real.squeeze(),
            out_2.measurement[0].real.squeeze(),
            rtol=rtol,
            atol=atol,
        )


common_sme_method = (
    set(CuSMESolver.avail_integrators().keys())
    & set(qutip.SMESolver.avail_integrators().keys())
)


@pytest.mark.parametrize("method", common_sme_method)
@pytest.mark.parametrize("H_td", [False, True])
@pytest.mark.parametrize("sc_td", [False, True])
@pytest.mark.parametrize("heterodyne", [False, True])
def test_sme_solver_gpu_vs_cpu(method, H_td, sc_td, heterodyne):
    """
    Validates that the GPU-backed cuquantum SMESolver yields results
    matching the CPU-backed QuTiP SMESolver for identical seeds.
    """
    options = {
        "method": method,
        "store_final_state": False,
        "keep_runs_results": False,
        "store_measurement": True,
        "progress_bar": False,
        "dt": 0.001
    }

    H, sc_ops, c_ops, e_ops, psi0 = setup_operators(3, 2)
    cpu_sol = qutip.SMESolver(
        H, sc_ops=sc_ops, c_ops=c_ops, heterodyne=heterodyne, options=options
    )

    gpu_options = options.copy()
    gpu_options["batch"] = 1

    with CuQuantumBackend(ctx):
        # We use the Dia format e_ops as it should be handle better in
        # qutip_cuquantum SMEsolve than the oposite.
        H, sc_ops, c_ops, _, psi0 = setup_operators(3, 2)
    gpu_sol = CuSMESolver(
        H, sc_ops=sc_ops, c_ops=c_ops, heterodyne=heterodyne, options=gpu_options
    )

    compare_evolution(cpu_sol, gpu_sol, psi0, e_ops, rtol=1e-8, atol=1e-10, compare_measurement=True)
