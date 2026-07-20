import numpy as np
import pytest
import qutip
import cuquantum.densitymat as cudense
import qutip_cuquantum
from qutip.solver.heom import DrudeLorentzBath
from qutip_cuquantum.heom.solver import CuHEOMSolver, CuHierarchyADOsState

# Monkeypatch-proof handle to the genuine CPU solver: this name was bound in
# solver.py at import time (before any activation) and is NOT rebound by
# set_as_default's monkeypatch of qutip.solver.heom.HEOMSolver.
from qutip_cuquantum.heom import solver as _heom_mod
ReferenceHEOMSolver = _heom_mod.HEOMSolver

cudm_ctx = cudense.WorkStream()
qutip_cuquantum.set_as_default(cudm_ctx)

QUIET = {"progress_bar": False}
# Drive both integrators to a tight tolerance so their difference is dominated
# by rounding, not by integration error. Comparisons then use a much looser
# tolerance (CMP), giving a ~1e4 margin: integrator noise passes, but any real
# RHS/routing/weight bug produces O(1e-2)+ deviations and fails.
TIGHT = {"rtol": 1e-9, "atol": 1e-9, "nsteps": 10_000}
CMP = dict(atol=1e-5, rtol=1e-5)


def _run_pair(H, bath, depth, rho0, tlist, e_ops=None, options=None):
    opts = {**QUIET, **TIGHT, **(options or {})}
    ref = ReferenceHEOMSolver(H, bath, depth, options=opts).run(
        rho0, tlist, e_ops=e_ops)
    got = CuHEOMSolver(H, bath, depth, options=opts).run(
        rho0, tlist, e_ops=e_ops)
    return ref, got


def _qubit_hamiltonian(eps=0.5, Delta=1.0):
    return 0.5 * eps * qutip.sigmaz() + 0.5 * Delta * qutip.sigmax()


def _rho0():
    # Start on the +z pole. Under H (which mixes z and x) plus a sigmaz bath,
    # all of <sigmaz>, <sigmax>, <sigmay> evolve non-trivially, so every
    # compared observable carries real signal.
    return 0.5 * (qutip.qeye(2) + qutip.sigmaz())


def test_dl_single_qubit_matches_cpu():
    H = _qubit_hamiltonian()
    bath = DrudeLorentzBath(
        qutip.sigmaz(), lam=0.5, gamma=1.0, T=0.5, Nk=2, combine=True,
    )
    rho0 = _rho0()
    tlist = np.linspace(0, 4, 15)
    e_ops = [qutip.sigmaz(), qutip.sigmax(), qutip.sigmay()]

    ref, got = _run_pair(H, bath, 3, rho0, tlist, e_ops=e_ops)

    for i in range(len(e_ops)):
        assert np.allclose(got.expect[i], ref.expect[i], **CMP), (
            f"e_op {i} mismatch: max|diff|="
            f"{np.max(np.abs(np.asarray(got.expect[i]) - ref.expect[i]))}"
        )


@pytest.mark.parametrize("max_depth", [1, 2, 3])
def test_depth_matches_cpu(max_depth):
    H = _qubit_hamiltonian()
    bath = DrudeLorentzBath(
        qutip.sigmaz(), lam=0.5, gamma=1.0, T=0.5, Nk=1, combine=True,
    )
    rho0 = _rho0()
    tlist = np.linspace(0, 4, 15)
    e_ops = [qutip.sigmaz(), qutip.sigmax()]

    ref, got = _run_pair(H, bath, max_depth, rho0, tlist, e_ops=e_ops)

    for i in range(len(e_ops)):
        assert np.allclose(got.expect[i], ref.expect[i], **CMP)


def test_multi_bath_matches_cpu():
    H = _qubit_hamiltonian()
    bath1 = DrudeLorentzBath(
        qutip.sigmaz(), lam=0.5, gamma=1.0, T=0.5, Nk=1, combine=True,
    )
    bath2 = DrudeLorentzBath(
        qutip.sigmax(), lam=0.3, gamma=0.8, T=0.5, Nk=1, combine=True,
    )
    rho0 = _rho0()
    tlist = np.linspace(0, 4, 15)
    e_ops = [qutip.sigmaz(), qutip.sigmax(), qutip.sigmay()]

    ref, got = _run_pair(H, [bath1, bath2], 2, rho0, tlist, e_ops=e_ops)

    for i in range(len(e_ops)):
        assert np.allclose(got.expect[i], ref.expect[i], **CMP)


def test_store_ados_extract_matches_cpu():
    H = _qubit_hamiltonian()
    bath = DrudeLorentzBath(
        qutip.sigmaz(), lam=0.5, gamma=1.0, T=0.5, Nk=1, combine=True,
    )
    rho0 = _rho0()
    tlist = np.linspace(0, 4, 15)

    ref, got = _run_pair(
        H, bath, 3, rho0, tlist, options={"store_ados": True},
    )

    ref_state = ref.ado_states[-1]
    got_state = got.ado_states[-1]
    labels = list(ref_state.labels)

    # Every ADO must match the CPU reference.
    for lbl in labels:
        a = got_state.extract(lbl).full()
        b = ref_state.extract(lbl).full()
        assert np.allclose(a, b, **CMP), f"ADO {lbl} mismatch"

    assert np.allclose(got_state.rho.full(), ref_state.rho.full(), **CMP)


def test_ado_state_roundtrip():
    H = _qubit_hamiltonian()
    bath = DrudeLorentzBath(
        qutip.sigmaz(), lam=0.5, gamma=1.0, T=0.5, Nk=1, combine=True,
    )
    rho0 = _rho0()
    depth = 3
    sz = qutip.sigmaz()
    # store_states forces per-step storage of ado_states even though e_ops is
    # given (otherwise only the final ADO state is kept).
    opts = {**QUIET, **TIGHT, "store_ados": True, "store_states": True}

    full = CuHEOMSolver(H, bath, depth, options=opts).run(
        rho0, np.linspace(0, 4, 15), e_ops=[sz],
    )

    first = CuHEOMSolver(H, bath, depth, options=opts).run(
        rho0, np.linspace(0, 2, 8), e_ops=[sz],
    )
    mid = first.ado_states[-1]
    resumed = CuHEOMSolver(H, bath, depth, options=opts).run(
        mid, np.linspace(2, 4, 8), e_ops=[sz],
    )

    # (a) resuming from a stored ADO state reproduces that state at t0.
    labels = list(mid.labels)
    for lbl in labels:
        assert np.allclose(
            resumed.ado_states[0].extract(lbl).full(),
            mid.extract(lbl).full(),
            atol=1e-8, rtol=1e-8,
        ), f"resumed initial ADO {lbl} does not match fed-in state"

    # (b) split evolution matches the single uninterrupted run at t=4.
    assert np.isclose(resumed.expect[0][-1], full.expect[0][-1], **CMP)


def test_bad_initial_state_dims():
    H = _qubit_hamiltonian()
    bath = DrudeLorentzBath(
        qutip.sigmaz(), lam=0.5, gamma=1.0, T=0.5, Nk=1, combine=True,
    )
    solver = CuHEOMSolver(H, bath, 2, options=QUIET)
    bad_rho = qutip.qeye(3) / 3  # square 3-level state -> wrong system dims
    with pytest.raises(ValueError):
        solver.run(bad_rho, np.linspace(0, 1, 3))
