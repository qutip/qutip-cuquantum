import qutip
import numpy as np
import qutip_cuquantum
import cuquantum.densitymat as cu_dense
import pytest

cudm_ctx = cu_dense.WorkStream()
qutip_cuquantum.set_as_default(cudm_ctx)


def test_sesolve_cte():
    H = (
        (qutip.sigmaz() & qutip.qeye(2) & qutip.qeye(2)) +
        (qutip.qeye(2) & qutip.sigmaz() & qutip.qeye(2)) +
        (qutip.qeye(2) & qutip.qeye(2) & qutip.sigmaz()) +
        (qutip.sigmax() & qutip.sigmay() & qutip.sigmaz())
    )

    result = qutip.sesolve(
        H, 
        qutip.rand_ket([2, 2, 2]), 
        np.linspace(0, 1, 11),
        e_ops=[qutip.num(2) & qutip.qeye(2) & qutip.qeye(2)]
    )
    assert len(result.expect) == 1
    assert len(result.expect[0]) == 11


def test_mesolve_cte():
    H = (
        (qutip.sigmaz() & qutip.qeye(2) & qutip.qeye(2)) +
        (qutip.qeye(2) & qutip.sigmaz() & qutip.qeye(2)) +
        (qutip.qeye(2) & qutip.qeye(2) & qutip.sigmaz())
    )

    c_ops = [
        (qutip.sigmap() & qutip.sigmam() & qutip.sigmax()),
        (qutip.sigmax() & qutip.sigmap() & qutip.sigmam()),
        (qutip.sigmam() & qutip.sigmax() & qutip.sigmap()),
    ]

    result = qutip.mesolve(
        H, 
        qutip.rand_ket([2, 2, 2]), 
        np.linspace(0, 1, 11),
        c_ops=c_ops,
        e_ops=[qutip.num(2) & qutip.qeye(2) & qutip.qeye(2)]
    )
    assert len(result.expect) == 1
    assert len(result.expect[0]) == 11


def test_mesolve_td_H():
    H = [(
        (qutip.sigmaz() & qutip.qeye(2) & qutip.qeye(2)) +
        (qutip.qeye(2) & qutip.sigmaz() & qutip.qeye(2)) +
        (qutip.qeye(2) & qutip.qeye(2) & qutip.sigmaz())
    ), [
        (qutip.sigmax() & qutip.sigmax() & qutip.sigmax()),
        lambda t: np.sin(t)
    ]]

    c_ops = [
        (qutip.sigmap() & qutip.sigmam() & qutip.sigmax()),
        (qutip.sigmax() & qutip.sigmap() & qutip.sigmam()),
        (qutip.sigmam() & qutip.sigmax() & qutip.sigmap()),
    ]

    result = qutip.mesolve(
        H,
        qutip.rand_ket([2, 2, 2]),
        np.linspace(0, 1, 11),
        c_ops=c_ops,
        e_ops=[qutip.num(2) & qutip.qeye(2) & qutip.qeye(2)]
    )
    assert len(result.expect) == 1
    assert len(result.expect[0]) == 11


def test_mesolve_td_c_op():

    def f(t):
        return np.sin(t)

    H = (
        (qutip.sigmaz() & qutip.qeye(2) & qutip.qeye(2)) +
        (qutip.qeye(2) & qutip.sigmaz() & qutip.qeye(2)) +
        (qutip.qeye(2) & qutip.qeye(2) & qutip.sigmaz())
    )

    c_ops = [
        (qutip.sigmap() & qutip.sigmam() & qutip.sigmax()),
        (qutip.sigmax() & qutip.sigmap() & qutip.sigmam()),
        [(qutip.sigmam() & qutip.sigmax() & qutip.sigmap()), f]
    ]

    result = qutip.mesolve(
        H, 
        qutip.rand_ket([2, 2, 2]), 
        np.linspace(0, 1, 11),
        c_ops=c_ops,
        e_ops=[qutip.num(2) & qutip.qeye(2) & qutip.qeye(2)]
    )
    assert len(result.expect) == 1
    assert len(result.expect[0]) == 11


def test_mesolve_H_func():

    def op(t):
        return qutip.sigmap() * np.sin(t)

    H0 = (
        (qutip.sigmaz() & qutip.qeye(2) & qutip.qeye(2)) +
        (qutip.qeye(2) & qutip.sigmaz() & qutip.qeye(2)) +
        (qutip.qeye(2) & qutip.qeye(2) & qutip.sigmaz())
    )

    H = H0 + (qutip.QobjEvo(op).trans() & qutip.qeye(2) & qutip.sigmaz())

    c_ops = [
        (qutip.sigmap() & qutip.sigmam() & qutip.sigmax()),
        (qutip.sigmax() & qutip.sigmap() & qutip.sigmam()),
        (qutip.sigmam() & qutip.sigmax() & qutip.sigmap()),
    ]

    result = qutip.mesolve(
        H, 
        qutip.rand_ket([2, 2, 2]), 
        np.linspace(0, 1, 11),
        c_ops=c_ops,
        e_ops=[qutip.num(2) & qutip.qeye(2) & qutip.qeye(2)]
    )
    assert len(result.expect) == 1
    assert len(result.expect[0]) == 11

