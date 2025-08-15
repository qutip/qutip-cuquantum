.. _qtcuqu_solver:

***********************************
cuQuantum Backend in QuTiP's Solver
***********************************


.. _mesolve:

Using cuQuantum in ``mesolve``
==============================

This pluggins adds integrations method that uses ``densitymat``: ``CuTsit5``, ``CuVern7`` and ``CuVern9``.

To use them, we need to call the ``set_as_default`` function:

.. code-block::

    import qutip
    import qutip_cuquantum
    from cuquantum.densitymat import WorkStream

    qutip_cuquantum.set_as_default(WorkStream())

** This operation in not reversible **

The workstream can be multi-gpu, multi-node. See nvidia's documentation for more details.


Once set, we can use qutip mostly as normal:

.. code-block::

    H = qutip.rand_herm(5)
    c_ops = [qutip.destroy(5)]
    rho0 = qutip.basis(5, 4)

    result = qutip.mesolve(H, rho0, [0, 1], c_ops=c_ops, e_ops=[qutip.num(5)])


This backend is optimized for large compound systems.


- Rectangular operators are not supported.
- Time dependent system of the form ``f(t) -> Qobj`` are supported, but not recommended.









