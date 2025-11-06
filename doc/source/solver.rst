.. _qtcuqu_solver:

************************************
cuQuantum Backend in QuTiP's Solvers
************************************

The ``qutip-cuquantum`` library provides a backend that leverages NVIDIA's cuQuantum SDK to accelerate the simulation of quantum systems.
This backend is specifically designed for simulating **large, composite quantum systems** by performing tensor network contractions on the GPU.

This document outlines how to activate and use this backend with QuTiP's solvers.

==================================
Enabling the cuQuantum Backend
==================================

To use the cuQuantum-based solvers, you must first enable the backend.
This is done by calling the ``set_as_default`` function and providing it with a ``WorkStream`` object from ``cuquantum.densitymat``.

.. code-block:: python

    import qutip
    import qutip_cuquantum
    from cuquantum.densitymat import WorkStream

    # Create a workstream for the backend.
    # This can be configured for multi-GPU and multi-node setups.
    stream = WorkStream()

    # Set cuQuantum as the default backend.
    qutip_cuquantum.set_as_default(stream)

The ``set_as_default`` function changes several QuTiP defaults to route computations through the cuQuantum library.
This includes setting the default data format for quantum objects (``Qobj``) to ``CuOperator`` and configuring the solvers to use GPU-compatible integrators.

This operation can be reversed with:

.. code-block:: python

    qutip_cuquantum.set_as_default(reverse=True)


The backend can also be enabled with a context:

.. code-block:: python

    with CuQuantumBackend(ctx):
        ...

However be careful when mixing core Qutip object and Qutip-cuQuantum's one.
Qutip's Qobj do not keep all the internal structure needed for cuQuantum's optimizations.
Qutip-cuQuantum's states can be distributed in multiple processes and unusable for many qutip's core features. 

==================================
Usage with Solvers
==================================

Once the backend is enabled, you can use compatible QuTiP solvers as you normally would.

Supported Solvers
-----------------
The backend has been tested and is known to work with the following solvers:

* ``qutip.sesolve``
* ``qutip.mesolve``
* ``qutip.mcsolve``

.. note::
    Other solvers have not been tested and may not function correctly with this backend.

Integration Methods
-------------------
The backend introduces its own set of ordinary differential equation (ODE) integration methods, which are wrappers around existing QuTiP methods, optimized for the ``CuOperator`` data type.
You must use one of the following methods in your solver options:

* ``"CuTsit5"``
* ``"CuVern7"``  (New default)
* ``"CuVern9"``

Using any other integration method will result in undefined behaviour.

Example
-------
Here is a simple example of running ``mesolve`` on the GPU:

.. code-block:: python

    # This example assumes the backend has already been set as shown above.

    N = 100
    H = qutip.tensor([qutip.rand_herm(N), qutip.qeye(2)])
    c_ops = [qutip.tensor(qutip.destroy(N), qutip.sigmax())]
    rho0 = qutip.tensor(qutip.basis(N, 0), qutip.basis(2, 1)).proj()
    times = [0.0, 1.0, 2.0]
    e_ops = [qutip.tensor(qutip.num(N), qutip.qeye(2))]

    # No options are needed if you want to use the default "CuVern7" method.
    result = qutip.mesolve(H, rho0, times, c_ops=c_ops, e_ops=e_ops)

=======================================
Performance and Advanced Usage
=======================================

When to Use This Backend
------------------------
The cuQuantum backend excels at simulating **large compound systems**.
Its performance advantage comes from using tensor network algorithms on the GPU, which is most effective when the Hilbert space is composed of many smaller subsystems.
For small, single-component systems, the overhead of GPU data transfers may make it slower than the default CPU-based solvers.

Multi-GPU and Multi-Node Execution
----------------------------------
The ``WorkStream`` object can be configured with MPI for distributed execution across multiple GPUs on a single node or across multiple nodes in a cluster.
For details on creating a multi-GPU or multi-node ``WorkStream``, please refer to the NVIDIA cuQuantum documentation.

However, users should be aware of the performance implications:

* **Single-Node, Multi-GPU**: Scaling from one to multiple GPUs on a single machine introduces significant overhead.
  A performance benefit is typically only seen for **very large** system sizes and requires a properly configured MPI and CUDA environment.

* **Multi-Node**: Extending to multiple nodes adds another layer of communication overhead.
  This is only beneficial for **extremely large** systems that cannot fit on a single node.
  Performance is highly dependent on the cluster's internode GPU communication protocols (e.g., NVLink, InfiniBand), which can be complex to debug and optimize.

==================================
Limitations and Best Practices
==================================

Please be aware of the following limitations when using the cuQuantum backend.

Rectangular Operators
---------------------
Rectangular operators (where the number of rows and columns are not equal) are not supported by the ``CuOperator`` data format.
If your calculation requires rectangular operators as an intermediate step, you must explicitly create them as dense matrices and only convert the final, square operator to the ``CuOperator`` format.

Time-Dependent Systems
----------------------
Time-dependent systems can be defined in two ways in QuTiP: as a list of ``[Operator, Coefficient]`` pairs or as a function that returns a ``Qobj`` at a given time ``t``.

* **Coefficient pairs**: This format is supported and works well with the backend.

* **Function-based `f(t) -> Qobj`**: This format works but can be inefficient if not constructed carefully.
  For best performance, the function should return a time-dependent operator defined on the **smallest possible Hilbert space**.
  The full operator can then be constructed using ``qutip.tensor`` and ``qutip.QobjEvo``.

For example, to create a time-dependent operator acting only on the first subsystem:

.. code-block:: python

    # Recommended approach:
    # Define the time-dependent part on the smallest Hilbert space.
    def g(t):
        # This returns a 2x2 Qobj.
        return (qutip.sigmax() * t).expm()

    # Build the full operator using QobjEvo and tensor products.
    oper = qutip.tensor(qutip.QobjEvo(g), qutip.qeye(M))


.. code-block:: python

    # Avoid this approach:
    # This function is inefficient as it creates a (2*M x 2*M) dense matrix
    # and lose the tensor structure used for efficient computations.
    def f(t):
        return qutip.tensor((qutip.sigmax() * t).expm(), qutip.qeye(M))

    oper = qutip.QobjEvo(f)
