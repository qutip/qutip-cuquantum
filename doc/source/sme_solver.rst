.. _gpu_stochastic_solvers:

==================================
GPU-Accelerated Stochastic Solvers
==================================

The ``qutip_cuquantum`` package provides GPU-accelerated versions of QuTiP's stochastic solvers,
allowing you to simulate Stochastic Master Equations (SME) and Stochastic Schrodinger Equations at scale.
This solver execute trajectories in parallel using **batching** directly on the GPU.
This approach significantly outperforms traditional CPU-based multiprocessing for large systems
or high trajectory counts.


GPU Batching
============

Unlike the CPU solver, which runs trajectories sequentially or distributes them across CPU cores via multiprocessing,
the GPU solver processes a **batch** of trajectories simultaneously in a single GPU execution pass.
A new solver option ``"batch"`` is introduced to control the size of these parallel chunks.


Basic Usage Example: SMESolver
==============================

To run an SME simulation on the GPU,
you must build your operators within the :class:`CuQuantumBackend` context manager.
You can then configure the :class:`SMESolver` almost identically to QuTiP's, but with a ``batch`` size.

.. code-block:: python

    import qutip as qt
    import numpy as np
    from qutip_cuquantum import CuQuantumBackend, SMESolver
    from cuquantum.densitymat import WorkStream

    # 1. Initialize the cuQuantum local WorkStream.
    # Note: MPI and batching do not work together.
    ctx = WorkStream()

    # Define system dimensions
    N = 3

    # 2. Build your operators inside the CuQuantumBackend context
    with CuQuantumBackend(ctx):
        a = qt.destroy(N)
        I = qt.qeye(N)
        n = qt.num(N)

        # Hamiltonian and collapse operators
        H = (n & n)
        a0 = a & a
        a1 = I & a.dag()

        # Expectation operators
        e1 = I & n
        e2 = n & I
        e3 = (n & I) + (I & n)

    # 3. Define the initial state (can be built outside the context)
    psi0 = qt.basis(N, N-1) & qt.basis(N, N-1)

    # 4. Instantiate the SMESolver with GPU batching options
    sol = SMESolver(
        H,
        sc_ops=[a0, a1],
        c_ops=[],
        heterodyne=False,
        options={
            "method": "platen",
            "batch": 10,  # Run 10 trajectories in parallel on the GPU
            "dt": 0.001,
            "store_final_state": False,
            "keep_runs_results": True,
            "store_measurement": True,
            "progress_bar": False,
        }
    )

    # 5. Run the simulation
    tlist = np.linspace(0, 0.1, 21)
    result = sol.run(
        psi0,
        tlist,
        e_ops=[e1, e2, e3],
        ntraj=10,
        seeds=1023
    )

Supported SDE methods for ``SMESolver``: ``"euler"``, ``"rouchon"``, ``"platen"``, ``"milstein"``, ``"pred_corr"``, ``"explicit1.5"``.
See `QuTiP's API documentation <https://qutip.readthedocs.io/en/stable/apidoc/solver.html#stochastic-integrator>`_ for details.

.. note::

  * With the same seed, operators, parameters, and options, the results will be identical to those obtained with
    QuTiP's :class:`SMESolver` (up to floating-point precision limits).
  * The ``batch`` size does not affect the total number of trajectories calculated or the physical output result.

.. warning::

  * **MPI is not supported with batching**: Multi-GPU features utilizing MPI communication cannot be used with this solver.
  * **The "map" option limits**: While the ``map`` option is still present in the configuration dictionary,
    only the default ``"serial"`` map is expected to work.
    There is currently no mechanism for individual multiprocessing workers to instantiate and manage their own CUDA ``WorkStream``.

Basic Usage Example: SSESolver
==============================

Stochastic Schrödinger simulations are very similar; ``qutip_cuquantum``'s ``SSESolver`` matches QuTiP's core features:

.. code-block:: python

    import qutip as qt
    import numpy as np
    from qutip_cuquantum import CuQuantumBackend, SSESolver
    from cuquantum.densitymat import WorkStream

    # 1. Initialize the cuQuantum local WorkStream.
    # Note: MPI and batching do not work together.
    ctx = WorkStream()

    # Define system dimensions
    N = 3

    # 2. Build your operators inside the CuQuantumBackend context
    with CuQuantumBackend(ctx):
        a = qt.destroy(N)
        I = qt.qeye(N)
        n = qt.num(N)

        # Hamiltonian and collapse operators
        H = (n & n)
        a0 = a & a
        a1 = I & a.dag()

        # Expectation operators
        e1 = I & n
        e2 = n & I
        e3 = (n & I) + (I & n)

    # 3. Define the initial state (can be built outside the context)
    psi0 = qt.basis(N, N-1) & qt.basis(N, N-1)

    # 4. Instantiate the SSESolver with GPU batching options
    sol = SSESolver(
        H,
        sc_ops=[a0, a1],
        heterodyne=False,
        options={
            "method": "platen",
            "batch": 10,         # Run 10 trajectories in parallel on the GPU
            "dt": 0.001,
            "store_final_state": False,
            "keep_runs_results": True,
            "store_measurement": True,
            "progress_bar": False,
        }
    )

    # 5. Run the simulation
    tlist = np.linspace(0, 0.1, 21)
    result = sol.run(
        psi0,
        tlist,
        e_ops={"e1": e1, "e2": e2, "e3": e3},
        ntraj=10,
        seeds=1023
    )

    print(result.average_e_data["e1"])


Supported SDE methods for ``SSESolver``: ``"euler"``, ``"rouchon"``, and ``"platen"``.
See `QuTiP's API documentation <https://qutip.readthedocs.io/en/stable/apidoc/solver.html#stochastic-integrator>`_ for a description of each method.


Memory Management & Batch Size Optimization
===========================================

Executing large batches on the GPU can quickly exhaust the available VRAM,
leading to Out-Of-Memory (OOM) errors.
To assist with this, a utility function is provided to estimate the maximum safe batch size for your specific system and GPU hardware.

.. function:: get_max_batch_size(state: CuState, headroom: float = 0.10, num_copies: int = 1) -> float

    Estimate the optimal number of states to batch in one run of a non-deterministic solver.

    :param state: A sample state initialized with a batch size of 1.
    :type state: CuState
    :param headroom: The fraction of GPU memory to leave unused (default: 0.10, representing 10%).
    :type headroom: float
    :param num_copies: The number of state copies required concurrently during integration. This includes solver derivatives, noise steps, and stored intermediate results.
    :type num_copies: int
    :return: The estimated maximum batch size.
    :rtype: float


While this utility is still undergoing active improvements,
it is fully available for users to integrate into their scripting workflows to avoid hard memory limits:

.. code-block:: python

  from qutip_cuquantum import get_max_batch_size

  max_batch = get_max_batch_size(sample_state, headroom=0.15, num_copies=3)
  print(f"Recommended maximum batch size: {int(max_batch)}")

If you plan on saving the state history (i.e. ``store_states=True``),
you must subtract the memory footprint of your stored trajectories
(:math:`\text{len(tlist)} \times \text{ntraj}`) from the estimated batch budget.


Required State Copies by SDE Solver Method
------------------------------------------

Let :math:`n` be the number of stochastic collapse operators
(which is doubled if running a ``heterodyne`` simulation).
The number of temporary state copies required concurrently by each solver method is:

* **Euler**: :math:`3 + n`
* **Platen**: :math:`4 + 3n`
* **Explicit Taylor (Order 1.5)**: :math:`6 + 5n + 2n^2`
* **Milstein**: :math:`4 + n + n^2`
* **Predictor-Corrector**: :math:`5 + n + n^2`
* **Rouchon**: :math:`3`

Ensure you leave extra buffer memory to hold your system operators.

.. note::

   With the **Rouchon** integration method, operators are only transferred to the GPU when starting the system evolution,
   and their memory overhead scales more aggressively with the number of stochastic collapse operators
   (``sc_ops``) compared to other SDE methods.
