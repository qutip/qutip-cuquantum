.. _usage-guide:

**************************************
cuQuantum Backend's Data Layers Guide
**************************************

What the Backend Provides
=========================

The backend introduces two new **data layers** — the internal representation
a ``Qobj`` uses to store its numerical data:

- ``CuOperator``: for (square) operators. Stores the operator symbolically
  as a sum of tensor product terms; the full matrix is never formed.
  The data is used to form cuQuantum's ``Operator`` object lazily when needed.
- ``CuState``: for state vectors and density matrices. Stores data
  via cuQuantum's ``DensePureState`` / ``DenseMixedState``.

A ``Qobj`` backed by ``CuOperator`` or ``CuState`` largely follows the same
QuTiP API as any other ``Qobj``. In practice, however, not all operations are
supported for every combination of data layers, and combining ``Qobj``\s of
different layers may trigger automatic conversions whose cost and correctness
depend on what layers are involved.
*This guide covers those nuances.*


Setup: WorkStream and Activation
=================================

The WorkStream context
----------------------

All ``CuState``\s construction and GPU computation require a cuQuantum
``WorkStream`` context:

.. code-block:: python

    from cuquantum.densitymat import WorkStream
    ctx = WorkStream()

``CuOperator`` is CPU-side and symbolic, so constructing one does not require
the context. ``CuState``, however, reads ``settings.cuDensity["ctx"]``
immediately in its constructor to allocate GPU memory. GPU computation
(applying operators to states, running solvers) also requires the ctx.
This ctx is registered when the backend is activated.

Activating the backend
----------------------

Both ``set_as_default(ctx)`` and the ``CuQuantumBackend(ctx)`` context manager
register the ctx and change the **default data type** used when constructing
new ``Qobj``\s without an explicit ``dtype``. Both also switch the default
solver integrator to ``CuVern7``.

.. code-block:: python

    import qutip_cuquantum

    # Method 1: permanent change
    qutip_cuquantum.set_as_default(ctx)

    # Method 2: scoped to a block; reverts on exit
    with qutip_cuquantum.CuQuantumBackend(ctx):
        ...

**What activation does NOT do:** it does not convert any existing ``Qobj``.
``Qobj``\s do not change their data layer upon entering or leaving the context
— only ``Qobj``\s constructed *after* activation, without an explicit
``dtype``, are affected.

Getting Operators and States into the Right Data Layer
======================================================

Explicit construction
---------------------

The ``dtype`` argument sets the data layer at construction, regardless of
whether the backend is active:

.. code-block:: python

    sz  = qutip.sigmaz(dtype='CuOperator')
    n   = qutip.num(20, dtype='CuOperator')
    psi = qutip.basis(2, 0, dtype='CuState')   # requires ctx to be registered

When no ``dtype`` is given, each factory function resolves the type from the
global default using its own **sparcity hint** — a hint that describes the
structural character of the operator it produces:

- ``"sparse"`` hint (e.g., ``sigmax``, ``sigmay``, ``sigmaz``) →
  ``CSR`` normally, ``CuOperator`` with plugin active
- ``"diagonal"`` hint (e.g., ``num``, ``create``, ``destroy``, ``qeye``) →
  ``Dia`` normally, ``CuOperator`` with plugin active
- ``"dense"`` hint → ``Dense`` always, regardless of whether the plugin is
  active

So activating the plugin changes the default for sparse and diagonal
operators, but **dense operators always stay** ``Dense``.

Explicit conversion
-------------------

Any ``Qobj`` can be converted to a different data layer at any time with
``.to()``:

.. code-block:: python

    sz_cu  = qutip.sigmaz().to('CuOperator')
    n_cu   = qutip.num(20).to('CuOperator')
    psi_cu = qutip.basis(2, 0).to('CuState')   # requires ctx to be registered

Implicit conversion during operations
--------------------------------------

When two ``Qobj``\s with different data layers are combined (``+``, ``@``,
``*``, ``qutip.tensor``, etc.), QuTiP automatically converts one operand
toward the other using the lowest-cost path:

- ``Dense``/``CSR``/``Dia`` → ``CuOperator``: cost 1
- ``CuOperator`` → ``Dense``: cost 10

**This only applies when at least one operand is already** ``CuOperator``.
Implicit conversion cannot produce ``CuOperator`` from scratch — if neither
operand is ``CuOperator``, the result will not be ``CuOperator`` either.

.. code-block:: python

    sx   = qutip.sigmax()                    # CSR
    n_cu = qutip.num(20).to('CuOperator')   # explicit CuOperator

    H = sx & n_cu   # sx is implicitly converted to CuOperator; result is CuOperator

**Pitfall:** ``CuOperator`` only supports square matrices. Implicit conversion
of a rectangular object (e.g., a ket) fails with
``ValueError("Rectangular CuOperator are not supported")``.

Building composite systems
--------------------------

Use ``qutip.tensor()`` or the ``&`` shorthand (they are equivalent) to build
operators on composite Hilbert spaces. For the result to be ``CuOperator``,
at least one operand must already be ``CuOperator``.

**With plugin active** — standard quantum operators (Pauli matrices,
number/creation/annihilation operators, identity) use sparse or diagonal
sparcity hints and therefore become ``CuOperator`` by default, so composite
Hamiltonians are naturally ``CuOperator`` without any explicit conversion:

.. code-block:: python

    qutip_cuquantum.set_as_default(ctx)

    H_xy = qutip.sigmax() & qutip.sigmay()   # all CuOperator

**Without plugin active** — ensure at least one operand is ``CuOperator``;
the others are pulled in by implicit conversion:

.. code-block:: python

    # Convert the subsystem operator to CuOperator explicitly
    sx_cu = qutip.sigmax().to('CuOperator')

    # sigmay() is Dense here; it gets implicitly converted to CuOperator
    # because sx_cu is CuOperator
    H_xy = sx_cu & qutip.sigmay()

Combining terms
---------------

**Addition** (``+``) appends terms to the CuOperator term list:

.. code-block:: python

    H  = qutip.sigmaz() & qutip.qeye(2)   # 1 term
    H += qutip.qeye(2) & qutip.sigmaz()   # 2 terms

**Multiplication** (``@`` or ``*`` between two Qobjs — equivalent for
operators) takes the Cartesian product of the two term lists, naturally
producing multi-mode coupling terms:

.. code-block:: python

    a1 = qutip.destroy(20) & qutip.qeye(20)   # a on mode 0
    a2 = qutip.qeye(20) & qutip.destroy(20)   # a on mode 1

    coupling  = a1.dag() @ a2    # one term with operators on both modes
    coupling += a1 @ a2.dag()

``@``/``*`` of an M-term operator by an N-term operator produces up to M×N
terms; ``+`` produces M+N terms.

**Scalar multiplication** scales every term's coefficient without changing the
term list:

.. code-block:: python

    H = w1 * (a1.dag() @ a1) + w2 * (a2.dag() @ a2) + g * coupling

Diagonal Operators: the Dia Format
====================================

Why it matters
--------------

When a ``CuOperator`` is applied to a ``CuState``, each per-mode operator
stored inside its terms is converted to a cuQuantum type at GPU execution
time:

- Stored ``_data.Dia`` → cuQuantum ``MultidiagonalOperator`` (efficient for
  diagonal/band-diagonal structure)
- Stored ``_data.Dense`` or other → cuQuantum ``DenseOperator``

This conversion is lazy — it happens at GPU execution time, not when
constructing or converting the ``CuOperator``. So the dtype of the object
stored inside the term determines GPU efficiency.

Default dtypes of common operators
-----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 25 30

   * - Operator
     - Default dtype
     - Stored in CuOperator as
     - GPU type at execution
   * - ``num(N)``
     - **Dia**
     - ``_data.Dia``
     - ``MultidiagonalOperator`` (efficient)
   * - ``create(N)``
     - **Dia**
     - ``_data.Dia``
     - ``MultidiagonalOperator`` (efficient)
   * - ``destroy(N)``
     - **Dia**
     - ``_data.Dia``
     - ``MultidiagonalOperator`` (efficient)
   * - ``qeye(N)``
     - **Dia**
     - ``_data.Dia``
     - ``MultidiagonalOperator`` (efficient)
   * - ``sigmaz()``
     - **CSR**
     - ``_data.Dense``
     - ``DenseOperator`` (see note below)
   * - ``sigmax()``
     - **CSR**
     - ``_data.Dense``
     - ``DenseOperator`` (tiny off-diagonal, appropriate)
   * - ``sigmay()``
     - **CSR**
     - ``_data.Dense``
     - ``DenseOperator`` (tiny off-diagonal, appropriate)

``num()``, ``create()``, ``destroy()``, ``qeye()`` default to ``Dia`` and
produce ``MultidiagonalOperator`` without any extra steps. ``sigmaz()`` is
structurally diagonal but defaults to ``CSR``, so it needs explicit handling.
``sigmax()`` and ``sigmay()`` are tiny off-diagonal matrices; ``DenseOperator``
is appropriate for them.

Handling sigmaz and custom diagonal operators
---------------------------------------------

**Without plugin active** — ``sigmaz()`` is ``CSR``; convert via ``Dia``
before converting to ``CuOperator``:

.. code-block:: python

    # stores _data.Dia -> MultidiagonalOperator
    sz_cu = qutip.sigmaz().to('Dia').to('CuOperator')

**With plugin active** — ``sigmaz()`` already returns ``CuOperator`` (with
``Dense`` stored inside). Calling ``.to('Dia')`` on it would first materialize
the full matrix (``CuOperator`` → ``Dense``). Use the ``dtype``
argument to bypass the default instead:

.. code-block:: python

    # stores _data.Dia -> MultidiagonalOperator
    sz_cu = qutip.sigmaz(dtype='Dia').to('CuOperator')

**Custom diagonal operators** constructed from arrays default to ``Dense``;
convert via ``Dia``:

.. code-block:: python

    H_diag = qutip.Qobj(np.diag(energies)).to('Dia').to('CuOperator')

or explicitly construct it as ``Dia`` via ``dtype`` argument:

.. code-block:: python

    H_diag = qutip.Qobj(np.diag(energies), dtype='Dia').to('CuOperator')

CuOperator: Tensor Structure and Performance
=============================================

The symbolic representation
----------------------------

``CuOperator`` stores operators as a **sum of tensor product terms**::

    op = sum_i  factor_i * (A_i0 x A_i1 x ... x A_iN)

Each term records which per-mode operator acts on which subsystem. The full
d^N × d^N matrix is never formed. cuQuantum exploits this factored structure
to apply operators to states efficiently on the GPU — this is a fundamental
source of the performance benefit.

The ``CuOperator`` container structure (term list, mode indices, transforms)
is **CPU-side**. The per-mode operators stored inside each term can be either
CPU-side QuTiP ``Data`` objects (``Dense``, ``Dia``) — the common case when
operators are built from QuTiP functions — or GPU-resident cuQuantum objects
(``DenseOperator``, ``MultidiagonalOperator`` backed by CuPy arrays) if
explicitly constructed that way. In the typical workflow, the per-mode
operators are CPU-side, and data is transferred to the GPU when the
``CuOperator`` is applied to a ``CuState``.

Preserving tensor structure
----------------------------

The symbolic structure is preserved by:

- ``CuOperator + CuOperator`` — appends terms
- ``CuOperator @ CuOperator`` / ``CuOperator * CuOperator`` — Cartesian
  product of terms
- ``kron(CuOperator, CuOperator)`` / ``A & B`` — extends mode indices
- ``adjoint``/``transpose``/``conj`` — applies transform flags per mode
- ``scalar * CuOperator`` — scales coefficients

The tensor structure is **destroyed** by converting to ``Dense`` — whether
via ``.to('Dense')``, ``.full()``, or ``.to_array()``. The reverse conversion
(materialised ``Dense`` back to ``CuOperator``) wraps the full d^N × d^N
matrix as a single dense term acting on the entire Hilbert space, with no
knowledge of the tensor product decomposition. Using such a dense term will
generally result in a considerable performance penalty.

.. code-block:: python

    # AVOID: destroys tensor structure
    H_bad = H.to('Dense').to('CuOperator')
    # for large composite systems, this creates an exponentially large matrix
    H_arr = H.full()

Conversion Reference
====================

Registered conversions
-----------------------

**CuOperator**

.. list-table::
   :header-rows: 1
   :widths: 15 15 10 50

   * - From
     - To
     - Weight
     - Notes
   * - ``Dense``
     - ``CuOperator``
     - 1
     - Square matrices only
   * - ``Dia``
     - ``CuOperator``
     - 1
     - Square matrices only
   * - ``CuOperator``
     - ``Dense``
     - 10
     - Expensive: materializes full matrix on CPU
   * - ``CuPyDense``
     - ``CuOperator``
     - 1
     - If qutip-cupy installed

**CuState**

.. list-table::
   :header-rows: 1
   :widths: 15 15 10 50

   * - From
     - To
     - Weight
     - Notes
   * - ``Dense``
     - ``CuState``
     - 1
     - Host2Device transfer; requires ctx to be registered
   * - ``CuState``
     - ``Dense``
     - 1
     - Device2Host transfer
   * - ``CuPyDense``
     - ``CuState``
     - 1
     - If qutip-cupy installed
   * - ``CuState``
     - ``CuPyDense``
     - 1
     - If qutip-cupy installed

No CSR conversion exists for either type. Paths through CSR go via Dense,
chained automatically.

Registered CuOperator operations
----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 20 40

   * - Operation
     - Result
     - Notes
   * - ``mul(CuOperator, scalar)``
     - ``CuOperator``
     - Symbolic — scales all term coefficients
   * - ``neg(CuOperator)``
     - ``CuOperator``
     - Symbolic — scales all coefficients by −1
   * - ``add(CuOperator, CuOperator)``
     - ``CuOperator``
     - Symbolic — appends term lists
   * - ``sub(CuOperator, CuOperator)``
     - ``CuOperator``
     - Symbolic — appends terms with negated factor
   * - ``adjoint(CuOperator)``
     - ``CuOperator``
     - Symbolic — sets transform flags per mode
   * - ``transpose(CuOperator)``
     - ``CuOperator``
     - Symbolic — sets transform flags per mode
   * - ``conj(CuOperator)``
     - ``CuOperator``
     - Symbolic — sets transform flags per mode
   * - ``kron(CuOperator, CuOperator)``
     - ``CuOperator``
     - Symbolic — extends mode index list; underlying operation for
       ``qutip.tensor`` / ``&``
   * - ``matmul(CuOperator, CuOperator)``
     - ``CuOperator``
     - Symbolic — Cartesian product of term lists; no GPU computation
   * - ``matmul(CuOperator, CuState)``
     - ``CuState``
     - GPU execution
   * - ``matmul(CuState, CuOperator)``
     - ``CuState``
     - GPU execution
   * - ``isequal(CuOperator, CuOperator)``
     - ``bool``
     - Materializes both operators to dense to compare; expensive
   * - ``isherm(CuOperator)``
     - ``bool``
     - Materializes operator to dense to check; expensive
