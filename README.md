# qutip-cuquantum

Provide cuQuantum's cuDensityMat as a backend for QuTiP.

This backend is specifically designed for simulating **large, composite quantum systems** by performing tensor network contractions on the GPU.

## Installation

qutip-cuquantum is available on pypi.

```
pip install -U qutip>=5.2.1
pip install qutip-cuquantum
```


It require cuquantum to be installed at runtime.
If you have cuda already installed, you can install dependencies with:

```
pip install qutip-cuquantum[cuda11]
```
or
```
pip install qutip-cuquantum[cuda12]
```

If it does not work, you may need to install cuQuantum yourself,
see [nvidia's documentation](https://docs.nvidia.com/cuda/cuquantum/latest/getting-started/index.html#installing-cuquantum) for installation instruction.

## Usage

In simple case, simply calling `set_as_default` before a qutip script should be sufficient to use the backend common solver:

```
import qutip_cuquantum
from cuquantum.densitymat import WorkStream

qutip_cuquantum.set_as_default(WorkStream())
```

qutip-cuquantum work well to speed-up large simulation using `mesolve` or `sesolve`.
However this backend is not compatible with advanced qutip solvers (brmesolve, HEOM) and other various feature.
