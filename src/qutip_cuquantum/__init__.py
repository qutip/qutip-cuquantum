try:
    import cupy
except ImportError as err:
    raise ImportError("\n".join([
        "CuPy is not installed or could not be imported.",
        "Please install the appropriate version for your CUDA toolkit manually.",
        "For example, for CUDA 12.x, run:",
        "    pip install cupy-cuda12x",
        "For other versions, please see the CuPy installation guide",
        "https://docs.cupy.dev/en/stable/install.html",
    ])) from err

try:
    import cuquantum
except ImportError as err:
    raise ImportError("\n".join([
        "cuQuantum is not installed or could not be imported.",
        "Please install the appropriate version for your CUDA toolkit manually.",
        "For example, for CUDA 12.x, run:",
        "    pip install cuquantum-python-cu12",
        "For other versions, please see the cuQuantum installation guide:",
        "https://docs.nvidia.com/cuda/cuquantum/latest/getting-started/index.html",
    ])) from err


import qutip
from qutip.core.options import QutipOptions
from .operator import CuOperator
from .state import CuState


# TODO: The split per density is not great
# Add an operator / state split in qutip?
qutip.core.data.to.register_group(
    ['cuDensity'],
    dense=qutip.core.data.Dense,
    sparse=CuOperator,
    diagonal=CuOperator
)

try:
    from qutip_cupy import CuPyDense

    def CuOperator_from_CuDense(mat):
        return CuOperator(mat)

    def CuPyDense_from_CuState(mat):
        return CuPyDense(mat.to_cupy())

    def CuState_from_CuPyDense(mat):
        return CuState(mat)

    qutip.core.data.to.add_conversions([
        (CuState, CuPyDense, CuState_from_CuPyDense),
        (CuPyDense, CuState, CuPyDense_from_CuState),
        (CuOperator, CuPyDense, CuOperator_from_CuDense),
    ])

except ImportError:
    pass


from .qobjevo import CuQobjEvo
from .ode import Result


class cuDensityOption(QutipOptions):
    _options = {"ctx": None}
    _settings_name = "cuDensity"
    _properties = {}


cuDensityOption_instance = cuDensityOption()
cuDensityOption_instance._set_as_global_default()


def set_as_default(ctx):
    qutip.settings.cuDensity["ctx"] = ctx
    qutip.settings.core["default_dtype"] = "cuDensity"
    qutip.settings.core['numpy_backend'] = cupy

    if True:  # if mpi, how to check from ctx?
        qutip.settings.core["auto_real_casting"] = False

    qutip.SESolver.solver_options['method'] = "CuVern7"
    qutip.MESolver.solver_options['method'] = "CuVern7"
    qutip.MCSolver.solver_options['method'] = "CuVern7"

    qutip.SESolver._resultclass = Result
    qutip.MESolver._resultclass = Result
    qutip.MCSolver._trajectory_resultclass = Result


# Cleaning the namespace
del Result
del QutipOptions
del qutip
