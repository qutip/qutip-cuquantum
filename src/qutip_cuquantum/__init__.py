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
    import cuquantum.densitymat
except ImportError as err:
    raise ImportError("\n".join([
        "cuQuantum.densitymat is not installed or could not be imported.",
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
import numpy


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
from .ode import Result, CuMCIntegrator
from qutip import settings
from qutip.solver import SESolver, MESolver, MCSolver, Result as BaseResult
from qutip.solver.mcsolve import MCIntegrator


class cuDensityOption(QutipOptions):
    _options = {"ctx": None}
    _settings_name = "cuDensity"
    _properties = {}


cuDensityOption_instance = cuDensityOption()
cuDensityOption_instance._set_as_global_default()


def set_as_default(ctx: cuquantum.densitymat.WorkStream=None, reverse=False):
    """
    Update qutip's default to use cuQuantum as a backend.

    Parameters
    ----------
    ctx: WorkStream
        A WorkStream instance from cuquantum.density.
        It can be set with mpi support for multi-gpu simulations.
        Can be ignored when ``reverse=True``.

    reverse: bool, default: False
        Undo the change of default backend to qutip core defaults.
    """
    if not reverse:
        settings.cuDensity["ctx"] = ctx
        settings.core["default_dtype"] = "cuDensity"
        settings.core['numpy_backend'] = cupy

        if True:  # if mpi, how to check from ctx?
            settings.core["auto_real_casting"] = False

        SESolver.solver_options['method'] = "CuVern7"
        MESolver.solver_options['method'] = "CuVern7"
        MCSolver.solver_options['method'] = "CuVern7"

        SESolver._resultclass = Result
        MESolver._resultclass = Result
        MCSolver._trajectory_resultclass = Result
        MCSolver._mc_integrator_class = CuMCIntegrator

    else:
        settings.core["default_dtype"] = "core"
        settings.core['numpy_backend'] = numpy
        settings.core["auto_real_casting"] = True

        SESolver.solver_options['method'] = "adams"
        MESolver.solver_options['method'] = "adams"
        MCSolver.solver_options['method'] = "vern7"

        SESolver._resultclass = BaseResult
        MESolver._resultclass = BaseResult
        MCSolver._trajectory_resultclass = BaseResult
        MCSolver._mc_integrator_class = MCIntegrator



class CuQuantumBackend:
    """
    A context manager class to temporarily set cuQuantum as the default
    backend.

    Parameters
    ----------
    ctx : cuquantum.densitymat.WorkStream
        A WorkStream instance from cuquantum.density.
        It can be set with mpi support for multi-gpu simulations.
    """
    def __init__(self, ctx):
        self.ctx = ctx
        self.previous_values = {}

    def __enter__(self):
        settings.cuDensity["ctx"] = self.ctx
        self.previous_values["default_dtype"] = qutip.settings.core["default_dtype"]
        settings.core["default_dtype"] = "cuDensity"
        self.previous_values["numpy_backend"] = qutip.settings.core["numpy_backend"]
        settings.core['numpy_backend'] = cupy

        self.previous_values["auto_real"] = settings.core["auto_real_casting"]
        if True:  # if mpi, how to check from ctx?
            settings.core["auto_real_casting"] = False

        self.previous_values["SESolverM"] = SESolver.solver_options['method']
        self.previous_values["MESolverM"] = MESolver.solver_options['method']
        self.previous_values["MCSolverM"] = MCSolver.solver_options['method']
        SESolver.solver_options['method'] = "CuVern7"
        MESolver.solver_options['method'] = "CuVern7"
        MCSolver.solver_options['method'] = "CuVern7"

        self.previous_values["SESolverR"] = SESolver._resultclass
        self.previous_values["MESolverR"] = MESolver._resultclass
        self.previous_values["MCSolverR"] = MCSolver._trajectory_resultclass
        self.previous_values["MCSolverI"] = MCSolver._mc_integrator_class
        SESolver._resultclass = Result
        MESolver._resultclass = Result
        MCSolver._trajectory_resultclass = Result
        MCSolver._mc_integrator_class = CuMCIntegrator

    def __exit__(self, exc_type, exc_value, traceback):
        settings.core["default_dtype"] = self.previous_values["default_dtype"]
        settings.core['numpy_backend'] = self.previous_values["numpy_backend"]
        settings.core["auto_real_casting"] = self.previous_values["auto_real"]
        SESolver.solver_options['method'] = self.previous_values["SESolverM"]
        MESolver.solver_options['method'] = self.previous_values["MESolverM"]
        MCSolver.solver_options['method'] = self.previous_values["MCSolverM"]
        SESolver._resultclass = self.previous_values["SESolverR"]
        MESolver._resultclass = self.previous_values["MESolverR"]
        MCSolver._trajectory_resultclass = self.previous_values["MCSolverR"]
        MCSolver._mc_integrator_class = self.previous_values["MCSolverI"]



# Cleaning the namespace
# del Result
# del QutipOptions
# del qutip
