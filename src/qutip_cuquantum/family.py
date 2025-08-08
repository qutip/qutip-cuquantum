"""QuTiP family package entry point."""

from .version import version


def version():
    """Return information to include in qutip.about()."""
    return "qutip-cuquantum", version
