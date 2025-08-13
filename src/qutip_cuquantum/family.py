"""QuTiP family package entry point."""

from .version import version as full_version


def version():
    """Return information to include in qutip.about()."""
    return "qutip-cuquantum", full_version
