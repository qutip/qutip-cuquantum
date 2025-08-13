""" Tests for qutip_cupy.family. """

import re

from qutip_cuquantum import family


class TestVersion:
    def test_version(self):
        pkg, version = family.version()
        assert pkg == "qutip-cuquantum"
        assert re.match(r"\d+\.\d+\.\d+.*", version)
