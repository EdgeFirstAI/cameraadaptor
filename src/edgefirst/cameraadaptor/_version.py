"""Version information for edgefirst-cameraadaptor.

The version is declared once, in ``pyproject.toml``, and read here from the
installed package's metadata.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("edgefirst-cameraadaptor")
except PackageNotFoundError:  # pragma: no cover - source tree not installed
    __version__ = "0+unknown"
