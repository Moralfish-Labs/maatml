"""MaatML: where a small task-specific model is built, weighed, and packaged."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("maatml")
except PackageNotFoundError:  # source tree with no install
    __version__ = "0.0.0+unknown"

__all__: list[str] = ["__version__"]
