"""raman-bayes: Bayesian Raman peak-detection pipeline (see docs/DESIGN.md)."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("raman-bayes")
except PackageNotFoundError:  # pragma: no cover - only when run from an uninstalled tree
    __version__ = "unknown"

__all__ = ["__version__"]
