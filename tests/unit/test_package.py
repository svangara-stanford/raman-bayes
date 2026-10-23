"""Packaging smoke tests: the package is installed, typed and has the §2 layout."""

import importlib
import shutil
from importlib.resources import files

import pytest

import raman_bayes

SUBPACKAGES = [
    "domain",
    "ports",
    "io",
    "storage",
    "preprocessing",
    "engine",
    "orchestration",
    "app",
    "config",
    "cli",
]


def test_version_resolves_from_metadata() -> None:
    assert raman_bayes.__version__
    assert raman_bayes.__version__ != "unknown"


def test_package_is_marked_typed() -> None:
    assert files("raman_bayes").joinpath("py.typed").is_file()


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_layout_subpackage_imports(name: str) -> None:
    module = importlib.import_module(f"raman_bayes.{name}")
    assert module.__doc__, f"raman_bayes.{name} needs a docstring stating its role"


def test_console_script_is_installed() -> None:
    assert shutil.which("raman") is not None
