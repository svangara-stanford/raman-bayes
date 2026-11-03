"""Scan folders and isolated settings shared by the ingest, Zarr and pipeline tests.

Every fixture copies from ``tests/fixtures/`` into ``tmp_path``; nothing reads ``data/raw/``.
Helpers are fixtures because ``--import-mode=importlib`` makes conftest unimportable.
"""

import os
import shutil
import subprocess
import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest

from raman_bayes.config.settings import PathsSettings

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
MINI = FIXTURES / "nanofinder_mini" / "NMC811_v0"

# Frame Dimentions edits that turn the real 50 x 25 grid into the 5 x 5 corner the mini fixture
# holds: stop = start + 4 * step on each axis.
_CORNER_GRID = (
    (b"Scanned points X : 50", b"Scanned points X : 5"),
    (b"Axis scan stop, [um]: 77.5378", b"Axis scan stop, [um]: 32.5159"),
    (b"Scanned points Y : 25", b"Scanned points Y : 5"),
    (b"Axis scan stop, [um]: 50.9488", b"Axis scan stop, [um]: 30.9570"),
)

ReplaceOnce = Callable[[Path, bytes, bytes], None]


def _replace_once(path: Path, old: bytes, new: bytes) -> None:
    data = path.read_bytes()
    assert data.count(old) == 1, (path.name, old)
    path.write_bytes(data.replace(old, new))


@pytest.fixture
def replace_once() -> ReplaceOnce:
    """Replace bytes that occur exactly once in a file."""
    return _replace_once


@pytest.fixture
def mini_scan(tmp_path: Path) -> Path:
    """Copy the mini fixture: the real 50 x 25 Info.txt and its 25 corner spectra."""
    return Path(shutil.copytree(MINI, tmp_path / "raw" / MINI.name))


@pytest.fixture
def corner_scan(mini_scan: Path) -> Path:
    """Copy the mini fixture with Info.txt edited to a complete 5 x 5 grid."""
    info = mini_scan / f"{mini_scan.name}-Info.txt"
    for old, new in _CORNER_GRID:
        _replace_once(info, old, new)
    return mini_scan


@pytest.fixture
def raman_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PathsSettings:
    """Point every profile at ``tmp_path/state`` through ``RAMAN_PATHS__*``; chdir to tmp_path."""
    for name in list(os.environ):
        if name.upper().startswith("RAMAN_"):
            monkeypatch.delenv(name)
    root = tmp_path / "state"
    paths = PathsSettings(
        data_root=root / "datasets", runs_root=root / "runs", catalog=root / "catalog.sqlite"
    )
    monkeypatch.setenv("RAMAN_PATHS__DATA_ROOT", str(paths.data_root))
    monkeypatch.setenv("RAMAN_PATHS__RUNS_ROOT", str(paths.runs_root))
    monkeypatch.setenv("RAMAN_PATHS__CATALOG", str(paths.catalog))
    monkeypatch.chdir(tmp_path)
    return paths


@pytest.fixture(scope="session")
def wheel(tmp_path_factory: pytest.TempPathFactory) -> zipfile.ZipFile:
    """Build the wheel once per session with hatchling, as CI's wheel-install job does."""
    exe = shutil.which("hatchling")
    assert exe is not None, "hatchling not on PATH; run via pixi"
    out = tmp_path_factory.mktemp("dist")
    subprocess.run(
        [exe, "build", "-t", "wheel", "-d", str(out)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        timeout=120,
    )
    (path,) = out.glob("raman_bayes-*.whl")
    return zipfile.ZipFile(path)
