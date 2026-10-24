"""Packaged default profile (DESIGN.md §2 Packaging, §9) and what the built wheel contains."""

import shutil
import subprocess
import zipfile
from importlib.resources import files
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
PROFILE_IN_WHEEL = "raman_bayes/config/profiles/local.yaml"

# The §9 local.yaml, copied by hand.
LOCAL_PROFILE = {
    "profile": "local",
    "paths": {
        "data_root": "./data/datasets",
        "runs_root": "./data/runs",
        "catalog": "./data/catalog.sqlite",
    },
    "backend": "local",
    "local": {"workers": 8},
    "engine": {"name": "mock"},
}


def test_local_profile_loads_via_importlib_resources() -> None:
    resource = files("raman_bayes.config").joinpath("profiles", "local.yaml")
    assert resource.is_file()
    assert yaml.safe_load(resource.read_text(encoding="utf-8")) == LOCAL_PROFILE


@pytest.fixture(scope="module")
def wheel(tmp_path_factory: pytest.TempPathFactory) -> zipfile.ZipFile:
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


def test_wheel_contains_package_data(wheel: zipfile.ZipFile) -> None:
    names = set(wheel.namelist())
    assert {PROFILE_IN_WHEEL, "raman_bayes/py.typed", "raman_bayes/cli/main.py"} <= names
    source = ROOT / "src" / PROFILE_IN_WHEEL
    assert wheel.read(PROFILE_IN_WHEEL) == source.read_bytes()


def test_wheel_declares_console_script(wheel: zipfile.ZipFile) -> None:
    (entry_points,) = [n for n in wheel.namelist() if n.endswith(".dist-info/entry_points.txt")]
    text = wheel.read(entry_points).decode()
    assert "[console_scripts]" in text
    assert "raman = raman_bayes.cli.main:app" in text


def test_wheel_has_only_the_package(wheel: zipfile.ZipFile) -> None:
    tops = {name.split("/", 1)[0] for name in wheel.namelist()}
    assert {t for t in tops if not t.endswith(".dist-info")} == {"raman_bayes"}
