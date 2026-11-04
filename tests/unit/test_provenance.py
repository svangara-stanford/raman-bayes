"""Code provenance and Slurm ids recorded in ``_DONE.json`` (DESIGN.md §7.2, §8.3)."""

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

import raman_bayes
from raman_bayes.app.provenance import CodeProvenance, code_provenance, slurm_ids

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(raman_bayes.__file__).resolve().parent


def _git(*args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True
    )
    return done.stdout.strip()


def test_code_provenance_in_this_checkout() -> None:
    found = code_provenance()
    assert found.package_version == raman_bayes.__version__
    assert found.git_sha == _git("rev-parse", "HEAD")
    assert found.git_dirty is (_git("status", "--porcelain", "--untracked-files=no") != "")
    assert found.pixi_lock_hash == hashlib.sha256((ROOT / "pixi.lock").read_bytes()).hexdigest()


def test_outside_a_checkout_git_fields_are_none(tmp_path: Path) -> None:
    package = tmp_path / "site-packages" / "raman_bayes"
    package.mkdir(parents=True)
    expected = CodeProvenance(raman_bayes.__version__, None, None, None)
    assert code_provenance(package) == expected


def test_a_package_inside_an_unrelated_repository_is_not_a_checkout(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    package = tmp_path / "venv" / "site-packages" / "raman_bayes"
    package.mkdir(parents=True)
    assert code_provenance(package).git_sha is None


def test_without_git_on_path_git_fields_are_none(monkeypatch: pytest.MonkeyPatch) -> None:
    assert shutil.which("git") is not None
    monkeypatch.setenv("PATH", "")
    found = code_provenance(PACKAGE)
    assert (found.git_sha, found.git_dirty, found.pixi_lock_hash) == (None, None, None)


@pytest.mark.parametrize(
    ("environ", "expected"),
    [
        ({"SLURM_ARRAY_JOB_ID": "8812", "SLURM_ARRAY_TASK_ID": "4"}, ("8812_4", "8812")),
        ({"SLURM_ARRAY_JOB_ID": "8812"}, (None, "8812")),
        ({}, (None, None)),
    ],
)
def test_slurm_ids(environ: dict[str, str], expected: tuple[str | None, str | None]) -> None:
    assert slurm_ids(environ) == expected
