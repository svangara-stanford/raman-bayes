"""The code and scheduler a task runs under, recorded in ``_DONE.json`` (DESIGN.md §7.2, §8.3).

The git fields describe the checkout the package runs from, so they are set only when the
package is ``src/raman_bayes`` of a git working tree, as in a development or pixi install, and
None for an installed wheel even inside some other repository.
"""

import hashlib
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import raman_bayes

__all__ = ["CodeProvenance", "code_provenance", "slurm_ids"]

_PACKAGE_DIR = Path(raman_bayes.__file__).resolve().parent
_GIT_TIMEOUT_S = 10


@dataclass(frozen=True)
class CodeProvenance:
    """Package version, git commit and whether the tree differs from it, and pixi.lock's hash."""

    package_version: str
    git_sha: str | None
    git_dirty: bool | None
    pixi_lock_hash: str | None


def code_provenance(package_dir: Path = _PACKAGE_DIR) -> CodeProvenance:
    """Describe the code in ``package_dir`` (default: this package).

    ``git_dirty`` ignores untracked files. ``pixi_lock_hash`` is the SHA-256 of the checkout's
    ``pixi.lock``. Every git field is None when ``package_dir`` is not in a checkout or git is
    not available.
    """
    version = raman_bayes.__version__
    top = _git(package_dir, "rev-parse", "--show-toplevel")
    if top is None or (Path(top) / "src" / "raman_bayes").resolve() != package_dir.resolve():
        return CodeProvenance(version, None, None, None)
    status = _git(package_dir, "status", "--porcelain", "--untracked-files=no")
    lock = Path(top) / "pixi.lock"
    return CodeProvenance(
        package_version=version,
        git_sha=_git(package_dir, "rev-parse", "HEAD"),
        git_dirty=None if status is None else status != "",
        pixi_lock_hash=hashlib.sha256(lock.read_bytes()).hexdigest() if lock.is_file() else None,
    )


def slurm_ids(environ: Mapping[str, str]) -> tuple[str | None, str | None]:
    """Return ``(<array_id>_<index>, <array_id>)`` from a Slurm array task's environment.

    Either is None when the variables it needs are not set.
    """
    array = environ.get("SLURM_ARRAY_JOB_ID")
    index = environ.get("SLURM_ARRAY_TASK_ID")
    job = f"{array}_{index}" if array is not None and index is not None else None
    return job, array


def _git(where: Path, *args: str) -> str | None:
    try:
        done = subprocess.run(
            ["git", "-C", str(where), *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=_GIT_TIMEOUT_S,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return done.stdout.strip() if done.returncode == 0 else None
