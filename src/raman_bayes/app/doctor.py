"""Check that this machine can run the pipeline with a profile (DESIGN.md §7.5, §10 doctor).

Each check is ok, warn, fail or skipped. Paths that do not exist yet are fine when they can
be created (the first write creates them). Datasets ingested more than 60 days ago draw a
warning, because cluster scratch space is purged after 90 days. Checks that only make sense
on Slurm (pixi_bin, live limits, quotas) are skipped on a local profile. Doctor writes
nothing.
"""

import os
import platform
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path

import raman_bayes
from raman_bayes.config.settings import LoadedSettings, Settings
from raman_bayes.domain.errors import CatalogError, RamanBayesError
from raman_bayes.engine.registry import available_engines, load_engine
from raman_bayes.storage.catalog import Catalog

__all__ = ["DATASET_WARN_DAYS", "Check", "CheckStatus", "DoctorReport", "run_doctor"]

DATASET_WARN_DAYS = 60


class CheckStatus(StrEnum):
    """The outcome of one check."""

    OK = "ok"
    WARN = "warn"
    FAIL = "fail"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class Check:
    """One named check and what it found."""

    name: str
    status: CheckStatus
    detail: str


@dataclass(frozen=True)
class DoctorReport:
    """The settings checked and every check, in the order they ran."""

    settings: Settings
    source: str
    engines: tuple[str, ...]
    checks: tuple[Check, ...]

    @property
    def failed(self) -> bool:
        """Whether any check failed."""
        return any(c.status is CheckStatus.FAIL for c in self.checks)


def run_doctor(loaded: LoadedSettings, *, now: datetime | None = None) -> DoctorReport:
    """Run every check for ``loaded``; ``now`` (default: the current time) dates datasets."""
    settings = loaded.settings
    paths = settings.paths
    checks = [
        Check(
            "python", CheckStatus.OK, f"Python {platform.python_version()} on {platform.system()}"
        ),
        Check("package", CheckStatus.OK, f"raman-bayes {raman_bayes.__version__}"),
        _engine(settings.engine.name),
        _folder("paths.data_root", paths.data_root),
        _folder("paths.runs_root", paths.runs_root),
        _folder("paths.catalog", paths.catalog.parent),
        _folder("paths.archive_root", paths.archive_root),
        *_catalog(paths.catalog, now or datetime.now(UTC)),
        *_slurm(settings),
    ]
    return DoctorReport(settings, loaded.source, available_engines(), tuple(checks))


def _engine(name: str) -> Check:
    installed = ", ".join(available_engines()) or "none"
    try:
        engine = load_engine(name)
    except RamanBayesError as exc:
        return Check("engine", CheckStatus.FAIL, str(exc))
    return Check(
        "engine", CheckStatus.OK, f"{engine.name} {engine.version} (installed: {installed})"
    )


def _folder(name: str, path: Path | None) -> Check:
    if path is None:
        return Check(name, CheckStatus.SKIPPED, "not set")
    existing = next(p for p in (path, *path.parents) if p.exists())
    if not existing.is_dir() or not os.access(existing, os.W_OK | os.X_OK):
        return Check(name, CheckStatus.FAIL, f"{path}: {existing} is not a writable folder")
    if existing == path:
        return Check(name, CheckStatus.OK, f"{path} exists and is writable")
    return Check(name, CheckStatus.OK, f"{path} will be created on first use")


def _catalog(path: Path, now: datetime) -> list[Check]:
    try:
        datasets = Catalog(path).list_datasets()
    except CatalogError as exc:
        return [Check("catalog", CheckStatus.FAIL, str(exc))]
    if not path.exists():
        catalog = Check("catalog", CheckStatus.OK, f"{path} will be created on first write")
    else:
        catalog = Check("catalog", CheckStatus.OK, f"{path}: {len(datasets)} dataset(s)")
    limit = timedelta(days=DATASET_WARN_DAYS)
    old = [
        (d.dataset_id, (now - d.created_at).days) for d in datasets if now - d.created_at > limit
    ]
    if not old:
        return [
            catalog,
            Check("datasets.age", CheckStatus.OK, f"none older than {DATASET_WARN_DAYS} days"),
        ]
    listed = ", ".join(f"{dataset_id} ({days} days)" for dataset_id, days in old)
    detail = (
        f"ingested more than {DATASET_WARN_DAYS} days ago: {listed}; "
        "scratch space is purged after 90 days"
    )
    return [catalog, Check("datasets.age", CheckStatus.WARN, detail)]


def _slurm(settings: Settings) -> list[Check]:
    names = ("slurm.pixi_bin", "slurm.limits", "quotas")
    if settings.slurm is None or settings.backend == "local":
        return [Check(n, CheckStatus.SKIPPED, "not used by the local backend") for n in names]
    pixi = settings.slurm.pixi_bin
    found = pixi.is_file() and os.access(pixi, os.X_OK)
    return [
        Check(
            names[0],
            CheckStatus.OK if found else CheckStatus.FAIL,
            f"{pixi}{'' if found else ' is not an executable file'}",
        ),
        Check(names[1], CheckStatus.SKIPPED, "live Slurm limits are not checked yet"),
        Check(names[2], CheckStatus.SKIPPED, "quotas are not checked yet"),
    ]
