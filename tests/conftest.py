"""Scan folders, stores, plans and isolated settings shared by the pipeline tests.

Every fixture copies from ``tests/fixtures/`` into ``tmp_path`` or builds data there; nothing
reads ``data/raw/``. Helpers are fixtures because ``--import-mode=importlib`` makes conftest
unimportable.
"""

import dataclasses
import logging
import os
import shutil
import subprocess
import zipfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from raman_bayes.app.ingest import IngestRequest, ingest
from raman_bayes.config.settings import PathsSettings
from raman_bayes.domain.dataset import DatasetManifest
from raman_bayes.domain.markers import DoneMarker
from raman_bayes.domain.priors import PriorsSpec
from raman_bayes.domain.priors_rules import priors_hash
from raman_bayes.domain.results import PeakPosterior, PosteriorSummary, WindowFitResult
from raman_bayes.domain.run import RunPlan, task_ranges
from raman_bayes.domain.uuid7 import uuid7
from raman_bayes.io.nanofinder import parse_info
from raman_bayes.ports.engine import EngineConfig
from raman_bayes.storage import shards
from raman_bayes.storage.result_tables import FitRecord, SpectrumPlace, fits_table, peaks_table
from raman_bayes.storage.zarr_dataset import (
    DatasetArrays,
    content_hash,
    make_dataset_id,
    open_dataset,
    write_dataset,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
MINI = FIXTURES / "nanofinder_mini" / "NMC811_v0"
DESIGN_EXAMPLE = FIXTURES / "priors" / "design_example.yaml"
PLANNED_AT = datetime(2026, 11, 3, 12, 0, tzinfo=UTC)

# Frame Dimentions edits that turn the real 50 x 25 grid into the 5 x 5 corner the mini fixture
# holds: stop = start + 4 * step on each axis.
_CORNER_GRID = (
    (b"Scanned points X : 50", b"Scanned points X : 5"),
    (b"Axis scan stop, [um]: 77.5378", b"Axis scan stop, [um]: 32.5159"),
    (b"Scanned points Y : 25", b"Scanned points Y : 5"),
    (b"Axis scan stop, [um]: 50.9488", b"Axis scan stop, [um]: 30.9570"),
)

ReplaceOnce = Callable[[Path, bytes, bytes], None]


@pytest.fixture(autouse=True)
def isolated_logging() -> Iterator[None]:
    """Undo any command's logging setup after each test.

    Commands configure the ``raman_bayes`` logger for their process, and tests run many
    commands in one process; pytest also attaches its capture handlers to every logger that
    does not propagate, so a logger left configured would change later tests.
    """
    logger = logging.getLogger("raman_bayes")
    saved = (logger.level, logger.propagate, list(logger.handlers))
    yield
    for handler in [h for h in logger.handlers if h not in saved[2]]:
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(saved[0])
    logger.propagate = saved[1]


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


def design_priors() -> PriorsSpec:
    """Return the §5 example recipe: windows lithiation (u1, u2) and carbon (d, g)."""
    return PriorsSpec.model_validate(yaml.safe_load(DESIGN_EXAMPLE.read_text(encoding="utf-8")))


@pytest.fixture
def priors() -> PriorsSpec:
    """Return the §5 example recipe as a model."""
    return design_priors()


@pytest.fixture
def mini_store(mini_scan: Path, raman_paths: PathsSettings) -> Path:
    """Ingest the mini fixture (25 real spectra, partial grid) and return its store."""
    (outcome,) = ingest(IngestRequest(path=mini_scan, allow_partial_grid=True), raman_paths)
    assert outcome.store_uri is not None, outcome.errors
    return outcome.store_uri


@pytest.fixture
def synthetic_store(tmp_path: Path) -> Callable[..., Path]:
    """Build a store of ``n`` synthetic spectra on the real axis; ``flags`` sets QC bits by id.

    Even spectra carry peaks at 465 and 568 cm^-1 (60 counts over noise of sd 3 on a sloped
    baseline); odd spectra are baseline and noise only. ``counts`` replaces the intensities.
    """

    def build(
        n: int, *, flags: Mapping[int, int] | None = None, counts: np.ndarray | None = None
    ) -> Path:
        info = parse_info(MINI / f"{MINI.name}-Info.txt")
        rng = np.random.default_rng(n)
        axis = np.linspace(-41.55539, 1945.8536, info.n_wn)
        peaks = 60 * (np.exp(-(((axis - 465) / 15) ** 2)) + np.exp(-(((axis - 568) / 15) ** 2)))
        signal = np.where((np.arange(n) % 2 == 0)[:, None], peaks, 0.0)
        if counts is None:
            counts = np.rint(200 + 0.05 * axis + signal + rng.normal(0, 3, (n, axis.size)))
        qc = np.zeros(n, dtype=np.uint16)
        for spectrum_id, bits in (flags or {}).items():
            qc[spectrum_id] = bits
        ids = np.arange(n)
        ix, iy = (ids % info.grid.nx).astype(np.int32), (ids // info.grid.nx).astype(np.int32)
        mask = np.zeros((info.grid.nz, info.grid.ny, info.grid.nx), dtype=bool)
        mask[0, iy, ix] = True
        arrays = DatasetArrays(
            intensity=counts.astype(np.float32),
            wavenumber=axis,
            spectrum_id=ids.astype(np.uint32),
            ix=ix,
            iy=iy,
            iz=np.zeros(n, dtype=np.int32),
            x_um=(info.grid.x.start_um + ix * info.grid.x.step_um).astype(np.float64),
            y_um=(info.grid.y.start_um + iy * info.grid.y.step_um).astype(np.float64),
            z_um=np.full(n, info.grid.z.start_um),
            source_file=tuple(f"SYN-z0y{y}x{x}.txt" for x, y in zip(ix, iy, strict=True)),
            qc_flags=qc,
            grid_mask=mask,
        )
        digest = content_hash(arrays.intensity, arrays.wavenumber, arrays.source_file)
        fields = info.model_dump()
        fields["grid"]["allow_partial"] = True
        fields |= {
            "dataset_id": make_dataset_id("SYN", digest),
            "sample": None,
            "n_spectra": n,
            "content_hash": digest,
            "created_at": PLANNED_AT,
        }
        path = tmp_path / f"synthetic-{n}.zarr"
        write_dataset(path, DatasetManifest.model_validate(fields), arrays)
        return path

    return build


@pytest.fixture
def write_plan(raman_paths: PathsSettings) -> Callable[..., Path]:
    """Write ``run_plan.json`` for ``store`` under ``runs_root`` and return the run folder.

    Plans are built directly, without the catalog, which ``run-task`` never reads. ``fields``
    override plan fields such as ``seed`` (default 7), ``engine`` and ``engine_version``.
    """

    def write(
        store: Path,
        *,
        spectra_per_task: int = 16,
        priors: PriorsSpec | None = None,
        **fields: Any,
    ) -> Path:
        recipe = priors or design_priors()
        manifest = open_dataset(store).manifest
        defaults: dict[str, Any] = {
            "run_id": str(uuid7()),
            "created_at": PLANNED_AT,
            "dataset_id": manifest.dataset_id,
            "dataset_hash": manifest.content_hash,
            "dataset_uri": str(store),
            "n_spectra": manifest.n_spectra,
            "priors": recipe,
            "priors_hash": priors_hash(recipe),
            "engine": "mock",
            "engine_version": "0.1.0",
            "engine_config": dataclasses.asdict(EngineConfig()),
            "backend": "local",
            "profile": "local",
            "resolved_config": {},
            "seed": 7,
            "spectra_per_task": spectra_per_task,
            "tasks": task_ranges(manifest.n_spectra, spectra_per_task),
        }
        plan = RunPlan.model_validate(defaults | fields)
        run_dir = raman_paths.runs_root / plan.run_id
        run_dir.mkdir(parents=True)
        (run_dir / "run_plan.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")
        return run_dir

    return write


@pytest.fixture
def finished_task() -> Callable[..., DoneMarker]:
    """Write a finished task folder by hand: one part pair and a matching ``_DONE.json``.

    Every spectrum in ``ids`` gets one fit per window of the §5 recipe, with ``status`` and
    ``diagnostics``; present peaks have posteriors. Other keywords override done-marker fields.
    """

    def write(
        run_dir: Path,
        task_index: int,
        ids: Sequence[int],
        *,
        status: str = "ok",
        diagnostics: Mapping[str, float] | None = None,
        **marker: Any,
    ) -> DoneMarker:
        recipe = design_priors()
        summary = PosteriorSummary(1.0, 1.0, 0.05, 0.9, 1.1)
        results = [
            WindowFitResult(
                spectrum_id=sid,
                window_id=w.id,
                status=status,  # type: ignore[arg-type]
                peaks=tuple(
                    PeakPosterior(p.id, 0.9, summary, summary, summary, summary, summary)
                    for p in w.peaks
                )
                if status == "ok"
                else (),
                diagnostics=dict(diagnostics or {}),
                runtime_s=0.001,
                engine="mock",
                engine_version="0.1.0",
            )
            for sid in ids
            for w in recipe.windows
        ]
        places = {sid: SpectrumPlace(sid, 0, 0, float(sid), 0.0, 42.0) for sid in ids}
        folder = shards.task_dir(run_dir, task_index)
        folder.mkdir(parents=True, exist_ok=True)
        shards.write_part(
            folder,
            0,
            fits_table(run_dir.name, [FitRecord(r, 1.0) for r in results]),
            peaks_table(run_dir.name, "ds", recipe, results, places),
        )
        parts = shards.summarize_parts(folder)
        fields: dict[str, Any] = {
            "run_id": run_dir.name,
            "task_index": task_index,
            "spec_start": min(ids),
            "spec_stop": max(ids) + 1,
            "parts": parts.parts,
            "n_fits": parts.n_fits,
            "n_ok": parts.n_ok,
            "n_failed": parts.n_failed,
            "n_skipped_qc": parts.n_skipped_qc,
            "n_peaks": parts.n_peaks,
            "shard_hash": parts.shard_hash,
            "host": "node-a",
            "pid": 100 + task_index,
            "slurm_job_id": None,
            "slurm_array_job_id": None,
            "started_at": PLANNED_AT,
            "completed_at": PLANNED_AT,
            "package_version": "0.1.0",
            "git_sha": "abc1234",
            "git_dirty": False,
            "pixi_lock_hash": "f" * 64,
            "engine": "mock",
            "engine_version": "0.1.0",
        }
        done = DoneMarker.model_validate(fields | marker)
        shards.write_done(folder, done)
        return done

    return write
