"""Run one task of a plan (DESIGN.md §7.2): the entry point every backend executes.

A task reads ``run_plan.json`` and the dataset store, and writes only its own shard folder;
it never reads the catalog, a profile or the priors file (the caller finds the run folder and
says how old a lock may get). Its spectra are fitted in sub-batches of 512, each written as a
pair of Parquet parts; a pair already written is kept, so a rerun resumes. Each spectrum's
seed depends only on the run seed, the window and the spectrum id, and sub-batches are
independent, so the parts are the same for any number of workers. The engine is used only
through the PeakEngine protocol, and nothing assumes how long a fit takes: the lock's
heartbeat keeps it fresh however long the task runs.
"""

import logging
import multiprocessing
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import TypeVar

import numpy as np
from pydantic import TypeAdapter

from raman_bayes.app.plan import read_plan
from raman_bayes.app.provenance import code_provenance, slurm_ids
from raman_bayes.config.settings import DEFAULT_LOCK_STALE_MINUTES
from raman_bayes.domain.dataset import QCFlag
from raman_bayes.domain.errors import (
    DatasetStoreError,
    EngineContractError,
    EngineMismatchError,
    InvalidEngineError,
    TaskIndexError,
    TaskLockedError,
    UnknownEngineError,
)
from raman_bayes.domain.markers import DoneMarker, LockHolder
from raman_bayes.domain.priors import Window
from raman_bayes.domain.results import WindowFitResult
from raman_bayes.domain.run import RunPlan, TaskSpec
from raman_bayes.engine.registry import load_engine
from raman_bayes.engine.seeding import spectrum_seeds
from raman_bayes.logging import task_log
from raman_bayes.ports.engine import EngineConfig, PeakEngine, PreparedSpectrum
from raman_bayes.preprocessing.prepare import RawSpectrum, prepare
from raman_bayes.storage.result_tables import FitRecord, SpectrumPlace, fits_table, peaks_table
from raman_bayes.storage.shards import (
    PartRows,
    part_is_valid,
    read_done,
    summarize_parts,
    task_dir,
    write_done,
    write_part,
)
from raman_bayes.storage.task_lock import acquire_lock, heartbeat, release_lock
from raman_bayes.storage.zarr_dataset import DatasetArrays, open_dataset

__all__ = [
    "SKIPPED_FLAGS",
    "SUB_BATCH_SPECTRA",
    "RunTaskStatus",
    "TaskOutcome",
    "TaskRequest",
    "run_task",
]

SUB_BATCH_SPECTRA = 512
SKIPPED_FLAGS = QCFlag.NAN_OR_INF | QCFlag.SATURATED  # not fitted; other flags are
_HEARTBEATS_PER_STALE = 4  # a lock is refreshed this many times within its stale limit
_ENGINE_CONFIG = TypeAdapter(EngineConfig)
_log = logging.getLogger(__name__)
_T = TypeVar("_T")


class RunTaskStatus(StrEnum):
    """What run-task did: finished the task, or found it done or held by another process."""

    DONE = "done"
    ALREADY_DONE = "already_done"
    ALREADY_RUNNING = "already_running"


@dataclass(frozen=True)
class TaskRequest:
    """Task ``task_index`` of the run in ``run_dir``.

    ``workers`` processes fit whole sub-batches in parallel; a lock whose last heartbeat is
    ``lock_stale`` old or more is taken over.
    """

    run_dir: Path
    task_index: int
    workers: int = 1
    lock_stale: timedelta = timedelta(minutes=DEFAULT_LOCK_STALE_MINUTES)


@dataclass(frozen=True)
class TaskOutcome:
    """The result of run-task.

    ``marker`` is set when the task is done (now or before); ``holder`` names the process
    holding the lock when the task is already running, and is None if its lock is unreadable.
    """

    status: RunTaskStatus
    task: TaskSpec
    marker: DoneMarker | None = None
    holder: LockHolder | None = None
    n_parts_written: int = 0
    n_parts_kept: int = 0


@dataclass(frozen=True)
class _SubBatch:
    """Spectra ``start`` to ``stop`` (exclusive) of a task, written as parts ``index``."""

    plan: RunPlan
    folder: Path
    index: int
    start: int
    stop: int

    @property
    def rows(self) -> PartRows:
        windows = self.plan.priors.windows
        return PartRows(
            spectrum_ids=tuple(range(self.start, self.stop)),
            n_windows=len(windows),
            n_peaks=sum(len(w.peaks) for w in windows),
        )


@dataclass(frozen=True)
class _Counts:
    """A sub-batch's fits by status, and what happened when ``fit_many`` raised."""

    n_ok: int
    n_failed: int
    n_skipped_qc: int
    refits: tuple[str, ...]


def run_task(request: TaskRequest) -> TaskOutcome:
    """Fit every spectrum of the task and write its parts and ``_DONE.json``.

    Writes nothing when the task is already done or a fresh lock holds it. Raises
    RunPlanNotFoundError or TaskIndexError for a run or task that does not exist,
    InvalidRunPlanError, EngineMismatchError when the plan's engine version is not installed,
    DatasetStoreError when the store is missing or changed since planning, and
    EngineContractError when the engine returns results that do not match its input. A task
    that raises after taking its lock leaves the lock behind.
    """
    plan = read_plan(request.run_dir)
    task = _task(plan, request.task_index)
    folder = task_dir(request.run_dir, task.task_index)
    done = read_done(folder)
    if done is not None:
        return TaskOutcome(RunTaskStatus.ALREADY_DONE, task, marker=done)
    _check_engine(plan)
    _check_store(plan)
    try:
        holder = acquire_lock(folder, request.lock_stale)
    except TaskLockedError as exc:
        return TaskOutcome(RunTaskStatus.ALREADY_RUNNING, task, holder=exc.holder)
    log = request.run_dir / "logs" / f"task-{task.task_index}.log"
    with heartbeat(folder, request.lock_stale / _HEARTBEATS_PER_STALE), task_log(log):
        done = read_done(folder)  # another process may have finished it while we took the lock
        if done is not None:
            release_lock(folder)
            return TaskOutcome(RunTaskStatus.ALREADY_DONE, task, marker=done)
        written, kept = _fit_task(plan, task, folder, request.workers)
        marker = _done_marker(plan, task, folder, holder)
        write_done(folder, marker)
        _log.info(
            "task %d done: %d fits, %d failed", task.task_index, marker.n_fits, marker.n_failed
        )
    release_lock(folder)
    return TaskOutcome(
        RunTaskStatus.DONE, task, marker=marker, n_parts_written=written, n_parts_kept=kept
    )


def _task(plan: RunPlan, index: int) -> TaskSpec:
    if not 0 <= index < len(plan.tasks):
        raise TaskIndexError(plan.run_id, len(plan.tasks), index)
    return plan.tasks[index]


def _check_engine(plan: RunPlan) -> None:
    try:
        engine = load_engine(plan.engine)
    except (UnknownEngineError, InvalidEngineError) as exc:
        raise EngineMismatchError(plan.run_id, plan.engine, plan.engine_version, str(exc)) from exc
    if engine.version != plan.engine_version:
        found = f"{engine.name} {engine.version} is installed"
        raise EngineMismatchError(plan.run_id, plan.engine, plan.engine_version, found)


def _check_store(plan: RunPlan) -> None:
    store = open_dataset(Path(plan.dataset_uri))
    if store.manifest.content_hash != plan.dataset_hash:
        problem = (
            f"content hash {store.manifest.content_hash[:10]} is not the plan's "
            f"{plan.dataset_hash[:10]}; the store changed since the run was planned"
        )
        raise DatasetStoreError(store.path, problem)


def _fit_task(plan: RunPlan, task: TaskSpec, folder: Path, workers: int) -> tuple[int, int]:
    starts = range(task.spec_start, task.spec_stop, SUB_BATCH_SPECTRA)
    batches = [
        _SubBatch(plan, folder, k, start, min(start + SUB_BATCH_SPECTRA, task.spec_stop))
        for k, start in enumerate(starts)
    ]
    todo = [b for b in batches if not part_is_valid(folder, b.index, b.rows)]
    _log.info(
        "task %d: spectra %d to %d in %d sub-batch(es), %d already written",
        task.task_index,
        task.spec_start,
        task.spec_stop - 1,
        len(batches),
        len(batches) - len(todo),
    )
    if workers == 1 or len(todo) <= 1:
        counts = [_fit_sub_batch(b) for b in todo]
    else:
        # spawn, not fork: a forked child would inherit the heartbeat thread and its locks.
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(min(workers, len(todo)), mp_context=context) as pool:
            counts = list(pool.map(_fit_sub_batch, todo))
    for batch, count in zip(todo, counts, strict=True):
        _log_counts(task, batch, count)
    return len(todo), len(batches) - len(todo)


def _log_counts(task: TaskSpec, batch: _SubBatch, counts: _Counts) -> None:
    for refit in counts.refits:
        _log.warning("task %d, sub-batch %d: %s", task.task_index, batch.index, refit)
    _log.info(
        "task %d, sub-batch %d (spectra %d to %d): %d ok, %d failed, %d skipped_qc",
        task.task_index,
        batch.index,
        batch.start,
        batch.stop - 1,
        counts.n_ok,
        counts.n_failed,
        counts.n_skipped_qc,
    )


def _fit_sub_batch(batch: _SubBatch) -> _Counts:
    # Runs in a worker process when there are several, so it loads what it needs itself.
    plan = batch.plan
    engine = load_engine(plan.engine)
    arrays = open_dataset(Path(plan.dataset_uri)).read(batch.start, batch.stop)
    records, refits = _fit_spectra(engine, plan, arrays)
    places = {
        int(sid): SpectrumPlace(
            int(arrays.ix[i]),
            int(arrays.iy[i]),
            int(arrays.iz[i]),
            float(arrays.x_um[i]),
            float(arrays.y_um[i]),
            float(arrays.z_um[i]),
        )
        for i, sid in enumerate(arrays.spectrum_id)
    }
    results = [r.result for r in records]
    write_part(
        batch.folder,
        batch.index,
        fits_table(plan.run_id, records),
        peaks_table(plan.run_id, plan.dataset_id, plan.priors, results, places),
    )
    statuses = [r.status for r in results]
    return _Counts(
        statuses.count("ok"), statuses.count("failed"), statuses.count("skipped_qc"), refits
    )


def _fit_spectra(
    engine: PeakEngine, plan: RunPlan, arrays: DatasetArrays
) -> tuple[list[FitRecord], tuple[str, ...]]:
    # Rows come out spectrum by spectrum, each spectrum's windows in recipe order.
    spectra = [
        RawSpectrum(int(sid), arrays.intensity[i], int(arrays.qc_flags[i]))
        for i, sid in enumerate(arrays.spectrum_id)
    ]
    fittable = [s for s in spectra if not s.qc_flags & SKIPPED_FLAGS]
    cfg = _ENGINE_CONFIG.validate_python(plan.engine_config)
    preprocessing = plan.priors.preprocessing
    by_window: list[dict[int, FitRecord]] = []
    refits: list[str] = []
    for w, window in enumerate(plan.priors.windows):
        prepared = [prepare(s, arrays.wavenumber, window, preprocessing) for s in fittable]
        seeds = spectrum_seeds(plan.seed, w, [p.spectrum_id for p in prepared])
        results, refit = _fit_window(engine, prepared, window, cfg, seeds)
        refits.extend(refit)
        by_window.append(
            {
                p.spectrum_id: FitRecord(r, p.noise_sd)
                for p, r in zip(prepared, results, strict=True)
            }
        )
    records = [
        by_window[w].get(s.spectrum_id) or FitRecord(_skipped(plan, s, window), None)
        for s in spectra
        for w, window in enumerate(plan.priors.windows)
    ]
    return records, tuple(refits)


def _skipped(plan: RunPlan, spectrum: RawSpectrum, window: Window) -> WindowFitResult:
    return WindowFitResult(
        spectrum_id=spectrum.spectrum_id,
        window_id=window.id,
        status="skipped_qc",
        peaks=(),
        diagnostics={},
        runtime_s=0.0,
        engine=plan.engine,
        engine_version=plan.engine_version,
    )


def _fit_window(
    engine: PeakEngine,
    prepared: Sequence[PreparedSpectrum],
    window: Window,
    cfg: EngineConfig,
    seeds: Sequence[np.random.SeedSequence],
) -> tuple[list[WindowFitResult], tuple[str, ...]]:
    if not prepared:
        return [], ()
    results = _engine_call(lambda: engine.fit_many(prepared, window, cfg, seeds))
    if not isinstance(results, Exception):
        _check_results(engine, results, prepared, window)
        return results, ()
    note = (
        f"window {window.id}: fit_many raised {_describe(results)}; "
        "refitting its spectra one at a time"
    )
    return [
        _fit_one(engine, s, window, cfg, seed) for s, seed in zip(prepared, seeds, strict=True)
    ], (note,)


def _fit_one(
    engine: PeakEngine,
    spectrum: PreparedSpectrum,
    window: Window,
    cfg: EngineConfig,
    seed: np.random.SeedSequence,
) -> WindowFitResult:
    result = _engine_call(lambda: engine.fit(spectrum, window, cfg, np.random.default_rng(seed)))
    if isinstance(result, Exception):
        return WindowFitResult(
            spectrum_id=spectrum.spectrum_id,
            window_id=window.id,
            status="failed",
            peaks=(),
            diagnostics={},
            runtime_s=0.0,
            engine=engine.name,
            engine_version=engine.version,
            error=_describe(result),
        )
    _check_results(engine, [result], [spectrum], window)
    return result


def _engine_call(call: Callable[[], _T]) -> _T | Exception:
    # The engine boundary. An engine is third-party code that may raise anything for one bad
    # spectrum; that must cost a failed row, not the task. This is the only place src/ catches
    # every Exception; KeyboardInterrupt and SystemExit are not Exceptions, so they propagate.
    try:
        return call()
    except Exception as exc:  # noqa: BLE001
        return exc


def _describe(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


def _check_results(
    engine: PeakEngine,
    results: Sequence[WindowFitResult],
    prepared: Sequence[PreparedSpectrum],
    window: Window,
) -> None:
    problem = _contract_problem(results, prepared, window)
    if problem is not None:
        raise EngineContractError(engine.name, engine.version, problem)


def _contract_problem(
    results: Sequence[WindowFitResult], prepared: Sequence[PreparedSpectrum], window: Window
) -> str | None:
    if len(results) != len(prepared):
        return f"{len(results)} results for {len(prepared)} spectra in window {window.id}"
    got = [r.spectrum_id for r in results]
    if got != [s.spectrum_id for s in prepared]:
        return f"results for spectra {got[:5]}... are not in the order of its input"
    peak_ids = {p.id for p in window.peaks}
    for result in results:
        if result.window_id != window.id:
            return f"a result for window {result.window_id!r} among window {window.id!r}'s"
        reported = [p.peak_id for p in result.peaks]
        if len(set(reported)) != len(reported) or not set(reported) <= peak_ids:
            return f"spectrum {result.spectrum_id}: peaks {reported} are not window {window.id}'s"
    return None


def _done_marker(plan: RunPlan, task: TaskSpec, folder: Path, holder: LockHolder) -> DoneMarker:
    summary = summarize_parts(folder)
    code = code_provenance()
    job, array = slurm_ids(os.environ)
    return DoneMarker(
        run_id=plan.run_id,
        task_index=task.task_index,
        spec_start=task.spec_start,
        spec_stop=task.spec_stop,
        parts=summary.parts,
        n_fits=summary.n_fits,
        n_ok=summary.n_ok,
        n_failed=summary.n_failed,
        n_skipped_qc=summary.n_skipped_qc,
        n_peaks=summary.n_peaks,
        shard_hash=summary.shard_hash,
        host=holder.host,
        pid=holder.pid,
        slurm_job_id=job,
        slurm_array_job_id=array,
        started_at=holder.started_at,
        completed_at=datetime.now(UTC),
        package_version=code.package_version,
        git_sha=code.git_sha,
        git_dirty=code.git_dirty,
        pixi_lock_hash=code.pixi_lock_hash,
        engine=plan.engine,
        engine_version=plan.engine_version,
    )
