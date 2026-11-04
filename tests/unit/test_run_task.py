"""``app.run_task``: one task of a plan, from the store to shard parts (DESIGN.md §7.2).

Plans come from the ``write_plan`` fixture, so the catalog is never involved; stores are the
ingested mini fixture (25 real spectra) or synthetic ones when a test needs more spectra or
chosen QC flags.
"""

import os
import socket
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from raman_bayes import __version__
from raman_bayes.app import run_task as run_task_module
from raman_bayes.app.run_task import SUB_BATCH_SPECTRA, RunTaskStatus, TaskRequest, run_task
from raman_bayes.config.settings import PathsSettings
from raman_bayes.domain.dataset import QCFlag
from raman_bayes.domain.errors import EngineContractError
from raman_bayes.domain.markers import LockHolder
from raman_bayes.domain.priors import PriorsSpec, Window
from raman_bayes.domain.results import WindowFitResult
from raman_bayes.engine.mock import MockEngine
from raman_bayes.ports.engine import EngineConfig, PreparedSpectrum
from raman_bayes.storage import shards, task_lock

WritePlan = Callable[..., Path]
SyntheticStore = Callable[..., Path]
SUMMARY_COLUMNS = [
    f"{q}_{s}"
    for q in ("center", "fwhm", "area", "height", "eta")
    for s in ("mean", "median", "sd", "q025", "q975")
]


def _run(run_dir: Path, task: int = 0, **request: Any) -> run_task_module.TaskOutcome:
    return run_task(TaskRequest(run_dir=run_dir, task_index=task, **request))


def _shard(run_dir: Path, task: int = 0) -> Path:
    return shards.task_dir(run_dir, task)


def _table(run_dir: Path, kind: str, task: int = 0) -> pa.Table:
    return pa.concat_tables(
        pq.read_table(p) for p in sorted(_shard(run_dir, task).glob(f"{kind}-*.parquet"))
    )


def _use_engine(monkeypatch: pytest.MonkeyPatch, engine: MockEngine) -> None:
    monkeypatch.setattr(run_task_module, "load_engine", lambda _name: engine)


class RecordingEngine(MockEngine):
    """The mock, recording the spectrum ids each ``fit_many`` call receives."""

    def __init__(self) -> None:
        self.batches: list[list[int]] = []

    def fit_many(
        self,
        ss: Sequence[PreparedSpectrum],
        window: Window,
        cfg: EngineConfig,
        seeds: Sequence[np.random.SeedSequence],
    ) -> list[WindowFitResult]:
        self.batches.append([s.spectrum_id for s in ss])
        return super().fit_many(ss, window, cfg, seeds)


# ---------------------------------------------------------------- a whole task


def test_task_writes_parts_and_done_marker(mini_store: Path, write_plan: WritePlan) -> None:
    run_dir = write_plan(mini_store)
    outcome = _run(run_dir, 1)
    assert outcome.status is RunTaskStatus.DONE
    assert (outcome.task.spec_start, outcome.task.spec_stop) == (16, 25)
    assert (outcome.n_parts_written, outcome.n_parts_kept) == (1, 0)
    shard = _shard(run_dir, 1)
    assert sorted(p.name for p in shard.iterdir()) == [
        "_DONE.json",
        "fits-000.parquet",
        "peaks-000.parquet",
    ]
    assert _table(run_dir, "fits", 1).num_rows == 9 * 2
    assert _table(run_dir, "peaks", 1).num_rows == 9 * 4


def test_done_marker_contents(
    mini_store: Path, write_plan: WritePlan, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "8812")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    run_dir = write_plan(mini_store)
    outcome = _run(run_dir)
    marker = shards.read_done(_shard(run_dir))
    assert marker is not None
    assert marker == outcome.marker
    assert (marker.run_id, marker.task_index) == (run_dir.name, 0)
    assert (marker.spec_start, marker.spec_stop) == (0, 16)
    assert marker.parts == ("fits-000.parquet", "peaks-000.parquet")
    assert marker.n_fits == 16 * 2
    assert marker.n_ok + marker.n_failed + marker.n_skipped_qc == marker.n_fits
    assert marker.n_peaks == 16 * 4
    assert marker.shard_hash == shards.summarize_parts(_shard(run_dir)).shard_hash
    assert (marker.host, marker.pid) == (socket.gethostname(), os.getpid())
    assert (marker.slurm_job_id, marker.slurm_array_job_id) == ("8812_0", "8812")
    assert marker.started_at <= marker.completed_at
    assert marker.package_version == __version__
    assert (marker.engine, marker.engine_version) == ("mock", "0.1.0")


def test_done_marker_atomic_and_lock_removed(
    mini_store: Path, write_plan: WritePlan, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = write_plan(mini_store)
    shard = _shard(run_dir)
    replaced: list[str] = []
    real = os.replace

    def replace_(src: Any, dst: Any) -> None:
        if Path(dst).name == shards.DONE_FILE:
            assert (shard / task_lock.LOCK_FILE).is_file()
            replaced.append(Path(src).name)
        real(src, dst)

    monkeypatch.setattr(os, "replace", replace_)
    _run(run_dir)
    assert replaced == ["_DONE.json.tmp"]
    assert not (shard / task_lock.LOCK_FILE).exists()
    assert list(shard.glob("*.tmp")) == []


def test_sub_batches_of_512(
    synthetic_store: SyntheticStore, write_plan: WritePlan, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert SUB_BATCH_SPECTRA == 512
    engine = RecordingEngine()
    _use_engine(monkeypatch, engine)
    run_dir = write_plan(synthetic_store(1100), spectra_per_task=1100)
    _run(run_dir)
    shard = _shard(run_dir)
    sizes = [
        len(set(pq.read_table(p, columns=["spectrum_id"]).column(0).to_pylist()))
        for p in sorted(shard.glob("fits-*.parquet"))
    ]
    assert sizes == [512, 512, 76]
    # fit_many gets whole sub-batches, once per window.
    assert [len(b) for b in engine.batches] == [512, 512, 512, 512, 76, 76]
    assert engine.batches[0] == list(range(512))


def test_valid_part_skipped_on_resume(
    mini_store: Path, write_plan: WritePlan, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_task_module, "SUB_BATCH_SPECTRA", 8)
    run_dir = write_plan(mini_store, spectra_per_task=25)
    _run(run_dir)
    shard = _shard(run_dir)
    parts = sorted(shard.glob("*.parquet"))
    stamps = {p.name: p.stat().st_mtime_ns for p in parts}
    (shard / shards.DONE_FILE).unlink()  # as if the task stopped before finishing
    outcome = _run(run_dir)
    assert (outcome.n_parts_written, outcome.n_parts_kept) == (0, 4)
    assert {p.name: p.stat().st_mtime_ns for p in parts} == stamps
    assert shards.read_done(shard) is not None


def test_invalid_part_rewritten_on_resume(
    mini_store: Path, write_plan: WritePlan, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_task_module, "SUB_BATCH_SPECTRA", 8)
    run_dir = write_plan(mini_store, spectra_per_task=25)
    _run(run_dir)
    shard = _shard(run_dir)
    before = _table(run_dir, "peaks")
    stamps = {p.name: p.stat().st_mtime_ns for p in shard.glob("*.parquet")}
    (shard / shards.DONE_FILE).unlink()
    peaks = shards.part_paths(shard, 1).peaks
    peaks.write_bytes(peaks.read_bytes()[:64])
    outcome = _run(run_dir)
    assert (outcome.n_parts_written, outcome.n_parts_kept) == (1, 3)
    after = {p.name: p.stat().st_mtime_ns for p in shard.glob("*.parquet")}
    changed = {name for name in stamps if after[name] != stamps[name]}
    assert changed == {"fits-001.parquet", "peaks-001.parquet"}
    assert _table(run_dir, "peaks").equals(before)


# ---------------------------------------------------------------- QC and the engine


def test_nan_and_saturated_skipped_without_fit(
    synthetic_store: SyntheticStore, write_plan: WritePlan, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = RecordingEngine()
    _use_engine(monkeypatch, engine)
    flags = {1: QCFlag.NAN_OR_INF, 2: QCFlag.SATURATED, 3: QCFlag.NAN_OR_INF | QCFlag.NEGATIVE}
    run_dir = write_plan(synthetic_store(8, flags=flags), spectra_per_task=8)
    _run(run_dir)
    fitted = {sid for batch in engine.batches for sid in batch}
    assert fitted == {0, 4, 5, 6, 7}
    fits = _table(run_dir, "fits").to_pylist()
    skipped = [r for r in fits if r["spectrum_id"] in {1, 2, 3}]
    assert len(skipped) == 6
    for row in skipped:
        assert row["status"] == "skipped_qc"
        assert row["noise_sd"] is None
        assert row["runtime_s"] == 0.0
        assert (row["engine"], row["engine_version"], row["error"]) == ("mock", "0.1.0", None)
    peaks = [r for r in _table(run_dir, "peaks").to_pylist() if r["spectrum_id"] in {1, 2, 3}]
    assert len(peaks) == 12
    assert {r["fit_status"] for r in peaks} == {"skipped_qc"}
    assert {r["presence_prob"] for r in peaks} == {None}


def test_negative_and_axis_mismatch_still_fitted(
    synthetic_store: SyntheticStore, write_plan: WritePlan, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = RecordingEngine()
    _use_engine(monkeypatch, engine)
    flags = {0: QCFlag.NEGATIVE, 1: QCFlag.AXIS_MISMATCH}
    run_dir = write_plan(synthetic_store(4, flags=flags), spectra_per_task=4)
    _run(run_dir)
    assert engine.batches == [[0, 1, 2, 3], [0, 1, 2, 3]]
    assert set(_table(run_dir, "fits").column("status").to_pylist()) == {"ok"}


class BatchBrokeError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("the batch broke")


class SpectrumBrokeError(ValueError):
    def __init__(self, spectrum_id: int) -> None:
        super().__init__(f"cannot fit spectrum {spectrum_id}")


class FlakyEngine(MockEngine):
    """``fit_many`` always raises; ``fit`` raises for spectrum 3 only."""

    def __init__(self) -> None:
        self.single: list[int] = []

    def fit_many(self, *_args: object, **_kwargs: object) -> list[WindowFitResult]:
        raise BatchBrokeError

    def fit(
        self, s: PreparedSpectrum, window: Window, cfg: EngineConfig, rng: np.random.Generator
    ) -> WindowFitResult:
        self.single.append(s.spectrum_id)
        if s.spectrum_id == 3:
            raise SpectrumBrokeError(s.spectrum_id)
        return super().fit(s, window, cfg, rng)


def test_per_spectrum_exception_becomes_failed_row(
    mini_store: Path, write_plan: WritePlan, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = write_plan(mini_store, seed=11)
    _run(reference)
    engine = FlakyEngine()
    _use_engine(monkeypatch, engine)
    run_dir = write_plan(mini_store, seed=11)
    _run(run_dir)
    assert engine.single == list(range(16)) * 2  # refitted one at a time, window by window
    fits = _table(run_dir, "fits").to_pylist()
    failed = [r for r in fits if r["status"] == "failed"]
    assert [(r["spectrum_id"], r["window_id"]) for r in failed] == [
        (3, "lithiation"),
        (3, "carbon"),
    ]
    assert {r["error"] for r in failed} == {"SpectrumBrokeError: cannot fit spectrum 3"}
    peaks = _table(run_dir, "peaks").drop_columns(["run_id"]).to_pylist()
    expected = _table(reference, "peaks").drop_columns(["run_id"]).to_pylist()
    # Same seeds, so every other spectrum matches a run whose fit_many worked.
    assert [r for r in peaks if r["spectrum_id"] != 3] == [
        r for r in expected if r["spectrum_id"] != 3
    ]
    for row in (r for r in peaks if r["spectrum_id"] == 3):
        assert row["fit_status"] == "failed"
        assert row["presence_prob"] is None


class BrokenEngine(MockEngine):
    """Returns results that break the fit_many contract in one of several ways."""

    def __init__(self, breach: str) -> None:
        self.breach = breach

    def fit_many(
        self,
        ss: Sequence[PreparedSpectrum],
        window: Window,
        cfg: EngineConfig,
        seeds: Sequence[np.random.SeedSequence],
    ) -> list[WindowFitResult]:
        results = super().fit_many(ss, window, cfg, seeds)
        if self.breach == "too_few":
            return results[:-1]
        if self.breach == "reordered":
            return results[::-1]
        if self.breach == "other_window":
            return [replace(r, window_id="elsewhere") for r in results]
        first = results[0]
        unknown = replace(first.peaks[0], peak_id="zz") if first.peaks else None
        return [replace(first, peaks=(unknown,)) if unknown else first, *results[1:]]


@pytest.mark.parametrize("breach", ["too_few", "reordered", "other_window", "unknown_peak"])
def test_results_that_break_the_contract_stop_the_task(
    mini_store: Path, write_plan: WritePlan, monkeypatch: pytest.MonkeyPatch, breach: str
) -> None:
    _use_engine(monkeypatch, BrokenEngine(breach))
    run_dir = write_plan(mini_store)
    with pytest.raises(EngineContractError, match=r"engine mock 0\.1\.0"):
        _run(run_dir)
    shard = _shard(run_dir)
    assert shards.read_done(shard) is None
    assert (shard / task_lock.LOCK_FILE).is_file()  # left behind: the task stopped early


# ---------------------------------------------------------------- isolation and output


def test_never_touches_catalog(
    mini_store: Path,
    write_plan: WritePlan,
    raman_paths: PathsSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = raman_paths.catalog.read_bytes()

    def refuse(*_args: object, **_kwargs: object) -> None:
        pytest.fail("run-task opened a SQLite database")

    monkeypatch.setattr(sqlite3, "connect", refuse)
    run_dir = write_plan(mini_store)
    assert _run(run_dir).status is RunTaskStatus.DONE
    assert raman_paths.catalog.read_bytes() == catalog


def test_logs_to_task_log(mini_store: Path, write_plan: WritePlan) -> None:
    run_dir = write_plan(mini_store)
    _run(run_dir, 1)
    log = (run_dir / "logs" / "task-1.log").read_text(encoding="utf-8")
    assert "task 1: spectra 16 to 24 in 1 sub-batch(es), 0 already written" in log
    assert "task 1, sub-batch 0 (spectra 16 to 24): 18 ok, 0 failed, 0 skipped_qc" in log
    assert "task 1 done" in log


def test_absent_peaks_null_in_rows(synthetic_store: SyntheticStore, write_plan: WritePlan) -> None:
    run_dir = write_plan(synthetic_store(12), spectra_per_task=12)
    _run(run_dir)
    rows = [r for r in _table(run_dir, "peaks").to_pylist() if r["window_id"] == "lithiation"]
    absent = [r for r in rows if r["presence_prob"] < 0.5]
    present = [r for r in rows if r["presence_prob"] >= 0.5]
    assert absent
    assert present
    for row in absent:
        assert [row[c] for c in SUMMARY_COLUMNS] == [None] * len(SUMMARY_COLUMNS)
    for row in present:
        assert None not in [row[c] for c in SUMMARY_COLUMNS]
    # The even spectra carry peaks 20 noise sds high; the mock finds every one of them.
    assert {r["spectrum_id"] for r in present} >= set(range(0, 12, 2))


def test_end_member_id_mapped_or_null(
    mini_store: Path, write_plan: WritePlan, priors: PriorsSpec
) -> None:
    data = priors.model_dump()
    data["end_members"][1]["peaks"] = ["d"]
    run_dir = write_plan(mini_store, priors=PriorsSpec.model_validate(data))
    _run(run_dir)
    pairs = set(
        zip(
            *_table(run_dir, "peaks").select(["peak_id", "end_member_id"]).to_pydict().values(),
            strict=True,
        )
    )
    assert pairs == {
        ("u1", "lithiated_oxide"),
        ("u2", "lithiated_oxide"),
        ("d", "carbon"),
        ("g", None),
    }


def test_workers_gt_1_identical_parts(
    mini_store: Path, write_plan: WritePlan, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_task_module, "SUB_BATCH_SPECTRA", 8)
    run_dir = write_plan(mini_store, spectra_per_task=25)
    _run(run_dir, workers=1)
    shard = _shard(run_dir)
    one = {
        kind: _table(run_dir, kind).drop_columns(["runtime_s"] if kind == "fits" else [])
        for kind in ("fits", "peaks")
    }
    for path in shard.iterdir():
        path.unlink()
    outcome = _run(run_dir, workers=3)
    assert (outcome.n_parts_written, outcome.n_parts_kept) == (4, 0)
    assert _table(run_dir, "peaks").equals(one["peaks"])
    assert _table(run_dir, "fits").drop_columns(["runtime_s"]).equals(one["fits"])


def test_lock_stale_after_is_the_request_value(mini_store: Path, write_plan: WritePlan) -> None:
    run_dir = write_plan(mini_store)
    shard = _shard(run_dir)
    shard.mkdir(parents=True)
    lock = shard / task_lock.LOCK_FILE
    lock.write_text(
        '{"host": "elsewhere", "pid": 7, "started_at": "2026-11-03T11:00:00Z"}', encoding="utf-8"
    )
    stamp = lock.stat().st_mtime - 600
    os.utime(lock, (stamp, stamp))
    held = _run(run_dir, lock_stale=timedelta(minutes=30))
    assert held.status is RunTaskStatus.ALREADY_RUNNING
    assert held.holder is not None
    assert held.holder.host == "elsewhere"
    assert _run(run_dir, lock_stale=timedelta(minutes=5)).status is RunTaskStatus.DONE


def test_sub_batch_with_every_spectrum_skipped_calls_no_engine(
    synthetic_store: SyntheticStore, write_plan: WritePlan, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = RecordingEngine()
    _use_engine(monkeypatch, engine)
    flags = dict.fromkeys(range(3), QCFlag.SATURATED)
    run_dir = write_plan(synthetic_store(3, flags=flags), spectra_per_task=3)
    outcome = _run(run_dir)
    assert engine.batches == []
    assert outcome.marker is not None
    assert (outcome.marker.n_fits, outcome.marker.n_skipped_qc) == (6, 6)


def test_task_finished_elsewhere_while_taking_the_lock(
    mini_store: Path, write_plan: WritePlan, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = write_plan(mini_store)
    _run(run_dir, 1)
    finished = shards.read_done(_shard(run_dir, 1))
    assert finished is not None
    real = task_lock.acquire_lock

    def acquire_then_find_done(folder: Path, stale: timedelta) -> LockHolder:
        holder = real(folder, stale)
        shards.write_done(folder, finished.model_copy(update={"task_index": 0}))
        return holder

    monkeypatch.setattr(run_task_module, "acquire_lock", acquire_then_find_done)
    outcome = _run(run_dir, 0)
    assert outcome.status is RunTaskStatus.ALREADY_DONE
    assert not (_shard(run_dir, 0) / task_lock.LOCK_FILE).exists()
    assert list(_shard(run_dir, 0).glob("*.parquet")) == []
