"""One task's shard folder: atomic parts, ``_LOCK`` and ``_DONE.json`` (DESIGN.md §7.2, §8.1)."""

import errno
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from raman_bayes.domain.errors import InvalidDoneMarkerError, TaskLockedError
from raman_bayes.domain.markers import DoneMarker, LockHolder
from raman_bayes.domain.priors import PriorsSpec
from raman_bayes.domain.results import (
    FITS_SCHEMA,
    PEAKS_SCHEMA,
    PeakPosterior,
    PosteriorSummary,
    WindowFitResult,
)
from raman_bayes.storage import atomic, shards, task_lock
from raman_bayes.storage.result_tables import FitRecord, SpectrumPlace, fits_table, peaks_table
from raman_bayes.storage.shards import PartRows

FIXTURE = Path(__file__).parents[1] / "fixtures" / "priors" / "design_example.yaml"
PRIORS = PriorsSpec.model_validate(yaml.safe_load(FIXTURE.read_text(encoding="utf-8")))
STALE = timedelta(minutes=30)
NOW = datetime(2026, 11, 3, 12, 0, tzinfo=UTC)
SUMMARY = PosteriorSummary(mean=1.0, median=1.0, sd=0.1, q025=0.8, q975=1.2)


def _results(ids: Sequence[int], *, present: bool = True) -> list[WindowFitResult]:
    def posterior(peak_id: str) -> PeakPosterior:
        if not present:
            return PeakPosterior(peak_id, 0.1, None, None, None, None, None)
        return PeakPosterior(peak_id, 0.9, SUMMARY, SUMMARY, SUMMARY, SUMMARY, SUMMARY)

    return [
        WindowFitResult(
            spectrum_id=sid,
            window_id=w.id,
            status="ok",
            peaks=tuple(posterior(p.id) for p in w.peaks),
            diagnostics={},
            runtime_s=0.001,
            engine="mock",
            engine_version="0.1.0",
        )
        for sid in ids
        for w in PRIORS.windows
    ]


def _tables(ids: Sequence[int], *, present: bool = True) -> tuple[pa.Table, pa.Table]:
    results = _results(ids, present=present)
    places = {sid: SpectrumPlace(sid, 0, 0, float(sid), 0.0, 0.0) for sid in ids}
    fits = fits_table("run", [FitRecord(r, 2.0) for r in results])
    return fits, peaks_table("run", "ds", PRIORS, results, places)


def _rows(ids: Sequence[int]) -> PartRows:
    return PartRows(spectrum_ids=tuple(ids), n_windows=2, n_peaks=4)


@pytest.fixture
def shard(tmp_path: Path) -> Path:
    path = shards.task_dir(tmp_path / "run", 3)
    path.mkdir(parents=True)
    return path


def _record_replaces(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []
    real = os.replace

    def replace(src: Any, dst: Any) -> None:
        calls.append((Path(src).name, Path(dst).name))
        real(src, dst)

    monkeypatch.setattr(os, "replace", replace)
    return calls


# ---------------------------------------------------------------- layout


def test_task_dir_and_part_names(tmp_path: Path) -> None:
    folder = shards.task_dir(tmp_path, 7)
    assert folder == tmp_path / "shards" / "task=00007"
    paths = shards.part_paths(folder, 12)
    assert (paths.fits.name, paths.peaks.name) == ("fits-012.parquet", "peaks-012.parquet")


# ---------------------------------------------------------------- atomic parts


def test_final_part_absent_until_replace(shard: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real = os.replace
    seen: list[str] = []

    def replace(src: Any, dst: Any) -> None:
        assert Path(src).name.endswith(".tmp")
        assert Path(src).is_file()
        assert not Path(dst).exists()
        seen.append(Path(dst).name)
        real(src, dst)

    monkeypatch.setattr(os, "replace", replace)
    paths = shards.write_part(shard, 0, *_tables([0, 1]))
    assert seen == ["fits-000.parquet", "peaks-000.parquet"]
    assert paths.fits.is_file()
    assert paths.peaks.is_file()
    assert list(shard.glob("*.tmp")) == []


def test_interrupted_write_leaves_no_final_file(
    shard: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(_table: pa.Table, where: Any) -> None:
        where.write(b"PAR1 half a file")
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))

    monkeypatch.setattr(pq, "write_table", broken)
    with pytest.raises(OSError, match=os.strerror(errno.ENOSPC)):
        shards.write_part(shard, 0, *_tables([0, 1]))
    paths = shards.part_paths(shard, 0)
    assert not paths.fits.exists()
    assert not paths.peaks.exists()
    assert not shards.part_is_valid(shard, 0, _rows([0, 1]))


def test_leftover_tmp_ignored_and_replaced_on_resume(shard: Path) -> None:
    paths = shards.part_paths(shard, 0)
    leftovers = [atomic.staging_path(paths.fits), atomic.staging_path(paths.peaks)]
    for leftover in leftovers:
        leftover.write_bytes(b"left by a crash")
    assert not shards.part_is_valid(shard, 0, _rows([0, 1]))
    shards.write_part(shard, 0, *_tables([0, 1]))
    assert shards.part_is_valid(shard, 0, _rows([0, 1]))
    assert not any(leftover.exists() for leftover in leftovers)
    assert shards.summarize_parts(shard).parts == ("fits-000.parquet", "peaks-000.parquet")


def test_all_null_column_stays_float32(shard: Path) -> None:
    paths = shards.write_part(shard, 0, *_tables([0, 1, 2], present=False))
    table = pq.read_table(paths.peaks)
    for name in ("center_mean", "area_q975", "eta_sd", "center_at_bound_frac"):
        column = table.column(name)
        assert column.type == pa.float32()
        assert column.null_count == table.num_rows
    assert table.schema.remove_metadata().equals(PEAKS_SCHEMA)


def test_parts_written_with_explicit_schemas(shard: Path) -> None:
    paths = shards.write_part(shard, 0, *_tables([0, 1]))
    assert pq.read_schema(paths.peaks).remove_metadata().equals(PEAKS_SCHEMA)
    assert pq.read_schema(paths.fits).remove_metadata().equals(FITS_SCHEMA)
    results = [
        WindowFitResult(0, "lithiation", "ok", (), {"rhat": 1.01}, 0.1, "e", "1"),
        WindowFitResult(0, "carbon", "ok", (), {}, 0.1, "e", "1"),
    ]
    fits = fits_table("run", [FitRecord(r, 1.0) for r in results])
    peaks = peaks_table("run", "ds", PRIORS, results, {0: SpectrumPlace(0, 0, 0, 0.0, 0.0, 0.0)})
    paths = shards.write_part(shard, 1, fits, peaks)
    schema = pq.read_schema(paths.fits).remove_metadata()
    assert schema.names == [*FITS_SCHEMA.names, "diag_rhat"]
    assert schema.field("diag_rhat").type == pa.float32()


def test_fsync_before_replace(shard: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    real_fsync, real_replace = os.fsync, os.replace

    def fsync(fd: int) -> None:
        events.append("fsync")
        real_fsync(fd)

    def replace(src: Any, dst: Any) -> None:
        events.append(f"replace {Path(dst).name}")
        real_replace(src, dst)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "replace", replace)
    shards.write_part(shard, 0, *_tables([0]))
    assert events == [
        "fsync",
        "replace fits-000.parquet",
        "fsync",
        "replace peaks-000.parquet",
    ]


def test_fits_written_before_peaks(shard: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _record_replaces(monkeypatch)
    shards.write_part(shard, 4, *_tables([0]))
    assert calls == [
        ("fits-004.parquet.tmp", "fits-004.parquet"),
        ("peaks-004.parquet.tmp", "peaks-004.parquet"),
    ]


# ---------------------------------------------------------------- part validity


def test_sub_batch_valid_only_with_both_files(shard: Path) -> None:
    rows = _rows([0, 1])
    paths = shards.write_part(shard, 0, *_tables([0, 1]))
    assert shards.part_is_valid(shard, 0, rows)
    paths.peaks.unlink()
    assert not shards.part_is_valid(shard, 0, rows)
    paths = shards.write_part(shard, 0, *_tables([0, 1]))
    paths.fits.unlink()
    assert not shards.part_is_valid(shard, 0, rows)


@pytest.mark.parametrize(
    "damage",
    ["truncated_peaks", "garbage_fits", "wrong_rows", "other_spectra", "swapped_schemas"],
)
def test_part_with_wrong_contents_is_invalid(shard: Path, damage: str) -> None:
    rows = _rows([0, 1])
    paths = shards.write_part(shard, 0, *_tables([0, 1]))
    if damage == "truncated_peaks":
        paths.peaks.write_bytes(paths.peaks.read_bytes()[:100])
    elif damage == "garbage_fits":
        paths.fits.write_bytes(b"not parquet")
    elif damage == "wrong_rows":
        shards.write_part(shard, 0, *_tables([0, 1, 2]))
    elif damage == "other_spectra":
        shards.write_part(shard, 0, *_tables([5, 6]))
    else:
        fits, peaks = _tables([0, 1])
        pq.write_table(peaks, paths.fits)
        pq.write_table(fits, paths.peaks)
    assert not shards.part_is_valid(shard, 0, rows)


def test_fits_schema_accepts_only_float32_diag_columns(shard: Path) -> None:
    fits, peaks = _tables([0])
    paths = shards.write_part(shard, 0, fits, peaks)
    for extra in (pa.field("diag_rhat", pa.float64()), pa.field("rhat", pa.float32())):
        column = pa.array([1.0] * fits.num_rows, type=extra.type)
        pq.write_table(fits.append_column(extra, column), paths.fits)
        assert not shards.part_is_valid(shard, 0, _rows([0]))


# ---------------------------------------------------------------- summary and _DONE.json


def test_summarize_parts_counts_rows_and_hashes_parts(shard: Path) -> None:
    shards.write_part(shard, 0, *_tables([0, 1]))
    results = [
        WindowFitResult(2, "lithiation", "failed", (), {}, 0.0, "mock", "0.1.0", "boom"),
        WindowFitResult(2, "carbon", "skipped_qc", (), {}, 0.0, "mock", "0.1.0"),
    ]
    fits = fits_table("run", [FitRecord(r, None) for r in results])
    peaks = peaks_table("run", "ds", PRIORS, results, {2: SpectrumPlace(2, 0, 0, 2.0, 0, 0)})
    shards.write_part(shard, 1, fits, peaks)
    (shard / "fits-002.parquet.tmp").write_bytes(b"ignored")
    summary = shards.summarize_parts(shard)
    names = ("fits-000.parquet", "fits-001.parquet", "peaks-000.parquet", "peaks-001.parquet")
    assert summary.parts == names
    assert (summary.n_fits, summary.n_ok, summary.n_failed, summary.n_skipped_qc) == (6, 4, 1, 1)
    assert summary.n_peaks == 12
    digest = hashlib.sha256()
    for name in names:
        file_hash = hashlib.sha256((shard / name).read_bytes()).hexdigest()
        digest.update(f"{name}\n{file_hash}\n".encode())
    assert summary.shard_hash == digest.hexdigest()


def _marker(**changes: Any) -> DoneMarker:
    fields: dict[str, Any] = {
        "run_id": "r",
        "task_index": 3,
        "spec_start": 0,
        "spec_stop": 2,
        "parts": ("fits-000.parquet", "peaks-000.parquet"),
        "n_fits": 4,
        "n_ok": 4,
        "n_failed": 0,
        "n_skipped_qc": 0,
        "n_peaks": 8,
        "shard_hash": "0" * 64,
        "host": "h",
        "pid": 1,
        "slurm_job_id": None,
        "slurm_array_job_id": None,
        "started_at": NOW,
        "completed_at": NOW,
        "package_version": "0.1.0",
        "git_sha": None,
        "git_dirty": None,
        "pixi_lock_hash": None,
        "engine": "mock",
        "engine_version": "0.1.0",
    }
    return DoneMarker.model_validate(fields | changes)


def test_done_marker_round_trip(shard: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert shards.read_done(shard) is None
    calls = _record_replaces(monkeypatch)
    marker = _marker()
    shards.write_done(shard, marker)
    assert calls == [("_DONE.json.tmp", "_DONE.json")]
    assert shards.read_done(shard) == marker
    assert json.loads((shard / "_DONE.json").read_text("utf-8"))["schema_version"] == "1.0"


def test_unreadable_done_marker_is_an_error(shard: Path) -> None:
    (shard / "_DONE.json").write_text("{", encoding="utf-8")
    with pytest.raises(InvalidDoneMarkerError, match=r"_DONE\.json"):
        shards.read_done(shard)


# ---------------------------------------------------------------- _LOCK


def _write_lock(shard: Path, holder: dict[str, Any], age: timedelta) -> Path:
    path = shard / task_lock.LOCK_FILE
    path.write_text(json.dumps(holder), encoding="utf-8")
    stamp = time.time() - age.total_seconds()
    os.utime(path, (stamp, stamp))
    return path


OTHER = {"host": "elsewhere.example", "pid": 4242, "started_at": "2026-11-03T11:00:00+00:00"}


def test_lock_is_o_excl_json_host_pid_started_at(
    shard: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    flags: list[int] = []
    real_open = os.open

    def recording_open(path: Any, flag: int, *args: Any) -> int:
        if Path(path).name == task_lock.LOCK_FILE:
            flags.append(flag)
        return real_open(path, flag, *args)

    monkeypatch.setattr(os, "open", recording_open)
    holder = task_lock.acquire_lock(shard, STALE, now=NOW)
    assert flags
    assert flags[0] & os.O_CREAT
    assert flags[0] & os.O_EXCL
    content = json.loads((shard / task_lock.LOCK_FILE).read_text("utf-8"))
    assert set(content) == {"host", "pid", "started_at"}
    assert content["host"] == socket.gethostname() == holder.host
    assert content["pid"] == os.getpid() == holder.pid
    assert datetime.fromisoformat(content["started_at"]) == NOW
    with pytest.raises(TaskLockedError, match=str(os.getpid())):
        task_lock.acquire_lock(shard, STALE)


def test_fresh_lock_from_another_host_holds(shard: Path) -> None:
    _write_lock(shard, OTHER, timedelta(minutes=29))
    with pytest.raises(TaskLockedError, match=r"elsewhere\.example") as caught:
        task_lock.acquire_lock(shard, STALE)
    assert caught.value.holder == LockHolder.model_validate(OTHER)


def test_stale_lock_taken_over(shard: Path) -> None:
    _write_lock(shard, OTHER, timedelta(minutes=31))
    state = task_lock.read_lock(shard, STALE)
    assert state is not None
    assert not state.fresh
    holder = task_lock.acquire_lock(shard, STALE, now=NOW)
    assert holder.pid == os.getpid()
    content = json.loads((shard / task_lock.LOCK_FILE).read_text("utf-8"))
    assert content["pid"] == os.getpid()
    assert list(shard.glob("*.tmp")) == []


def test_lock_of_a_dead_process_on_this_host_is_stale(shard: Path) -> None:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    dead = {"host": socket.gethostname(), "pid": child.pid, "started_at": NOW.isoformat()}
    _write_lock(shard, dead, timedelta(seconds=1))
    state = task_lock.read_lock(shard, STALE)
    assert state is not None
    assert state.holder is not None
    assert not state.fresh
    assert task_lock.acquire_lock(shard, STALE).pid == os.getpid()


def test_holder_owned_by_another_user_counts_as_alive(
    shard: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def kill(_pid: int, _signal: int) -> None:
        raise PermissionError(errno.EPERM, os.strerror(errno.EPERM))

    monkeypatch.setattr(os, "kill", kill)
    root = {"host": socket.gethostname(), "pid": 1, "started_at": NOW.isoformat()}
    _write_lock(shard, root, timedelta(seconds=1))
    state = task_lock.read_lock(shard, STALE)
    assert state is not None
    assert state.fresh


def test_lock_released_while_being_examined_is_created_again(
    shard: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_lock(shard, OTHER, timedelta(seconds=1))
    real = task_lock.read_lock
    looks: list[int] = []

    def released_meanwhile(folder: Path, stale_after: timedelta) -> task_lock.LockState | None:
        looks.append(1)
        (folder / task_lock.LOCK_FILE).unlink()
        return real(folder, stale_after)

    monkeypatch.setattr(task_lock, "read_lock", released_meanwhile)
    assert task_lock.acquire_lock(shard, STALE).pid == os.getpid()
    assert looks == [1]


@pytest.mark.parametrize(("minutes", "fresh"), [(1, True), (31, False)])
def test_unreadable_lock_is_judged_by_age(shard: Path, minutes: int, fresh: bool) -> None:
    path = shard / task_lock.LOCK_FILE
    path.write_text("", encoding="utf-8")  # created, not yet written
    stamp = time.time() - minutes * 60
    os.utime(path, (stamp, stamp))
    state = task_lock.read_lock(shard, STALE)
    assert state is not None
    assert state.holder is None
    assert state.fresh is fresh


def test_no_lock_reads_as_none_and_release_is_idempotent(shard: Path) -> None:
    assert task_lock.read_lock(shard, STALE) is None
    task_lock.acquire_lock(shard, STALE)
    task_lock.release_lock(shard)
    assert not (shard / task_lock.LOCK_FILE).exists()
    task_lock.release_lock(shard)


def _wait_for(condition: Callable[[], bool]) -> bool:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


def test_heartbeat_keeps_the_lock_fresh(shard: Path) -> None:
    task_lock.acquire_lock(shard, STALE)
    path = shard / task_lock.LOCK_FILE
    old = time.time() - 3600
    os.utime(path, (old, old))
    with task_lock.heartbeat(shard, timedelta(milliseconds=10)):
        assert _wait_for(lambda: path.stat().st_mtime > old + 3000)
    state = task_lock.read_lock(shard, STALE)
    assert state is not None
    assert state.fresh


def test_heartbeat_stops_when_the_lock_goes(shard: Path) -> None:
    task_lock.acquire_lock(shard, STALE)
    with task_lock.heartbeat(shard, timedelta(milliseconds=10)) as beat:
        task_lock.release_lock(shard)
        assert _wait_for(lambda: not beat.is_alive())
    assert not (shard / task_lock.LOCK_FILE).exists()
