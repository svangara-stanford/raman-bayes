"""LocalBackend and the submit driver (DESIGN.md §7.3, §7.4): spawn pool, poll, cancel, resubmit.

Real submissions run tasks in spawned processes, which cannot see test monkeypatches; a task is
made to fail by turning its log file into a folder, and to look held elsewhere by a fresh lock
from another host.
"""

import dataclasses
import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any, ClassVar

import pytest

from raman_bayes.app.run_task import TaskRequest, run_task, run_task_in_worker
from raman_bayes.app.runs import open_run
from raman_bayes.app.submit import SubmitRequest, SubmitStatus, submit_run
from raman_bayes.config.settings import PathsSettings, load_settings
from raman_bayes.domain.errors import (
    InvalidSubmissionsError,
    MaxAttemptsError,
    RunNotInCatalogError,
)
from raman_bayes.domain.markers import LockHolder
from raman_bayes.domain.run import RunStatus, SubmissionRecord, TaskStatus
from raman_bayes.orchestration import local
from raman_bayes.orchestration.local import LOCAL_DIR, LocalBackend
from raman_bayes.storage import shards, task_lock
from raman_bayes.storage.catalog import Catalog, RunRecord
from raman_bayes.storage.run_files import append_submission, read_plan, read_submissions

WritePlan = Callable[..., Path]
STALE = timedelta(minutes=30)
OTHER_HOST = {"host": "elsewhere.example", "pid": 4242, "started_at": "2026-11-03T11:00:00Z"}


def _backend(paths: PathsSettings, workers: int = 2) -> LocalBackend:
    runner = partial(run_task_in_worker, lock_stale=STALE)
    return LocalBackend(paths.runs_root, workers, STALE, runner)


def _lock(run_dir: Path, task: int, holder: dict[str, Any], minutes_old: float = 0) -> None:
    folder = shards.task_dir(run_dir, task)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / task_lock.LOCK_FILE
    path.write_text(json.dumps(holder), encoding="utf-8")
    stamp = time.time() - minutes_old * 60
    os.utime(path, (stamp, stamp))


def _record(run_id: str, attempt: int, tasks: tuple[int, ...]) -> SubmissionRecord:
    return SubmissionRecord(
        run_id=run_id,
        attempt=attempt,
        backend="local",
        submitted_at=datetime(2026, 11, 3, 11, 0, tzinfo=UTC),
        host=socket.gethostname(),
        dry_run=False,
        task_indices=tasks,
    )


def _register(paths: PathsSettings, run_dir: Path) -> None:
    """Put a run written by ``write_plan`` in the catalog, as ``plan`` would."""
    plan = read_plan(run_dir)
    catalog = Catalog(paths.catalog)
    catalog.add_priors_spec(plan.priors, plan.priors_hash, "{}", plan.created_at)
    record = RunRecord(
        run_id=plan.run_id,
        dataset_id=plan.dataset_id,
        priors_hash=plan.priors_hash,
        engine=plan.engine,
        engine_version=plan.engine_version,
        config_hash="c",
        git_sha=None,
        backend=plan.backend,
        profile=plan.profile,
        seed=plan.seed,
        created_at=plan.created_at,
    )
    catalog.create_run(record, plan.tasks)


class RecordingPool(ProcessPoolExecutor):
    made: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        RecordingPool.made.append(kwargs)
        super().__init__(*args, **kwargs)


# ---------------------------------------------------------------- submit and wait


def test_submit_runs_all_tasks_in_spawn_pool(
    mini_store: Path,
    write_plan: WritePlan,
    raman_paths: PathsSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    RecordingPool.made.clear()
    monkeypatch.setattr(local, "ProcessPoolExecutor", RecordingPool)
    run_dir = write_plan(mini_store)
    backend = _backend(raman_paths, workers=8)
    backend.submit(read_plan(run_dir))
    assert (run_dir / LOCAL_DIR / task_lock.LOCK_FILE).is_file()  # held while the pool runs
    assert backend.wait(run_dir.name) == {}
    assert not (run_dir / LOCAL_DIR / task_lock.LOCK_FILE).exists()
    (made,) = RecordingPool.made
    assert made["mp_context"].get_start_method() == "spawn"
    assert made["max_workers"] == 2  # 8 workers, but only 2 tasks
    assert made["max_tasks_per_child"] == 1
    for task in (0, 1):
        assert shards.read_done(shards.task_dir(run_dir, task)) is not None
    states = backend.poll(run_dir.name)
    assert {s.status for s in states.values()} == {TaskStatus.DONE}


def test_submit_returns_record_attempt_1(
    mini_store: Path, write_plan: WritePlan, raman_paths: PathsSettings
) -> None:
    run_dir = write_plan(mini_store)
    backend = _backend(raman_paths)
    before = datetime.now(UTC)
    record = backend.submit(read_plan(run_dir))
    backend.wait(run_dir.name)
    assert record.run_id == run_dir.name
    assert (record.attempt, record.backend, record.dry_run) == (1, "local", False)
    assert record.task_indices == (0, 1)
    assert record.host == socket.gethostname()
    assert (record.slurm_array_job_id, record.script) == (None, None)
    assert before <= record.submitted_at <= datetime.now(UTC)
    assert read_submissions(run_dir) == ()  # the driver records submissions, not the backend


def test_dry_run_writes_nothing_changes_nothing(
    mini_store: Path,
    write_plan: WritePlan,
    raman_paths: PathsSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = write_plan(mini_store)
    _register(raman_paths, run_dir)
    before = sorted(p.relative_to(run_dir) for p in run_dir.rglob("*"))
    catalog = raman_paths.catalog.read_bytes()

    def no_pool(*_args: object, **_kwargs: object) -> None:
        pytest.fail("a dry run started a pool")

    monkeypatch.setattr(local, "ProcessPoolExecutor", no_pool)
    result = submit_run(SubmitRequest(run_dir.name, dry_run=True), load_settings())
    assert result.status is SubmitStatus.DRY_RUN
    assert result.record is not None
    assert (result.record.dry_run, result.record.attempt, result.record.task_indices) == (
        True,
        1,
        (0, 1),
    )
    assert sorted(p.relative_to(run_dir) for p in run_dir.rglob("*")) == before
    assert raman_paths.catalog.read_bytes() == catalog


# ---------------------------------------------------------------- poll


def test_poll_maps_markers_to_states(
    mini_store: Path, write_plan: WritePlan, raman_paths: PathsSettings
) -> None:
    run_dir = write_plan(mini_store, spectra_per_task=5)  # tasks 0 to 4
    run_task(TaskRequest(run_dir, 2))
    _lock(run_dir, 1, OTHER_HOST)
    _lock(run_dir, 3, OTHER_HOST, minutes_old=31)
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    dead = {"host": socket.gethostname(), "pid": child.pid, "started_at": "2026-11-03T11:00:00Z"}
    _lock(run_dir, 4, dead)
    backend = _backend(raman_paths)

    states = backend.poll(run_dir.name)
    assert {i: s.status for i, s in states.items()} == {
        0: TaskStatus.PENDING,
        1: TaskStatus.RUNNING,
        2: TaskStatus.DONE,
        3: TaskStatus.FAILED,
        4: TaskStatus.FAILED,
    }
    done = shards.read_done(shards.task_dir(run_dir, 2))
    assert done is not None
    assert (states[2].n_ok, states[2].n_failed, states[2].has_done_marker) == (
        done.n_ok,
        done.n_failed,
        True,
    )
    assert "elsewhere.example" in (states[3].error or "")
    assert "logs/task-3.log" in (states[3].error or "")
    assert str(child.pid) in (states[4].error or "")
    assert {s.attempts for s in states.values()} == {0}
    assert {s.slurm_job_id for s in states.values()} == {None}

    # Once submitted, a task that has not started is pending while its submission runs ...
    append_submission(run_dir, _record(run_dir.name, 1, (0, 1, 2, 3, 4)))
    task_lock.acquire_lock(run_dir / LOCAL_DIR, STALE)
    states = backend.poll(run_dir.name)
    assert states[0].status is TaskStatus.PENDING
    assert {s.attempts for s in states.values()} == {1}
    # ... and cancelled once the submission is gone, since nothing will start it.
    task_lock.release_lock(run_dir / LOCAL_DIR)
    states = backend.poll(run_dir.name)
    assert states[0].status is TaskStatus.CANCELLED
    assert "no longer running" in (states[0].error or "")


def test_stopped_task_awaiting_its_retry_is_pending(
    mini_store: Path, write_plan: WritePlan, raman_paths: PathsSettings
) -> None:
    run_dir = write_plan(mini_store)
    _lock(run_dir, 0, OTHER_HOST, minutes_old=31)  # left by attempt 1, started 11:00
    append_submission(run_dir, _record(run_dir.name, 1, (0, 1)))
    later = _record(run_dir.name, 2, (0,)).model_copy(
        update={"submitted_at": datetime(2026, 11, 3, 12, 0, tzinfo=UTC)}
    )
    append_submission(run_dir, later)
    task_lock.acquire_lock(run_dir / LOCAL_DIR, STALE)
    assert _backend(raman_paths).poll(run_dir.name)[0].status is TaskStatus.PENDING
    task_lock.release_lock(run_dir / LOCAL_DIR)
    assert _backend(raman_paths).poll(run_dir.name)[0].status is TaskStatus.FAILED


def test_poll_done_marker_wins(
    mini_store: Path, write_plan: WritePlan, raman_paths: PathsSettings
) -> None:
    run_dir = write_plan(mini_store)
    run_task(TaskRequest(run_dir, 0))
    _lock(run_dir, 0, OTHER_HOST)
    state = _backend(raman_paths).poll(run_dir.name)[0]
    assert state.status is TaskStatus.DONE
    assert state.has_done_marker


# ---------------------------------------------------------------- cancel


def test_cancel_terminates_running_tasks(
    mini_store: Path, write_plan: WritePlan, raman_paths: PathsSettings
) -> None:
    run_dir = write_plan(mini_store, spectra_per_task=5)
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        here = {"host": socket.gethostname(), "started_at": "2026-11-03T11:00:00Z"}
        _lock(run_dir, 0, here | {"pid": sleeper.pid})
        _lock(run_dir, 1, here | {"pid": os.getpid()})  # never signal ourselves
        _lock(run_dir, 2, OTHER_HOST)  # cannot be signalled from here
        _backend(raman_paths).cancel(run_dir.name)
        assert sleeper.wait(timeout=10) == -signal.SIGTERM
    finally:
        sleeper.kill()
        sleeper.wait()


# ---------------------------------------------------------------- the driver: failure, resubmission


def _submit(run_id: str, **request: bool) -> Any:
    return submit_run(SubmitRequest(run_id, **request), load_settings())


def test_failure_with_nothing_in_flight_marks_run_failed(
    mini_store: Path, write_plan: WritePlan, raman_paths: PathsSettings
) -> None:
    run_dir = write_plan(mini_store)
    _register(raman_paths, run_dir)
    (run_dir / "logs" / "task-1.log").mkdir(parents=True)  # task 1 cannot open its log
    result = _submit(run_dir.name)
    assert result.status is SubmitStatus.FINISHED
    assert result.run is not None
    assert result.run.status is RunStatus.FAILED
    assert set(result.failures) == {1}
    states = {t.task_index: t.state for t in result.tasks}
    assert states[0].status is TaskStatus.DONE
    assert states[1].status is TaskStatus.FAILED
    assert (states[0].attempts, states[1].attempts) == (1, 1)


def _held_elsewhere_then_released(raman_paths: PathsSettings, run_dir: Path) -> None:
    # Task 1 looks held by another host, so this submission skips it; then that lock goes stale.
    _register(raman_paths, run_dir)
    _lock(run_dir, 1, OTHER_HOST)
    assert _submit(run_dir.name).run.status is RunStatus.RUNNING
    _lock(run_dir, 1, OTHER_HOST, minutes_old=31)


def test_resubmit_only_tasks_without_done(
    mini_store: Path, write_plan: WritePlan, raman_paths: PathsSettings
) -> None:
    run_dir = write_plan(mini_store)
    _held_elsewhere_then_released(raman_paths, run_dir)
    done_0 = (shards.task_dir(run_dir, 0) / shards.DONE_FILE).read_bytes()
    result = _submit(run_dir.name, resubmit_failed=True)
    assert result.record is not None
    assert (result.record.attempt, result.record.task_indices) == (2, (1,))
    assert result.run is not None
    assert result.run.status is RunStatus.COMPLETE
    assert (shards.task_dir(run_dir, 0) / shards.DONE_FILE).read_bytes() == done_0
    assert {t.task_index: t.state.attempts for t in result.tasks} == {0: 1, 1: 2}


def test_submission_json_is_array_in_attempt_order(
    mini_store: Path, write_plan: WritePlan, raman_paths: PathsSettings
) -> None:
    run_dir = write_plan(mini_store)
    _held_elsewhere_then_released(raman_paths, run_dir)
    _submit(run_dir.name, resubmit_failed=True)
    stored = json.loads((run_dir / "submission.json").read_text(encoding="utf-8"))
    assert isinstance(stored, list)
    assert [(r["attempt"], r["task_indices"], r["dry_run"]) for r in stored] == [
        (1, [0, 1], False),
        (2, [1], False),
    ]
    assert list(read_submissions(run_dir)) == [SubmissionRecord.model_validate(r) for r in stored]


def test_resubmit_refuses_task_after_3_attempts(
    mini_store: Path, write_plan: WritePlan, raman_paths: PathsSettings
) -> None:
    run_dir = write_plan(mini_store)
    _register(raman_paths, run_dir)
    catalog = Catalog(raman_paths.catalog)
    for old, new in [
        (RunStatus.PLANNED, RunStatus.SUBMITTED),
        (RunStatus.SUBMITTED, RunStatus.RUNNING),
        (RunStatus.RUNNING, RunStatus.FAILED),
    ]:
        catalog.transition(run_dir.name, old, new)
    run_task(TaskRequest(run_dir, 0))
    for attempt in (1, 2, 3):
        tasks = (1,) if attempt > 1 else (0, 1)
        append_submission(run_dir, _record(run_dir.name, attempt, tasks))
    _lock(run_dir, 1, OTHER_HOST, minutes_old=31)
    with pytest.raises(MaxAttemptsError, match=r"task 1 .* 3 attempts"):
        _submit(run_dir.name, resubmit_failed=True)
    assert len(read_submissions(run_dir)) == 3
    run = catalog.get_run(run_dir.name)
    assert run is not None
    assert run.status is RunStatus.FAILED


def test_worker_runs_one_task_with_the_given_lock_limit(
    mini_store: Path, write_plan: WritePlan
) -> None:
    run_dir = write_plan(mini_store)
    _lock(run_dir, 0, OTHER_HOST, minutes_old=10)
    held = run_task_in_worker(run_dir, 0, lock_stale=STALE)
    assert held.holder == LockHolder.model_validate(OTHER_HOST)
    done = run_task_in_worker(run_dir, 0, lock_stale=timedelta(minutes=5))
    assert done.marker is not None


def test_resubmit_of_a_running_run_keeps_it_running(
    mini_store: Path, write_plan: WritePlan, raman_paths: PathsSettings
) -> None:
    run_dir = write_plan(mini_store, spectra_per_task=5)  # tasks 0 to 4
    _register(raman_paths, run_dir)
    _lock(run_dir, 0, OTHER_HOST)  # still running elsewhere
    _lock(run_dir, 1, OTHER_HOST)
    assert _submit(run_dir.name).run.status is RunStatus.RUNNING
    _lock(run_dir, 1, OTHER_HOST, minutes_old=31)  # task 1 stopped; task 0 has not

    preview = _submit(run_dir.name, resubmit_failed=True, dry_run=True)
    assert preview.status is SubmitStatus.DRY_RUN
    assert preview.record is not None
    assert (preview.record.dry_run, preview.record.attempt) == (True, 2)
    assert preview.record.task_indices == (1,)
    assert len(read_submissions(run_dir)) == 1

    result = _submit(run_dir.name, resubmit_failed=True)
    assert result.record is not None
    assert result.record.task_indices == (1,)
    assert result.run.status is RunStatus.RUNNING
    states = {t.task_index: t.state.status for t in result.tasks}
    assert states == {0: TaskStatus.RUNNING, **dict.fromkeys(range(1, 5), TaskStatus.DONE)}


def test_cancel_after_the_task_exits_is_harmless(
    mini_store: Path,
    write_plan: WritePlan,
    raman_paths: PathsSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = write_plan(mini_store)
    _lock(run_dir, 0, {"host": socket.gethostname(), "pid": 1, "started_at": "2026-11-03T11:00Z"})
    real_kill = os.kill
    sent: list[int] = []

    def kill(pid: int, sig: int) -> None:
        if sig == signal.SIGTERM:
            sent.append(pid)
            raise ProcessLookupError(pid)
        real_kill(pid, sig)

    monkeypatch.setattr(os, "kill", kill)
    _backend(raman_paths).cancel(run_dir.name, [0])
    assert sent == [1]


def test_unreadable_submission_list_is_an_error(
    mini_store: Path, write_plan: WritePlan, raman_paths: PathsSettings
) -> None:
    run_dir = write_plan(mini_store)
    (run_dir / "submission.json").write_text('{"attempt": 1}', encoding="utf-8")
    with pytest.raises(InvalidSubmissionsError, match=r"submission\.json"):
        _backend(raman_paths).poll(run_dir.name)


def test_reread_of_a_run_missing_from_the_catalog_is_an_error(
    mini_store: Path, write_plan: WritePlan, raman_paths: PathsSettings, tmp_path: Path
) -> None:
    run_dir = write_plan(mini_store)
    _register(raman_paths, run_dir)
    context = open_run(run_dir.name, load_settings())
    assert context.reread() == context.run
    elsewhere = dataclasses.replace(context, catalog=Catalog(tmp_path / "other.sqlite"))
    with pytest.raises(RunNotInCatalogError, match=run_dir.name):
        elsewhere.reread()
