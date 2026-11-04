"""LocalBackend (DESIGN.md §7.3): a run's tasks in a pool of spawned processes on this machine.

``submit`` starts the pool and returns its record at once; ``wait`` blocks until every task
has stopped. Each task runs in a process of its own (``max_tasks_per_child=1``), so a task's
lock names a process that exits when the task does, and a lock whose process is gone means
the task stopped early. While a submission runs, its process holds ``runs/<id>/local/_LOCK``
with the same heartbeat as a task lock; ``poll`` uses it to tell a task still waiting for a
worker from one whose submission ended before starting it.

The backend runs whatever ``runner(run_dir, task_index)`` does, which must be picklable; the
app passes run-task. It writes only that lock: the driver records submissions and the catalog.
"""

import logging
import multiprocessing
import os
import signal
import socket
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from raman_bayes.domain.markers import LockHolder
from raman_bayes.domain.run import RunPlan, SubmissionRecord, TaskState, TaskStatus
from raman_bayes.storage.run_files import read_plan, read_submissions
from raman_bayes.storage.shards import read_done, task_dir
from raman_bayes.storage.task_lock import (
    LockState,
    acquire_lock,
    heartbeat,
    read_lock,
    release_lock,
)

__all__ = ["LOCAL_DIR", "LocalBackend", "TaskRunner"]

LOCAL_DIR = "local"
TaskRunner = Callable[[Path, int], object]
_HEARTBEATS_PER_STALE = 4
_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Running:
    """A submission whose pool is running: what to wait on, and what to close after."""

    futures: Mapping[int, Future[object]]
    cleanup: ExitStack


class LocalBackend:
    """Runs tasks with up to ``workers`` processes; a lock ``lock_stale`` old is stale."""

    name = "local"

    def __init__(
        self, runs_root: Path, workers: int, lock_stale: timedelta, runner: TaskRunner
    ) -> None:
        self.runs_root = runs_root
        self.workers = workers
        self.lock_stale = lock_stale
        self.runner = runner
        self._running: dict[str, _Running] = {}

    def submit(self, plan: RunPlan, *, dry_run: bool = False) -> SubmissionRecord:
        """Start every task of ``plan``; with ``dry_run``, return the record and start nothing.

        Raises TaskLockedError when another process is running a submission of this run.
        """
        return self._start(plan.run_id, tuple(t.task_index for t in plan.tasks), dry_run)

    def resubmit(self, run_id: str, tasks: Sequence[int]) -> SubmissionRecord:
        """Start ``tasks`` of ``run_id`` again, as the next attempt."""
        return self._start(run_id, tuple(sorted(tasks)), dry_run=False)

    def wait(self, run_id: str) -> dict[int, str]:
        """Block until the submission of ``run_id`` has stopped; return the tasks that raised.

        The mapping gives each such task's error as ``<Type>: <message>``. The pool and the
        submission lock are released even if waiting is interrupted.
        """
        running = self._running.pop(run_id)
        failures: dict[int, str] = {}
        with running.cleanup:
            for index, future in running.futures.items():
                error = future.exception()
                if error is not None:
                    failures[index] = f"{type(error).__name__}: {error}"
                    _log.warning("run %s, task %d stopped: %s", run_id, index, failures[index])
        return failures

    def poll(self, run_id: str) -> Mapping[int, TaskState]:
        """Return each task's state from its markers, attempts from ``submission.json``.

        A done marker wins. A fresh task lock is running and a stale one failed, unless the
        task is queued again in a running submission. A task without a lock is pending while
        its submission runs (or before any) and cancelled once that submission has ended.
        """
        run_dir = self.runs_root / run_id
        records = read_submissions(run_dir)
        attempts = Counter(i for r in records for i in r.task_indices)
        last = {i: r.submitted_at for r in records for i in r.task_indices}
        submitter = read_lock(run_dir / LOCAL_DIR, self.lock_stale)
        live = submitter is not None and submitter.fresh
        return {
            t.task_index: self._state(run_dir, t.task_index, attempts[t.task_index], live, last)
            for t in read_plan(run_dir).tasks
        }

    def cancel(self, run_id: str, tasks: Sequence[int] | None = None) -> None:
        """Send SIGTERM to the live processes holding ``tasks`` (default: all) on this host.

        A process on another host cannot be signalled from here and is reported in the log;
        this process is never signalled.
        """
        run_dir = self.runs_root / run_id
        indices = tasks if tasks is not None else [t.task_index for t in read_plan(run_dir).tasks]
        for index in indices:
            lock = read_lock(task_dir(run_dir, index), self.lock_stale)
            if lock is not None and lock.fresh and lock.holder is not None:
                _terminate(run_id, index, lock.holder)

    def _start(self, run_id: str, tasks: tuple[int, ...], dry_run: bool) -> SubmissionRecord:
        run_dir = self.runs_root / run_id
        record = SubmissionRecord(
            run_id=run_id,
            attempt=len(read_submissions(run_dir)) + 1,
            backend="local",
            submitted_at=datetime.now(UTC),
            host=socket.gethostname(),
            dry_run=dry_run,
            task_indices=tasks,
        )
        if dry_run:
            return record
        with ExitStack() as cleanup:
            lock_dir = run_dir / LOCAL_DIR
            acquire_lock(lock_dir, self.lock_stale)
            cleanup.callback(release_lock, lock_dir)
            cleanup.enter_context(heartbeat(lock_dir, self.lock_stale / _HEARTBEATS_PER_STALE))
            pool = ProcessPoolExecutor(
                max_workers=min(self.workers, len(tasks)),
                mp_context=multiprocessing.get_context("spawn"),
                max_tasks_per_child=1,
            )
            cleanup.callback(pool.shutdown, wait=True, cancel_futures=True)
            futures = {i: pool.submit(self.runner, run_dir, i) for i in tasks}
            self._running[run_id] = _Running(futures, cleanup.pop_all())
        return record

    def _state(
        self,
        run_dir: Path,
        index: int,
        attempts: int,
        live: bool,
        last: Mapping[int, datetime],
    ) -> TaskState:
        folder = task_dir(run_dir, index)
        done = read_done(folder)
        if done is not None:
            return TaskState(
                status=TaskStatus.DONE,
                attempts=attempts,
                has_done_marker=True,
                n_ok=done.n_ok,
                n_failed=done.n_failed,
            )
        lock = read_lock(folder, self.lock_stale)
        if lock is not None and (lock.fresh or not _awaiting_retry(lock, index, live, last)):
            status = TaskStatus.RUNNING if lock.fresh else TaskStatus.FAILED
            error = None if lock.fresh else _stopped(lock, index)
            return TaskState(status=status, attempts=attempts, has_done_marker=False, error=error)
        if attempts and not live:
            error = "queued, but the submission that would run it is no longer running"
            return TaskState(
                status=TaskStatus.CANCELLED, attempts=attempts, has_done_marker=False, error=error
            )
        return TaskState(status=TaskStatus.PENDING, attempts=attempts, has_done_marker=False)


def _awaiting_retry(lock: LockState, index: int, live: bool, last: Mapping[int, datetime]) -> bool:
    # A stale lock from an attempt older than the submission now running: the task is queued.
    if not live or lock.holder is None or index not in last:
        return False
    return lock.holder.started_at < last[index]


def _stopped(lock: LockState, index: int) -> str:
    holder = lock.holder
    who = (
        f"pid {holder.pid} on {holder.host}, started {holder.started_at.isoformat()}"
        if holder is not None
        else "its lock was never completed"
    )
    return f"run-task stopped before finishing ({who}); see logs/task-{index}.log"


def _terminate(run_id: str, index: int, holder: LockHolder) -> None:
    if holder.host != socket.gethostname():
        _log.warning(
            "run %s, task %d runs on %s; cancel it there (pid %d)",
            run_id,
            index,
            holder.host,
            holder.pid,
        )
        return
    if holder.pid == os.getpid():
        return
    try:
        os.kill(holder.pid, signal.SIGTERM)
    except ProcessLookupError:
        return  # it finished between reading the lock and the signal
