"""The SQLite catalog (DESIGN.md §8.5) and the run state machine it enforces (§7.4).

Only the CLI driver writes the catalog; tasks never open it (§15 #5). It uses a rollback
journal, not WAL, because WAL needs shared memory that cluster network filesystems do not
provide. Every write is one ``BEGIN IMMEDIATE`` transaction with a 30 s busy timeout, so a
second driver waits instead of failing. Reading a catalog that does not exist returns nothing
and creates nothing; the first write creates the file and its parent directories.
"""

import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import JsonValue

from raman_bayes.domain.dataset import DatasetManifest
from raman_bayes.domain.errors import (
    IllegalTransitionError,
    StaleStatusError,
    UnknownRunError,
    UnknownTaskError,
    UnsupportedCatalogError,
)
from raman_bayes.domain.priors import PriorsSpec
from raman_bayes.domain.run import IN_FLIGHT, RunStatus, TaskSpec, TaskState, TaskStatus

__all__ = [
    "BUSY_TIMEOUT_S",
    "RUN_TRANSITIONS",
    "SCHEMA_VERSION",
    "Catalog",
    "DatasetRow",
    "RunRecord",
    "RunRow",
    "TaskRow",
]

SCHEMA_VERSION = 1
BUSY_TIMEOUT_S = 30.0

_S = RunStatus
RUN_TRANSITIONS: frozenset[tuple[RunStatus, RunStatus]] = frozenset(
    {
        (_S.PLANNED, _S.SUBMITTED),
        (_S.SUBMITTED, _S.RUNNING),
        (_S.RUNNING, _S.COMPLETE),
        (_S.SUBMITTED, _S.FAILED),
        (_S.RUNNING, _S.FAILED),
        (_S.FAILED, _S.SUBMITTED),
        (_S.COMPLETE, _S.COLLECTED),
        (_S.PLANNED, _S.CANCELLED),
        (_S.SUBMITTED, _S.CANCELLED),
        (_S.RUNNING, _S.CANCELLED),
        (_S.FAILED, _S.CANCELLED),
    }
)


def _one_of(enum: type[RunStatus] | type[TaskStatus]) -> str:
    return ", ".join(f"'{member.value}'" for member in enum)


_SCHEMA = (
    """CREATE TABLE dataset (
        dataset_id TEXT PRIMARY KEY,
        scan TEXT NOT NULL,
        sample TEXT,
        content_hash TEXT NOT NULL,
        n_spectra INTEGER NOT NULL,
        n_wn INTEGER NOT NULL,
        store_uri TEXT NOT NULL,
        manifest_json TEXT NOT NULL,
        created_at TEXT NOT NULL)""",
    """CREATE TABLE priors_spec (
        priors_hash TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        version TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('draft', 'approved')),
        yaml_text TEXT NOT NULL,
        created_at TEXT NOT NULL)""",
    f"""CREATE TABLE run (
        run_id TEXT PRIMARY KEY,
        dataset_id TEXT NOT NULL REFERENCES dataset (dataset_id),
        priors_hash TEXT NOT NULL REFERENCES priors_spec (priors_hash),
        engine TEXT NOT NULL,
        engine_version TEXT NOT NULL,
        config_hash TEXT NOT NULL,
        git_sha TEXT,
        backend TEXT NOT NULL,
        profile TEXT NOT NULL,
        seed INTEGER NOT NULL,
        status TEXT NOT NULL CHECK (status IN ({_one_of(RunStatus)})),
        created_at TEXT NOT NULL,
        completed_at TEXT,
        result_uri TEXT)""",
    f"""CREATE TABLE task (
        run_id TEXT NOT NULL REFERENCES run (run_id),
        task_index INTEGER NOT NULL,
        spec_start INTEGER NOT NULL,
        spec_stop INTEGER NOT NULL,
        status TEXT NOT NULL CHECK (status IN ({_one_of(TaskStatus)})),
        attempts INTEGER NOT NULL,
        slurm_job_id TEXT,
        n_ok INTEGER,
        n_failed INTEGER,
        error TEXT,
        PRIMARY KEY (run_id, task_index))""",
)


@dataclass(frozen=True)
class DatasetRow:
    """One ``dataset`` row, with ``manifest_json`` parsed back into the manifest."""

    dataset_id: str
    scan: str
    sample: str | None
    content_hash: str
    n_spectra: int
    n_wn: int
    store_uri: str
    manifest: DatasetManifest
    created_at: datetime

    def as_json(self) -> dict[str, JsonValue]:
        """Return the row as JSON values; the manifest is nested under ``manifest``."""
        row: dict[str, JsonValue] = {
            key: value for key, value in asdict(self).items() if key != "manifest"
        }
        row["created_at"] = self.created_at.isoformat()
        row["manifest"] = self.manifest.model_dump(mode="json")
        return row


@dataclass(frozen=True)
class RunRecord:
    """What ``plan`` records about a new run; the catalog adds status and completion."""

    run_id: str
    dataset_id: str
    priors_hash: str
    engine: str
    engine_version: str
    config_hash: str
    git_sha: str | None
    backend: str
    profile: str
    seed: int
    created_at: datetime


@dataclass(frozen=True)
class RunRow(RunRecord):
    """One ``run`` row."""

    status: RunStatus
    completed_at: datetime | None
    result_uri: str | None

    def as_json(self) -> dict[str, JsonValue]:
        """Return the row as JSON values (enum values, ISO 8601 times)."""
        row: dict[str, JsonValue] = asdict(self)
        row["status"] = self.status.value
        row["created_at"] = self.created_at.isoformat()
        row["completed_at"] = self.completed_at.isoformat() if self.completed_at else None
        return row


@dataclass(frozen=True)
class TaskRow:
    """One ``task`` row: the plan's range and the last known state."""

    task_index: int
    spec_start: int
    spec_stop: int
    state: TaskState

    def as_json(self) -> dict[str, JsonValue]:
        """Return the §8.5 task columns other than ``run_id`` as JSON values."""
        state = self.state.model_dump(mode="json", exclude={"has_done_marker"})
        return {
            "task_index": self.task_index,
            "spec_start": self.spec_start,
            "spec_stop": self.spec_stop,
            **state,
        }


class Catalog:
    """The catalog file at ``path``; each method opens and closes its own connection."""

    def __init__(self, path: Path) -> None:
        self.path = path

    # ------------------------------------------------------------ reads

    def get_dataset(self, dataset_id: str) -> DatasetRow | None:
        """Return the dataset row, or None when it is not registered."""
        rows = self._read("SELECT * FROM dataset WHERE dataset_id = ?", (dataset_id,))
        return _dataset_row(rows[0]) if rows else None

    def list_datasets(self) -> list[DatasetRow]:
        """Return every dataset, oldest first."""
        return [_dataset_row(r) for r in self._read("SELECT * FROM dataset ORDER BY created_at")]

    def get_run(self, run_id: str) -> RunRow | None:
        """Return the run row, or None when there is no such run."""
        rows = self._read("SELECT * FROM run WHERE run_id = ?", (run_id,))
        return _run_row(rows[0]) if rows else None

    def list_runs(self) -> list[RunRow]:
        """Return every run, oldest first."""
        return [_run_row(r) for r in self._read("SELECT * FROM run ORDER BY created_at, run_id")]

    def tasks(self, run_id: str) -> list[TaskRow]:
        """Return the run's tasks in index order (empty for an unknown run)."""
        query = "SELECT * FROM task WHERE run_id = ? ORDER BY task_index"
        return [_task_row(r) for r in self._read(query, (run_id,))]

    # ------------------------------------------------------------ writes

    def add_dataset(self, manifest: DatasetManifest, store_uri: Path) -> bool:
        """Register a dataset; return False, changing nothing, if its id is already there."""
        values = (
            manifest.dataset_id,
            manifest.scan,
            manifest.sample,
            manifest.content_hash,
            manifest.n_spectra,
            manifest.n_wn,
            str(store_uri),
            manifest.model_dump_json(),
            manifest.created_at.isoformat(),
        )
        with self._write() as conn:
            cursor = conn.execute(
                "INSERT INTO dataset VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (dataset_id) DO NOTHING",
                values,
            )
            return cursor.rowcount == 1

    def add_priors_spec(
        self, spec: PriorsSpec, priors_hash: str, yaml_text: str, created_at: datetime
    ) -> bool:
        """Register a recipe by hash; return False, changing nothing, if it is already there."""
        meta = spec.spec
        values = (
            priors_hash,
            meta.name,
            meta.version,
            meta.status,
            yaml_text,
            created_at.isoformat(),
        )
        with self._write() as conn:
            cursor = conn.execute(
                "INSERT INTO priors_spec VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (priors_hash) DO NOTHING",
                values,
            )
            return cursor.rowcount == 1

    def create_run(self, record: RunRecord, tasks: Sequence[TaskSpec]) -> None:
        """Insert the run as ``planned`` and its tasks as ``pending`` with 0 attempts, atomically.

        Raises sqlite3.IntegrityError, leaving the catalog unchanged, for a duplicate run or task
        or a dataset or recipe that is not registered.
        """
        run = asdict(record) | {"status": RunStatus.PLANNED.value}
        run["created_at"] = record.created_at.isoformat()
        with self._write() as conn:
            conn.execute(
                "INSERT INTO run (run_id, dataset_id, priors_hash, engine, engine_version, "
                "config_hash, git_sha, backend, profile, seed, created_at, status) VALUES "
                "(:run_id, :dataset_id, :priors_hash, :engine, :engine_version, :config_hash, "
                ":git_sha, :backend, :profile, :seed, :created_at, :status)",
                run,
            )
            conn.executemany(
                "INSERT INTO task (run_id, task_index, spec_start, spec_stop, status, attempts) "
                "VALUES (?, ?, ?, ?, ?, 0)",
                [
                    (
                        record.run_id,
                        t.task_index,
                        t.spec_start,
                        t.spec_stop,
                        TaskStatus.PENDING.value,
                    )
                    for t in tasks
                ],
            )

    def transition(
        self, run_id: str, expected: RunStatus, new: RunStatus, *, now: datetime | None = None
    ) -> None:
        """Move the run from ``expected`` to ``new``; ``now`` stamps ``completed_at`` on complete.

        Raises IllegalTransitionError for a change §7.4 does not allow, UnknownRunError for an
        unknown run and StaleStatusError when the run is no longer ``expected``.
        """
        if (expected, new) not in RUN_TRANSITIONS:
            raise IllegalTransitionError(run_id, expected.value, new.value)
        with self._write() as conn:
            current = _status(conn, run_id)
            if current is not expected:
                raise StaleStatusError(run_id, expected.value, current.value)
            _set_status(conn, run_id, new, now or datetime.now(UTC))

    def mark_collected(self, run_id: str, result_uri: str, *, now: datetime | None = None) -> None:
        """Move a complete run to collected and record where its results are, atomically.

        Raises UnknownRunError, or StaleStatusError when the run is not complete.
        """
        with self._write() as conn:
            current = _status(conn, run_id)
            if current is not RunStatus.COMPLETE:
                raise StaleStatusError(run_id, RunStatus.COMPLETE.value, current.value)
            _set_status(conn, run_id, RunStatus.COLLECTED, now or datetime.now(UTC))
            conn.execute("UPDATE run SET result_uri = ? WHERE run_id = ?", (result_uri, run_id))

    def update_task(self, run_id: str, task_index: int, state: TaskState) -> None:
        """Store ``state`` as the task's last known state; raises UnknownTaskError if absent."""
        with self._write() as conn:
            _update_task(conn, run_id, task_index, state)

    def refresh(
        self, run_id: str, states: Mapping[int, TaskState], *, now: datetime | None = None
    ) -> tuple[RunStatus, ...]:
        """Store polled task states, then apply every run transition they imply, in order.

        Returns the statuses entered (``(running, complete)`` when a submitted run turns out
        finished). Raises UnknownRunError or UnknownTaskError; the catalog is then unchanged.
        """
        with self._write() as conn:
            current = _status(conn, run_id)
            for task_index, state in states.items():
                _update_task(conn, run_id, task_index, state)
            rows = conn.execute("SELECT status FROM task WHERE run_id = ?", (run_id,))
            task_statuses = [TaskStatus(row["status"]) for row in rows]
            entered: list[RunStatus] = []
            while (new := _implied(current, task_statuses)) is not None:
                _set_status(conn, run_id, new, now or datetime.now(UTC))
                entered.append(new)
                current = new
            return tuple(entered)

    # ------------------------------------------------------------ connections

    def _read(self, query: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        if not self.path.exists():
            return []
        conn = sqlite3.connect(f"{self.path.resolve().as_uri()}?mode=ro", uri=True)
        try:
            conn.row_factory = sqlite3.Row
            if _check_version(conn, self.path) == 0:
                return []
            return conn.execute(query, params).fetchall()
        finally:
            conn.close()

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=BUSY_TIMEOUT_S, isolation_level=None)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode = DELETE")
            conn.execute("PRAGMA foreign_keys = ON")
            with conn:  # commits on success, rolls back if the body raises
                conn.execute("BEGIN IMMEDIATE")
                if _check_version(conn, self.path) == 0:
                    for statement in _SCHEMA:
                        conn.execute(statement)
                    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                yield conn
        finally:
            conn.close()


def _check_version(conn: sqlite3.Connection, path: Path) -> int:
    # 0 is a file never written by raman-bayes; anything else must be this version.
    version: int = conn.execute("PRAGMA user_version").fetchone()[0]
    if version not in (0, SCHEMA_VERSION):
        raise UnsupportedCatalogError(path, version, SCHEMA_VERSION)
    return version


def _status(conn: sqlite3.Connection, run_id: str) -> RunStatus:
    row = conn.execute("SELECT status FROM run WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        raise UnknownRunError(run_id)
    return RunStatus(row["status"])


def _set_status(conn: sqlite3.Connection, run_id: str, new: RunStatus, now: datetime) -> None:
    completed_at = now.isoformat() if new is RunStatus.COMPLETE else None
    conn.execute(
        "UPDATE run SET status = ?, completed_at = coalesce(?, completed_at) WHERE run_id = ?",
        (new.value, completed_at, run_id),
    )


def _update_task(conn: sqlite3.Connection, run_id: str, task_index: int, s: TaskState) -> None:
    cursor = conn.execute(
        "UPDATE task SET status = ?, attempts = ?, slurm_job_id = ?, n_ok = ?, n_failed = ?, "
        "error = ? WHERE run_id = ? AND task_index = ?",
        (
            s.status.value,
            s.attempts,
            s.slurm_job_id,
            s.n_ok,
            s.n_failed,
            s.error,
            run_id,
            task_index,
        ),
    )
    if cursor.rowcount != 1:
        raise UnknownTaskError(run_id, task_index)


def _implied(current: RunStatus, tasks: Sequence[TaskStatus]) -> RunStatus | None:
    # The §7.4 transitions a poll can trigger, checked in the order the table lists them.
    if current is RunStatus.SUBMITTED and any(
        s in (TaskStatus.RUNNING, TaskStatus.DONE) for s in tasks
    ):
        return RunStatus.RUNNING
    if current is RunStatus.RUNNING and all(s is TaskStatus.DONE for s in tasks):
        return RunStatus.COMPLETE
    stopped = not any(s in IN_FLIGHT for s in tasks)
    unfinished = not all(s is TaskStatus.DONE for s in tasks)
    if current in (RunStatus.SUBMITTED, RunStatus.RUNNING) and stopped and unfinished:
        return RunStatus.FAILED
    return None


def _dataset_row(row: sqlite3.Row) -> DatasetRow:
    return DatasetRow(
        dataset_id=row["dataset_id"],
        scan=row["scan"],
        sample=row["sample"],
        content_hash=row["content_hash"],
        n_spectra=row["n_spectra"],
        n_wn=row["n_wn"],
        store_uri=row["store_uri"],
        manifest=DatasetManifest.model_validate_json(row["manifest_json"]),
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def _run_row(row: sqlite3.Row) -> RunRow:
    fields = dict(row)
    fields["status"] = RunStatus(fields["status"])
    fields["created_at"] = datetime.fromisoformat(fields["created_at"])
    if fields["completed_at"] is not None:
        fields["completed_at"] = datetime.fromisoformat(fields["completed_at"])
    return RunRow(**fields)


def _task_row(row: sqlite3.Row) -> TaskRow:
    status = TaskStatus(row["status"])
    state = TaskState(
        status=status,
        attempts=row["attempts"],
        slurm_job_id=row["slurm_job_id"],
        has_done_marker=status is TaskStatus.DONE,
        n_ok=row["n_ok"],
        n_failed=row["n_failed"],
        error=row["error"],
    )
    return TaskRow(row["task_index"], row["spec_start"], row["spec_stop"], state)
