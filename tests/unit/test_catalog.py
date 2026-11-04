"""SQLite catalog (DESIGN.md §8.5) and the run state machine it enforces (§7.4)."""

import dataclasses
import itertools
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from raman_bayes.domain.dataset import DatasetManifest
from raman_bayes.domain.errors import (
    IllegalTransitionError,
    StaleStatusError,
    UnknownRunError,
    UnsupportedCatalogError,
)
from raman_bayes.domain.priors import PriorsSpec
from raman_bayes.domain.run import RunStatus, TaskSpec, TaskState, TaskStatus
from raman_bayes.storage.catalog import Catalog, RunRecord

S = RunStatus
NOW = datetime(2026, 11, 2, 12, 0, tzinfo=UTC)
FIXTURE = Path(__file__).parents[1] / "fixtures" / "priors" / "design_example.yaml"
PRIORS_TEXT = FIXTURE.read_text(encoding="utf-8")
PRIORS = PriorsSpec.model_validate(yaml.safe_load(PRIORS_TEXT))

# §8.5 columns in order, copied by hand; never generated from the code under test.
TABLES = {
    "dataset": [
        "dataset_id",
        "scan",
        "sample",
        "content_hash",
        "n_spectra",
        "n_wn",
        "store_uri",
        "manifest_json",
        "created_at",
    ],
    "priors_spec": ["priors_hash", "name", "version", "status", "yaml_text", "created_at"],
    "run": [
        "run_id",
        "dataset_id",
        "priors_hash",
        "engine",
        "engine_version",
        "config_hash",
        "git_sha",
        "backend",
        "profile",
        "seed",
        "status",
        "created_at",
        "completed_at",
        "result_uri",
    ],
    "task": [
        "run_id",
        "task_index",
        "spec_start",
        "spec_stop",
        "status",
        "attempts",
        "slurm_job_id",
        "n_ok",
        "n_failed",
        "error",
    ],
}
# The §7.4 transition table, copied by hand.
LEGAL = {
    (S.PLANNED, S.SUBMITTED),
    (S.SUBMITTED, S.RUNNING),
    (S.RUNNING, S.COMPLETE),
    (S.SUBMITTED, S.FAILED),
    (S.RUNNING, S.FAILED),
    (S.FAILED, S.SUBMITTED),
    (S.COMPLETE, S.COLLECTED),
    (S.PLANNED, S.CANCELLED),
    (S.SUBMITTED, S.CANCELLED),
    (S.RUNNING, S.CANCELLED),
    (S.FAILED, S.CANCELLED),
}
ILLEGAL = sorted(set(itertools.product(RunStatus, repeat=2)) - LEGAL)
# A path through legal transitions from planned to each status.
PATH_TO = {
    S.PLANNED: [],
    S.SUBMITTED: [S.SUBMITTED],
    S.RUNNING: [S.SUBMITTED, S.RUNNING],
    S.COMPLETE: [S.SUBMITTED, S.RUNNING, S.COMPLETE],
    S.COLLECTED: [S.SUBMITTED, S.RUNNING, S.COMPLETE, S.COLLECTED],
    S.FAILED: [S.SUBMITTED, S.FAILED],
    S.CANCELLED: [S.CANCELLED],
}


def _manifest(dataset_id: str = "NMC811_v0-0123456789", **overrides: Any) -> DatasetManifest:
    point = {"start_um": 0.0, "stop_um": 0.0, "step_um": 0.0}
    x = {"start_um": 0.0, "stop_um": 4.0, "step_um": 1.0}
    fields: dict[str, Any] = {
        "scan": "NMC811_v0",
        "acquired_at": datetime(2026, 5, 28, 15, 47, 23),
        "duration_s": 2090.0,
        "instrument": {"system": "s", "detector": "d", "positioner": "p", "raw_stage": "A\n"},
        "acq": {
            "exposure_s": 0.2,
            "n_accum": 5,
            "ccd_temp_c": -72.0,
            "central_pixel": 505,
            "head_model": "DU920P_BEX2D",
        },
        "n_wn": 1024,
        "n_series": 1,
        "grid": {"nx": 5, "ny": 1, "nz": 1, "x": x, "y": point, "z": point},
        "dataset_id": dataset_id,
        "sample": "S1",
        "n_spectra": 5,
        "content_hash": "ab" * 32,
        "created_at": NOW,
    }
    return DatasetManifest.model_validate(fields | overrides)


def _run_record(run_id: str = "0190a1b2-c3d4-7e5f-8a9b-0c1d2e3f4a5b") -> RunRecord:
    return RunRecord(
        run_id=run_id,
        dataset_id="NMC811_v0-0123456789",
        priors_hash="cd" * 32,
        engine="mock",
        engine_version="0.1.0",
        config_hash="ef" * 32,
        git_sha="abc1234",
        backend="local",
        profile="local",
        seed=12345,
        created_at=NOW,
    )


TASKS = (
    TaskSpec(task_index=0, spec_start=0, spec_stop=3),
    TaskSpec(task_index=1, spec_start=3, spec_stop=5),
)


@pytest.fixture
def catalog(tmp_path: Path) -> Catalog:
    cat = Catalog(tmp_path / "catalog.sqlite")
    cat.add_dataset(_manifest(), tmp_path / "NMC811_v0-0123456789.zarr")
    cat.add_priors_spec(PRIORS, "cd" * 32, PRIORS_TEXT, NOW)
    return cat


@pytest.fixture
def run_id(catalog: Catalog) -> str:
    record = _run_record()
    catalog.create_run(record, TASKS)
    return record.run_id


def _sql(catalog: Catalog, query: str) -> list[tuple[Any, ...]]:
    with closing(sqlite3.connect(catalog.path)) as conn, conn:
        rows: list[tuple[Any, ...]] = conn.execute(query).fetchall()
    return rows


def _walk(catalog: Catalog, run_id: str, target: RunStatus) -> None:
    current = S.PLANNED
    for step in PATH_TO[target]:
        catalog.transition(run_id, current, step)
        current = step


def _state(status: TaskStatus, **fields: Any) -> TaskState:
    done = status is TaskStatus.DONE
    defaults: dict[str, Any] = {"attempts": 1, "has_done_marker": done}
    if done:
        defaults |= {"n_ok": 3, "n_failed": 0}
    return TaskState.model_validate({"status": status, **defaults, **fields})


# ---------------------------------------------------------------- schema and pragmas


def test_tables_and_columns_match_section_8_5(catalog: Catalog) -> None:
    for table, columns in TABLES.items():
        found = [row[1] for row in _sql(catalog, f"PRAGMA table_info({table})")]
        assert found == columns, table
    names = {row[0] for row in _sql(catalog, "SELECT name FROM sqlite_master WHERE type='table'")}
    assert names == set(TABLES)
    pk = [row[1] for row in _sql(catalog, "PRAGMA table_info(task)") if row[5]]
    assert pk == ["run_id", "task_index"]


def test_rollback_journal_not_wal(catalog: Catalog) -> None:
    assert _sql(catalog, "PRAGMA journal_mode") == [("delete",)]
    assert not Path(f"{catalog.path}-wal").exists()


def test_user_version_is_1(catalog: Catalog) -> None:
    assert _sql(catalog, "PRAGMA user_version") == [(1,)]


def test_catalog_with_other_user_version_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "old.sqlite"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("PRAGMA user_version = 7")
    with pytest.raises(UnsupportedCatalogError, match="version 7"):
        Catalog(path).list_datasets()


# ---------------------------------------------------------------- state machine


@pytest.mark.parametrize(("source", "target"), sorted(LEGAL))
def test_every_legal_transition_succeeds(
    catalog: Catalog, run_id: str, source: RunStatus, target: RunStatus
) -> None:
    _walk(catalog, run_id, source)
    catalog.transition(run_id, source, target)
    run = catalog.get_run(run_id)
    assert run is not None
    assert run.status is target


@pytest.mark.parametrize(("source", "target"), ILLEGAL)
def test_every_illegal_transition_raises(
    catalog: Catalog, run_id: str, source: RunStatus, target: RunStatus
) -> None:
    assert len(ILLEGAL) == 38
    _walk(catalog, run_id, source)
    with pytest.raises(IllegalTransitionError) as exc:
        catalog.transition(run_id, source, target)
    assert source.value in str(exc.value)
    assert target.value in str(exc.value)
    run = catalog.get_run(run_id)
    assert run is not None
    assert run.status is source


def test_transition_from_stale_expected_status_raises(catalog: Catalog, run_id: str) -> None:
    catalog.transition(run_id, S.PLANNED, S.SUBMITTED)
    with pytest.raises(StaleStatusError, match="submitted"):
        catalog.transition(run_id, S.PLANNED, S.CANCELLED)
    run = catalog.get_run(run_id)
    assert run is not None
    assert run.status is S.SUBMITTED


def test_refresh_applies_transitions_in_order(catalog: Catalog, run_id: str) -> None:
    pending = {0: _state(TaskStatus.PENDING), 1: _state(TaskStatus.PENDING)}
    assert catalog.refresh(run_id, pending, now=NOW) == ()  # planned: nothing to move
    catalog.transition(run_id, S.PLANNED, S.SUBMITTED)
    assert catalog.refresh(run_id, pending, now=NOW) == ()

    done = {0: _state(TaskStatus.DONE), 1: _state(TaskStatus.DONE)}
    assert catalog.refresh(run_id, done, now=NOW) == (S.RUNNING, S.COMPLETE)
    run = catalog.get_run(run_id)
    assert run is not None
    assert run.status is S.COMPLETE
    assert run.completed_at == NOW


def test_refresh_marks_failed_only_with_nothing_in_flight(catalog: Catalog) -> None:
    for i, (states, expected) in enumerate(
        [
            ({0: _state(TaskStatus.FAILED), 1: _state(TaskStatus.RUNNING)}, (S.RUNNING,)),
            ({0: _state(TaskStatus.FAILED), 1: _state(TaskStatus.PENDING)}, ()),
            ({0: _state(TaskStatus.FAILED), 1: _state(TaskStatus.DONE)}, (S.RUNNING, S.FAILED)),
            ({0: _state(TaskStatus.TIMEOUT), 1: _state(TaskStatus.FAILED)}, (S.FAILED,)),
        ]
    ):
        record = _run_record(f"0190a1b2-c3d4-7e5f-8a9b-0c1d2e3f4a5{i}")
        catalog.create_run(record, TASKS)
        catalog.transition(record.run_id, S.PLANNED, S.SUBMITTED)
        assert catalog.refresh(record.run_id, states, now=NOW) == expected


def test_status_columns_hold_enum_values(catalog: Catalog, run_id: str) -> None:
    catalog.transition(run_id, S.PLANNED, S.SUBMITTED)
    assert _sql(catalog, "SELECT status FROM run") == [("submitted",)]
    assert _sql(catalog, "SELECT status FROM task ORDER BY task_index") == [
        ("pending",),
        ("pending",),
    ]
    with pytest.raises(sqlite3.IntegrityError):
        _sql(catalog, "UPDATE run SET status = 'finished'")
    with pytest.raises(sqlite3.IntegrityError):
        _sql(catalog, "UPDATE task SET status = 'DONE'")


# ---------------------------------------------------------------- rows


def test_dataset_insert_and_get_round_trip_manifest(catalog: Catalog, tmp_path: Path) -> None:
    row = catalog.get_dataset("NMC811_v0-0123456789")
    assert row is not None
    assert row.manifest == _manifest()
    assert row.store_uri == str(tmp_path / "NMC811_v0-0123456789.zarr")
    assert (row.scan, row.sample, row.n_spectra, row.n_wn) == ("NMC811_v0", "S1", 5, 1024)
    assert row.content_hash == "ab" * 32
    assert row.created_at == NOW
    assert catalog.get_dataset("absent") is None
    assert [d.dataset_id for d in catalog.list_datasets()] == ["NMC811_v0-0123456789"]


def test_duplicate_dataset_insert_reports_exists(catalog: Catalog, tmp_path: Path) -> None:
    assert catalog.add_dataset(_manifest(), tmp_path / "elsewhere.zarr") is False
    row = catalog.get_dataset("NMC811_v0-0123456789")
    assert row is not None
    assert row.store_uri == str(tmp_path / "NMC811_v0-0123456789.zarr")
    assert catalog.add_dataset(_manifest("NMC811_v0-9999999999"), tmp_path / "b.zarr") is True


def test_priors_spec_insert_idempotent_by_hash(catalog: Catalog) -> None:
    assert catalog.add_priors_spec(PRIORS, "cd" * 32, PRIORS_TEXT, NOW) is False
    assert catalog.add_priors_spec(PRIORS, "ee" * 32, PRIORS_TEXT, NOW) is True
    rows = _sql(catalog, "SELECT priors_hash, name, version, status FROM priors_spec ORDER BY 1")
    assert rows == [
        ("cd" * 32, "nmc_commercial", "1.0.0", "approved"),
        ("ee" * 32, "nmc_commercial", "1.0.0", "approved"),
    ]


def test_run_and_task_rows_created_atomically(catalog: Catalog) -> None:
    record = _run_record()
    duplicate = (*TASKS, TaskSpec(task_index=1, spec_start=5, spec_stop=6))
    with pytest.raises(sqlite3.IntegrityError):
        catalog.create_run(record, duplicate)
    assert catalog.get_run(record.run_id) is None
    assert _sql(catalog, "SELECT count(*) FROM task") == [(0,)]

    catalog.create_run(record, TASKS)
    run = catalog.get_run(record.run_id)
    assert run is not None
    assert run.status is S.PLANNED
    assert run.completed_at is None
    tasks = catalog.tasks(record.run_id)
    assert [(t.task_index, t.spec_start, t.spec_stop) for t in tasks] == [(0, 0, 3), (1, 3, 5)]
    assert {(t.state.status, t.state.attempts) for t in tasks} == {(TaskStatus.PENDING, 0)}

    unknown_dataset = dataclasses.replace(
        _run_record("0190a1b2-c3d4-7e5f-8a9b-0c1d2e3f4a5c"), dataset_id="x"
    )
    with pytest.raises(sqlite3.IntegrityError):
        catalog.create_run(unknown_dataset, TASKS)


def test_task_row_updated_from_task_state(catalog: Catalog, run_id: str) -> None:
    state = _state(TaskStatus.DONE, attempts=2, slurm_job_id="123_1", n_ok=3, n_failed=0)
    catalog.update_task(run_id, 1, state)
    tasks = catalog.tasks(run_id)
    assert tasks[1].state == state
    assert tasks[0].state.status is TaskStatus.PENDING
    failed = _state(TaskStatus.FAILED, attempts=2, error="exit code 4")
    catalog.update_task(run_id, 0, failed)
    assert catalog.tasks(run_id)[0].state.error == "exit code 4"


def test_catalog_created_with_parent_dirs_on_first_write(tmp_path: Path) -> None:
    path = tmp_path / "a" / "b" / "catalog.sqlite"
    cat = Catalog(path)
    assert cat.list_datasets() == []
    assert cat.list_runs() == []
    assert cat.get_dataset("x") is None
    assert not path.parent.exists()
    cat.add_dataset(_manifest(), tmp_path / "d.zarr")
    assert path.is_file()


def test_mark_collected_records_result_uri_and_needs_complete(
    catalog: Catalog, run_id: str
) -> None:
    _walk(catalog, run_id, S.RUNNING)
    with pytest.raises(StaleStatusError, match="expected status complete, but it is running"):
        catalog.mark_collected(run_id, "/runs/r/results")
    catalog.transition(run_id, S.RUNNING, S.COMPLETE)
    catalog.mark_collected(run_id, "/archive/r/results")
    run = catalog.get_run(run_id)
    assert run is not None
    assert (run.status, run.result_uri) == (S.COLLECTED, "/archive/r/results")


def test_require_run_raises_for_an_unknown_run(catalog: Catalog, run_id: str) -> None:
    assert catalog.require_run(run_id).run_id == run_id
    with pytest.raises(UnknownRunError, match="nope"):
        catalog.require_run("nope")
