"""Run models (DESIGN.md §7.1, §7.3, §7.4, §8.5): plan, task specs, statuses, submissions."""

import hashlib
import json
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import pytest
import yaml
from hypothesis import given
from hypothesis import strategies as st
from pydantic import BaseModel, ValidationError

from raman_bayes.domain import run
from raman_bayes.domain._base import DomainModel
from raman_bayes.domain.priors import PriorsSpec
from raman_bayes.domain.priors_rules import priors_hash
from raman_bayes.domain.run import (
    RunConfig,
    RunPlan,
    RunStatus,
    SubmissionRecord,
    TaskSpec,
    TaskState,
    TaskStatus,
    config_hash,
    task_ranges,
)
from raman_bayes.domain.uuid7 import uuid7

# Copied by hand from DESIGN.md; never generated from the code under test.
RUN_STATUS_VALUES = {
    "planned",
    "submitted",
    "running",
    "complete",
    "collected",
    "cancelled",
    "failed",
}
TASK_STATUS_VALUES = {
    "pending",
    "running",
    "done",
    "failed",
    "timeout",
    "out_of_memory",
    "preempted",
    "cancelled",
    "unknown",
}
# §8.5 task table columns.
CATALOG_TASK_COLUMNS = {
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
}

MODEL_KEYS: list[tuple[type[BaseModel], set[str]]] = [
    (TaskSpec, {"task_index", "spec_start", "spec_stop"}),
    (
        RunPlan,
        {
            "schema_version",
            "run_id",
            "created_at",
            "dataset_id",
            "dataset_hash",
            "dataset_uri",
            "n_spectra",
            "priors",
            "priors_hash",
            "engine",
            "engine_version",
            "engine_config",
            "backend",
            "profile",
            "resolved_config",
            "seed",
            "spectra_per_task",
            "tasks",
        },
    ),
    (
        RunConfig,
        {"dataset_id", "priors_path", "engine", "spectra_per_task", "seed", "profile"},
    ),
    (
        TaskState,
        {
            "status",
            "attempts",
            "slurm_job_id",
            "has_done_marker",
            "n_ok",
            "n_failed",
            "error",
        },
    ),
    (
        SubmissionRecord,
        {
            "run_id",
            "backend",
            "submitted_at",
            "dry_run",
            "task_indices",
            "slurm_array_job_id",
            "script",
        },
    ),
]

PROVISIONAL: list[type] = [TaskStatus, TaskState, SubmissionRecord]

FIXTURE = Path(__file__).parents[1] / "fixtures" / "priors" / "design_example.yaml"
PRIORS = PriorsSpec.model_validate(yaml.safe_load(FIXTURE.read_text(encoding="utf-8")))
PRIORS_HASH = priors_hash(PRIORS)
RUN_IDS = [str(uuid7()) for _ in range(5)]


# ---------------------------------------------------------------- enums


@pytest.mark.parametrize(
    ("enum", "values"),
    [(RunStatus, RUN_STATUS_VALUES), (TaskStatus, TASK_STATUS_VALUES)],
    ids=["RunStatus", "TaskStatus"],
)
def test_enum_values(enum: type[StrEnum], values: set[str]) -> None:
    assert issubclass(enum, StrEnum)
    assert {member.value for member in enum} == values
    assert {member.name.lower() for member in enum} == values


def test_run_status_has_no_transition_table() -> None:
    # Transitions are enforced in storage/catalog.py (Phase 1), not on the enum.
    assert not [name for name in vars(RunStatus) if "transition" in name.lower()]


# ---------------------------------------------------------------- structure


@pytest.mark.parametrize(("model", "keys"), MODEL_KEYS, ids=lambda v: getattr(v, "__name__", ""))
def test_key_names(model: type[BaseModel], keys: set[str]) -> None:
    assert set(model.model_fields) == keys


def test_task_state_mirrors_catalog_task_table() -> None:
    assert set(TaskState.model_fields) - {"has_done_marker"} <= CATALOG_TASK_COLUMNS


def test_every_model_is_frozen_and_forbids_extra() -> None:
    models = {
        obj
        for obj in vars(run).values()
        if isinstance(obj, type) and issubclass(obj, DomainModel) and obj.__module__ == run.__name__
    }
    assert models == {m for m, _ in MODEL_KEYS}
    for model in models:
        assert model.model_config.get("frozen") is True, model.__name__
        assert model.model_config.get("extra") == "forbid", model.__name__


@pytest.mark.parametrize("obj", PROVISIONAL, ids=lambda o: o.__name__)
def test_provisional_marker(obj: type) -> None:
    assert obj.__doc__ is not None
    assert "provisional" in obj.__doc__.lower()


def test_optional_defaults() -> None:
    state = TaskState(status=TaskStatus.PENDING, attempts=0, has_done_marker=False)
    assert (state.slurm_job_id, state.n_ok, state.n_failed, state.error) == (None,) * 4
    config = RunConfig(dataset_id="d", priors_path=Path("p.yaml"), engine="mock", profile="local")
    assert (config.spectra_per_task, config.seed) == (None, None)


# ---------------------------------------------------------------- TaskState


def test_task_state_parses_status_string() -> None:
    state = TaskState.model_validate(
        {"status": "out_of_memory", "attempts": 2, "has_done_marker": False}
    )
    assert state.status is TaskStatus.OUT_OF_MEMORY


def test_task_state_rejects_unknown_status() -> None:
    with pytest.raises(ValidationError) as exc:
        TaskState.model_validate({"status": "zombie", "attempts": 0, "has_done_marker": False})
    assert exc.value.errors()[0]["loc"] == ("status",)


def test_task_state_is_frozen() -> None:
    state = TaskState(status=TaskStatus.RUNNING, attempts=1, has_done_marker=False)
    with pytest.raises(ValidationError, match="frozen"):
        state.status = TaskStatus.DONE  # type: ignore[misc]


# ---------------------------------------------------------------- SubmissionRecord


def _submission(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "run_id": "0190b3d4-0000-7000-8000-000000000000",
        "backend": "local",
        "submitted_at": datetime(2026, 10, 23, 12, 0, tzinfo=UTC),
        "dry_run": False,
        "task_indices": [0, 1],
    }
    return fields | overrides


def test_submission_submitted_at_must_be_aware() -> None:
    SubmissionRecord.model_validate(_submission())
    with pytest.raises(ValidationError) as exc:
        SubmissionRecord.model_validate(_submission(submitted_at=datetime(2026, 10, 23, 12, 0)))
    assert exc.value.errors()[0]["type"] == "timezone_aware"


# ---------------------------------------------------------------- round-trips

ints = st.integers(min_value=0, max_value=2**31 - 1)
idents = st.text(min_size=1, max_size=40)
json_values = st.recursive(
    st.none() | st.booleans() | st.integers() | st.text(),
    lambda inner: st.lists(inner, max_size=3) | st.dictionaries(st.text(), inner, max_size=3),
    max_leaves=10,
)
json_objects = st.dictionaries(st.text(), json_values, max_size=4)
utc_dt = st.datetimes(
    min_value=datetime(1990, 1, 1), max_value=datetime(2100, 1, 1), timezones=st.just(UTC)
)
sizes = st.integers(min_value=1, max_value=5000).flatmap(
    lambda n: st.tuples(st.just(n), st.integers(min_value=1, max_value=n + 10))
)
plans = sizes.flatmap(
    lambda ns: st.builds(
        RunPlan,
        run_id=st.sampled_from(RUN_IDS),
        created_at=utc_dt,
        dataset_id=idents,
        dataset_hash=idents,
        dataset_uri=idents,
        n_spectra=st.just(ns[0]),
        priors=st.just(PRIORS),
        priors_hash=st.just(PRIORS_HASH),
        engine=idents,
        engine_version=idents,
        engine_config=json_objects,
        backend=st.sampled_from(["local", "slurm_array"]),
        profile=idents,
        resolved_config=json_objects,
        seed=st.integers(min_value=0, max_value=2**63 - 1),
        spectra_per_task=st.just(ns[1]),
        tasks=st.just(task_ranges(*ns)),
    )
)
submissions = st.builds(
    SubmissionRecord,
    run_id=idents,
    backend=st.sampled_from(["local", "slurm_array"]),
    submitted_at=utc_dt,
    dry_run=st.booleans(),
    task_indices=st.lists(ints, max_size=5).map(tuple),
    slurm_array_job_id=st.none() | st.from_regex(r"\A[0-9]{1,9}\Z"),
    script=st.none() | st.text(),
)
task_states = st.builds(
    TaskState,
    status=st.sampled_from(TaskStatus),
    attempts=ints,
    slurm_job_id=st.none() | st.from_regex(r"\A[0-9]{1,9}_[0-9]{1,4}\Z"),
    has_done_marker=st.booleans(),
    n_ok=st.none() | ints,
    n_failed=st.none() | ints,
    error=st.none() | st.text(),
)


@given(plans)
def test_run_plan_json_round_trip(plan: RunPlan) -> None:
    assert RunPlan.model_validate_json(plan.model_dump_json()) == plan


@given(submissions)
def test_submission_json_round_trip(record: SubmissionRecord) -> None:
    assert SubmissionRecord.model_validate_json(record.model_dump_json()) == record


@given(task_states)
def test_task_state_json_round_trip(state: TaskState) -> None:
    assert TaskState.model_validate_json(state.model_dump_json()) == state


def test_run_config_json_round_trip() -> None:
    config = RunConfig(
        dataset_id="NMC811_v0-0123456789",
        priors_path=Path("configs/priors/nmc_commercial.yaml"),
        engine="mock",
        spectra_per_task=8,
        seed=12345,
        profile="local",
    )
    assert RunConfig.model_validate_json(config.model_dump_json()) == config


# ---------------------------------------------------------------- RunConfig and RunPlan rules


def _plan(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "run_id": RUN_IDS[0],
        "created_at": datetime(2026, 11, 2, 12, 0, tzinfo=UTC),
        "dataset_id": "NMC811_v0-0123456789",
        "dataset_hash": "ab" * 32,
        "dataset_uri": "/data/datasets/NMC811_v0-0123456789.zarr",
        "n_spectra": 25,
        "priors": PRIORS,
        "priors_hash": PRIORS_HASH,
        "engine": "mock",
        "engine_version": "0.1.0",
        "engine_config": {"n_iter": 6000, "burn_in": 3000, "thin": 5, "options": {}},
        "backend": "local",
        "profile": "local",
        "resolved_config": {},
        "seed": 7,
        "spectra_per_task": 16,
        "tasks": [(0, 0, 16), (1, 16, 25)],
    }
    fields |= overrides
    fields["tasks"] = [
        {"task_index": k, "spec_start": a, "spec_stop": b} for k, a, b in fields["tasks"]
    ]
    return fields


def _error_types(fields: dict[str, Any]) -> set[str]:
    with pytest.raises(ValidationError) as exc:
        RunPlan.model_validate(fields)
    return {e["type"] for e in exc.value.errors()}


def test_valid_plan_and_schema_version() -> None:
    plan = RunPlan.model_validate(_plan())
    assert plan.schema_version == "1.0"
    assert plan.tasks == task_ranges(25, 16)


@pytest.mark.parametrize(
    "tasks",
    [
        [(0, 0, 16)],  # does not reach n_spectra
        [(0, 0, 16), (1, 17, 25)],  # gap
        [(0, 0, 16), (1, 15, 25)],  # overlap
        [(1, 0, 16), (0, 16, 25)],  # out of order
        [(0, 0, 10), (1, 10, 25)],  # not k * spectra_per_task
        [(0, 0, 16), (1, 16, 26)],  # beyond n_spectra
        [],
    ],
)
def test_run_plan_tasks_must_partition_n_spectra(tasks: list[tuple[int, int, int]]) -> None:
    assert "tasks_not_partition" in _error_types(_plan(tasks=tasks))


def test_run_plan_priors_hash_must_match() -> None:
    assert "priors_hash_mismatch" in _error_types(_plan(priors_hash="0" * 64))


@pytest.mark.parametrize(
    "run_id",
    [str(uuid.uuid4()), RUN_IDS[0].upper(), RUN_IDS[0].replace("-", ""), "run-1"],
)
def test_run_plan_run_id_must_be_uuid7(run_id: str) -> None:
    assert "run_id_not_uuid7" in _error_types(_plan(run_id=run_id))


def test_run_plan_created_at_must_be_aware() -> None:
    assert "timezone_aware" in _error_types(_plan(created_at=datetime(2026, 11, 2, 12, 0)))


def test_config_hash_is_sha256_of_the_fit_configuration() -> None:
    plan = RunPlan.model_validate(_plan())
    fit = {"engine": "mock", "engine_version": "0.1.0", "engine_config": plan.engine_config}
    text = json.dumps(fit, sort_keys=True, separators=(",", ":"))
    assert config_hash(plan) == hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_config_hash_ignores_where_and_how_the_run_executes() -> None:
    laptop = RunPlan.model_validate(_plan(resolved_config={"paths": {"runs_root": "/a/runs"}}))
    cluster = RunPlan.model_validate(
        _plan(
            run_id=RUN_IDS[1],
            dataset_uri="/scratch/datasets/NMC811_v0-0123456789.zarr",
            backend="slurm_array",
            profile="sherlock",
            resolved_config={
                "paths": {"runs_root": "/scratch/runs"},
                "profile_source": "/home/u/sherlock.yaml",
                "local": {"workers": 3},
                "slurm": {"pixi_bin": "/home/u/.pixi/bin/pixi"},
            },
            seed=8,
        )
    )
    assert config_hash(laptop) == config_hash(cluster)


@pytest.mark.parametrize(
    "change",
    [
        {"engine": "bayes_v1"},
        {"engine_version": "0.2.0"},
        {"engine_config": {"n_iter": 8000, "burn_in": 3000, "thin": 5, "options": {}}},
    ],
)
def test_config_hash_tracks_engine_version_and_config(change: dict[str, Any]) -> None:
    assert config_hash(RunPlan.model_validate(_plan(**change))) != config_hash(
        RunPlan.model_validate(_plan())
    )


def test_seed_range() -> None:
    config = {"dataset_id": "d", "priors_path": "p.yaml", "engine": "mock", "profile": "local"}
    for ok in (0, 2**63 - 1):
        assert RunConfig.model_validate({**config, "seed": ok}).seed == ok
        assert RunPlan.model_validate(_plan(seed=ok)).seed == ok
    for bad in (-1, 2**63):
        with pytest.raises(ValidationError):
            RunConfig.model_validate({**config, "seed": bad})
        with pytest.raises(ValidationError):
            RunPlan.model_validate(_plan(seed=bad))


def test_run_config_spectra_per_task_at_least_1() -> None:
    config = {"dataset_id": "d", "priors_path": "p.yaml", "engine": "mock", "profile": "local"}
    assert RunConfig.model_validate({**config, "spectra_per_task": 1}).spectra_per_task == 1
    with pytest.raises(ValidationError):
        RunConfig.model_validate({**config, "spectra_per_task": 0})
    with pytest.raises(ValidationError):
        RunPlan.model_validate(_plan(spectra_per_task=0, tasks=[]))


def test_run_config_and_run_plan_not_provisional() -> None:
    for model in (RunConfig, RunPlan):
        assert model.__doc__ is not None
        assert "provisional" not in model.__doc__.lower(), model.__name__
