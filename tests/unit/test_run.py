"""Run models (DESIGN.md §7.1, §7.3, §7.4, §8.5): plan, task specs, statuses, submissions."""

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import BaseModel, ValidationError

from raman_bayes.domain import run
from raman_bayes.domain._base import DomainModel
from raman_bayes.domain.priors import BaselineSpec, DespikeSpec, PreprocessingSpec
from raman_bayes.domain.run import (
    RunConfig,
    RunPlan,
    RunStatus,
    SubmissionRecord,
    TaskSpec,
    TaskState,
    TaskStatus,
)

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
            "run_id",
            "dataset_id",
            "dataset_hash",
            "priors_hash",
            "engine",
            "engine_config",
            "preprocessing",
            "profile",
            "resolved_config",
            "seed",
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

PROVISIONAL: list[type] = [RunConfig, TaskStatus, TaskState, SubmissionRecord]

PREPROCESSING = PreprocessingSpec(
    margin_cm1=25.0,
    despike=DespikeSpec(method="median_mad", kernel=3, thresh=3.0),
    baseline=BaselineSpec(method="als", lam=1.0e5, p=0.01),
    smoothing=None,
    clamp_nonnegative=True,
)


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
task_specs = st.builds(TaskSpec, task_index=ints, spec_start=ints, spec_stop=ints)
plans = st.builds(
    RunPlan,
    run_id=idents,
    dataset_id=idents,
    dataset_hash=idents,
    priors_hash=idents,
    engine=idents,
    engine_config=json_objects,
    preprocessing=st.just(PREPROCESSING),
    profile=idents,
    resolved_config=json_objects,
    seed=st.integers(min_value=0, max_value=2**64 - 1),
    tasks=st.lists(task_specs, max_size=5).map(tuple),
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
