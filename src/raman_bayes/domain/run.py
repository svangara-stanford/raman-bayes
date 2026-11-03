"""Run models (DESIGN.md §7): the request, the frozen run plan, tasks, statuses and submissions.

RunStatus lists the §7.4 states only; legal transitions are enforced by the catalog (§7.4).
A RunPlan checks itself on construction and on load: its run id is a canonical UUIDv7, its
``priors_hash`` is the §5 rule 9 hash of the recipe it carries, and its tasks are exactly the
contiguous partition of the spectra into ``spectra_per_task`` blocks. ``config_hash`` is the
catalog's hash of the fit configuration (§8.5). TaskStatus, TaskState and
SubmissionRecord are provisional until sub-step 12 settles them.
"""

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, Field, JsonValue, field_validator, model_validator
from pydantic_core import PydanticCustomError

from raman_bayes.domain._base import DomainModel
from raman_bayes.domain.priors import PriorsSpec
from raman_bayes.domain.priors_rules import priors_hash
from raman_bayes.domain.uuid7 import is_uuid7

__all__ = [
    "PLAN_SCHEMA_VERSION",
    "SEED_BITS",
    "SEED_LIMIT",
    "RunConfig",
    "RunPlan",
    "RunStatus",
    "SubmissionRecord",
    "TaskSpec",
    "TaskState",
    "TaskStatus",
    "config_hash",
    "task_ranges",
]

PLAN_SCHEMA_VERSION = "1.0"
SEED_BITS = 63  # seeds fit the catalog's signed 64-bit INTEGER column
SEED_LIMIT = 2**SEED_BITS

Seed = Annotated[int, Field(ge=0, lt=SEED_LIMIT)]
SpectraPerTask = Annotated[int, Field(ge=1)]


class RunStatus(StrEnum):
    """Run lifecycle states (§7.4)."""

    PLANNED = "planned"
    SUBMITTED = "submitted"
    RUNNING = "running"
    COMPLETE = "complete"
    COLLECTED = "collected"
    CANCELLED = "cancelled"
    FAILED = "failed"


class TaskSpec(DomainModel):
    """One array task: spectra ``spec_start`` (inclusive) to ``spec_stop`` (exclusive)."""

    task_index: int
    spec_start: int
    spec_stop: int


def task_ranges(n_spectra: int, spectra_per_task: int) -> tuple[TaskSpec, ...]:
    """Return the tasks covering ``0..n_spectra``: task k is ``[k*s, min((k+1)*s, n_spectra))``."""
    starts = range(0, n_spectra, spectra_per_task)
    return tuple(
        TaskSpec(task_index=k, spec_start=start, spec_stop=min(start + spectra_per_task, n_spectra))
        for k, start in enumerate(starts)
    )


class RunConfig(DomainModel):
    """What ``raman plan`` was asked for, before the plan resolves it (§7.1).

    ``engine`` is ``--engine`` or the profile's ``engine.name``; ``profile`` is the profile as
    given (name or path). ``seed`` None means draw one at plan time.
    """

    dataset_id: str
    priors_path: Path
    engine: str
    spectra_per_task: SpectraPerTask | None = None
    seed: Seed | None = None
    profile: str


class RunPlan(DomainModel):
    """Contents of ``runs/<run_id>/run_plan.json``: everything ``run-task`` needs (§7.1).

    ``dataset_hash`` is the manifest's ``content_hash``; ``dataset_uri`` the absolute store
    path; ``engine_config`` the EngineConfig fields; ``resolved_config`` the settings after
    precedence and expansion. Preprocessing is ``priors.preprocessing``.
    """

    schema_version: str = PLAN_SCHEMA_VERSION
    run_id: str
    created_at: AwareDatetime
    dataset_id: str
    dataset_hash: str
    dataset_uri: str
    n_spectra: Annotated[int, Field(ge=1)]
    priors: PriorsSpec
    priors_hash: str
    engine: str
    engine_version: str
    engine_config: dict[str, JsonValue]
    backend: Literal["local", "slurm_array"]
    profile: str
    resolved_config: dict[str, JsonValue]
    seed: Seed
    spectra_per_task: SpectraPerTask
    tasks: tuple[TaskSpec, ...]

    @field_validator("run_id")
    @classmethod
    def _run_id_is_uuid7(cls, run_id: str) -> str:
        if not is_uuid7(run_id):
            raise PydanticCustomError(
                "run_id_not_uuid7",
                "run_id {run_id!r} is not a lowercase hyphenated UUIDv7",
                {"run_id": run_id},
            )
        return run_id

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        expected = priors_hash(self.priors)
        if self.priors_hash != expected:
            raise PydanticCustomError(
                "priors_hash_mismatch",
                "priors_hash {given} is not the hash of the recipe in the plan ({expected})",
                {"given": self.priors_hash, "expected": expected},
            )
        if self.tasks != task_ranges(self.n_spectra, self.spectra_per_task):
            raise PydanticCustomError(
                "tasks_not_partition",
                "tasks must split spectra 0..{n} into consecutive blocks of {s}, in order",
                {"n": self.n_spectra, "s": self.spectra_per_task},
            )
        return self


def config_hash(plan: RunPlan) -> str:
    """Return the SHA-256 hex digest of the plan's engine, engine version and engine config.

    The digest is taken over sorted, compact JSON. Runs with the same ``dataset_id``,
    ``priors_hash``, ``config_hash`` and ``seed`` are the same analysis and should give the same
    results, so where and how a run executes (paths, profile, backend, workers) is left out.
    """
    fit = {
        "engine": plan.engine,
        "engine_version": plan.engine_version,
        "engine_config": plan.engine_config,
    }
    text = json.dumps(fit, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TaskStatus(StrEnum):
    """Provisional: per-task state reported by a backend (markers merged with sacct, §7.3)."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    TIMEOUT = "timeout"
    OUT_OF_MEMORY = "out_of_memory"
    PREEMPTED = "preempted"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class TaskState(DomainModel):
    """Provisional: one task as seen by ``ExecutionBackend.poll``; mirrors the §8.5 task table.

    ``has_done_marker`` records whether ``shards/task=N/_DONE.json`` exists, the source of truth.
    """

    status: TaskStatus
    attempts: int
    slurm_job_id: str | None = None
    has_done_marker: bool
    n_ok: int | None = None
    n_failed: int | None = None
    error: str | None = None


class SubmissionRecord(DomainModel):
    """Provisional: contents of ``runs/<run_id>/submission.json`` (§8.1).

    ``submitted_at`` is pipeline-generated, so it must be timezone-aware (written in UTC).
    """

    run_id: str
    backend: str
    submitted_at: AwareDatetime
    dry_run: bool
    task_indices: tuple[int, ...]
    slurm_array_job_id: str | None = None
    script: str | None = None
