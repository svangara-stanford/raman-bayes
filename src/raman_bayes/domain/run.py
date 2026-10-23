"""Run models (DESIGN.md §7): the frozen run plan, task specs, statuses and submissions.

RunStatus lists the §7.4 states only; legal transitions are enforced by the catalog (§7.4).
RunConfig, TaskStatus, TaskState and SubmissionRecord are named by §2/§7.3 but not defined
there; their shapes are provisional until Phase 1 settles them in DESIGN.md.
"""

from enum import StrEnum
from pathlib import Path

from pydantic import AwareDatetime, JsonValue

from raman_bayes.domain._base import DomainModel
from raman_bayes.domain.priors import PreprocessingSpec


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


class RunPlan(DomainModel):
    """Contents of ``runs/<run_id>/run_plan.json`` (§7.1). ``run_id`` is a UUIDv7 string."""

    run_id: str
    dataset_id: str
    dataset_hash: str
    priors_hash: str
    engine: str
    engine_config: dict[str, JsonValue]
    preprocessing: PreprocessingSpec
    profile: str
    resolved_config: dict[str, JsonValue]
    seed: int
    tasks: tuple[TaskSpec, ...]


class RunConfig(DomainModel):
    """Provisional: the user's request to ``raman plan`` before it is resolved into a RunPlan."""

    dataset_id: str
    priors_path: Path
    engine: str
    spectra_per_task: int | None = None
    seed: int | None = None
    profile: str


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
