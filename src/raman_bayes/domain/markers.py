"""The marker files a task leaves in ``shards/task=N/`` (DESIGN.md §7.2).

``_LOCK`` names the process working on the task; ``_DONE.json`` records the finished task,
its counts and the code and machine it ran on. Collect takes a run's provenance from the done
markers, and ``poll`` reads both to tell a task's state.
"""

from typing import Annotated, Literal

from pydantic import AwareDatetime, Field

from raman_bayes.domain._base import DomainModel

__all__ = ["DONE_SCHEMA_VERSION", "DoneMarker", "LockHolder"]

DONE_SCHEMA_VERSION: Literal["1.0"] = "1.0"

Count = Annotated[int, Field(ge=0)]


class LockHolder(DomainModel):
    """Contents of ``_LOCK``: which process holds the task (host and pid), and since when."""

    host: str
    pid: Annotated[int, Field(ge=1)]
    started_at: AwareDatetime


class DoneMarker(DomainModel):
    """Contents of ``_DONE.json``: a finished task, what it wrote, where and with what code.

    The counts are rows of the task's parts: fits rows (``n_ok + n_failed + n_skipped_qc ==
    n_fits``) and peaks rows. ``shard_hash`` covers the files in ``parts``. ``slurm_job_id``
    is ``<array_id>_<index>``; both Slurm ids are None off Slurm. The git fields are None
    when the package does not run from a git checkout.
    """

    schema_version: Literal["1.0"] = DONE_SCHEMA_VERSION
    run_id: str
    task_index: Count
    spec_start: Count
    spec_stop: Count
    parts: tuple[str, ...]
    n_fits: Count
    n_ok: Count
    n_failed: Count
    n_skipped_qc: Count
    n_peaks: Count
    shard_hash: str
    host: str
    pid: Annotated[int, Field(ge=1)]
    slurm_job_id: str | None
    slurm_array_job_id: str | None
    started_at: AwareDatetime
    completed_at: AwareDatetime
    package_version: str
    git_sha: str | None
    git_dirty: bool | None
    pixi_lock_hash: str | None
    engine: str
    engine_version: str
