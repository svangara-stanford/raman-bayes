"""ExecutionBackend Protocol (DESIGN.md §7.3): LocalBackend and SlurmArrayBackend implement it.

Every backend executes ``raman run-task`` per task; ``poll`` treats ``_DONE.json`` markers as the
source of truth. TaskState and SubmissionRecord are in ``domain/run.py``.
"""

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from raman_bayes.domain.run import RunPlan, SubmissionRecord, TaskState


@runtime_checkable
class ExecutionBackend(Protocol):
    """Runs the tasks of a RunPlan somewhere and reports their state."""

    name: str

    def submit(self, plan: RunPlan, *, dry_run: bool = False) -> SubmissionRecord:
        """Start every task of ``plan``; with ``dry_run``, start nothing and write nothing."""

    def poll(self, run_id: str) -> Mapping[int, TaskState]:
        """Return one TaskState per task index of ``run_id``."""

    def cancel(self, run_id: str, tasks: Sequence[int] | None = None) -> None:
        """Stop ``tasks`` of ``run_id``, or every task when ``tasks`` is None."""

    def resubmit(self, run_id: str, tasks: Sequence[int]) -> SubmissionRecord:
        """Start ``tasks`` of ``run_id`` again and return the new attempt's record."""
