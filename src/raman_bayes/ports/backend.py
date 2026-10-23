"""ExecutionBackend Protocol (DESIGN.md §7.3): LocalBackend and SlurmArrayBackend implement it.

Every backend executes ``raman run-task`` per task; ``poll`` treats ``_DONE.json`` markers as the
source of truth. TaskState and SubmissionRecord are provisional models (``domain/run.py``).
"""

from collections.abc import Mapping, Sequence
from typing import Protocol

from raman_bayes.domain.run import RunPlan, SubmissionRecord, TaskState


class ExecutionBackend(Protocol):
    name: str

    def submit(self, plan: RunPlan, *, dry_run: bool = False) -> SubmissionRecord: ...

    def poll(self, run_id: str) -> Mapping[int, TaskState]: ...

    def cancel(self, run_id: str, tasks: Sequence[int] | None = None) -> None: ...

    def resubmit(self, run_id: str, tasks: Sequence[int]) -> SubmissionRecord: ...
