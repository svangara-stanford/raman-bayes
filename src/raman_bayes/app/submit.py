"""Submit, resubmit, cancel and run as the CLI driver (DESIGN.md §7.3, §7.4, §10).

The driver checks the run's status, has the backend start the tasks, records the submission
in ``submission.json`` and moves the run through the §7.4 states; tasks never touch the
catalog. On the local backend a submission is waited for: when ``submit_run`` returns, every
task has stopped and the run has been refreshed, so it is complete, failed or cancelled.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from raman_bayes.app.plan import PlanResult, plan_run
from raman_bayes.app.runs import RunContext, open_run
from raman_bayes.app.status import refresh
from raman_bayes.config.settings import LoadedSettings
from raman_bayes.domain.errors import LocalBackendOnlyError, MaxAttemptsError, SubmitRefusedError
from raman_bayes.domain.run import IN_FLIGHT, RunConfig, RunStatus, SubmissionRecord
from raman_bayes.storage.catalog import RunRow, TaskRow
from raman_bayes.storage.run_files import append_submission

__all__ = [
    "MAX_ATTEMPTS",
    "CancelResult",
    "CancelStatus",
    "SubmitRequest",
    "SubmitResult",
    "SubmitStatus",
    "cancel_run",
    "run_local",
    "submit_run",
]

MAX_ATTEMPTS = 3
_DONE_WITH = (RunStatus.COMPLETE, RunStatus.COLLECTED)


class SubmitStatus(StrEnum):
    """What a submit did: previewed, ran to the end, or found nothing to submit."""

    DRY_RUN = "dry_run"
    FINISHED = "finished"
    NOTHING_TO_DO = "nothing_to_do"


@dataclass(frozen=True)
class SubmitRequest:
    """``raman submit RUN [--dry-run] [--resubmit-failed]``."""

    run_id: str
    dry_run: bool = False
    resubmit_failed: bool = False


@dataclass(frozen=True)
class SubmitResult:
    """The run after a submit, and the submission's record (none when there was nothing to do).

    ``failures`` maps each task whose process raised to its error.
    """

    status: SubmitStatus
    run: RunRow
    record: SubmissionRecord | None = None
    tasks: tuple[TaskRow, ...] = ()
    failures: Mapping[int, str] = field(default_factory=dict)


def submit_run(request: SubmitRequest, loaded: LoadedSettings) -> SubmitResult:
    """Submit a planned run, or with ``resubmit_failed`` its stopped tasks, and wait for it.

    Raises RunNotInCatalogError, BackendNotSupportedError, SubmitRefusedError when the run's
    status does not allow the submission, MaxAttemptsError when a task to resubmit has used
    MAX_ATTEMPTS, and TaskLockedError when another process is running a submission of it.
    """
    context = open_run(request.run_id, loaded)
    if request.resubmit_failed:
        return _resubmit(context, dry_run=request.dry_run)
    run = context.run
    if run.status in (RunStatus.SUBMITTED, RunStatus.RUNNING, *_DONE_WITH):
        return SubmitResult(SubmitStatus.NOTHING_TO_DO, run)
    if run.status is RunStatus.FAILED:
        raise SubmitRefusedError(run.run_id, run.status, "use --resubmit-failed to retry it")
    if run.status is RunStatus.CANCELLED:
        raise SubmitRefusedError(run.run_id, run.status, "plan a new run instead")
    if request.dry_run:
        record = context.backend.submit(context.plan, dry_run=True)
        return SubmitResult(SubmitStatus.DRY_RUN, run, record)
    context.catalog.transition(run.run_id, RunStatus.PLANNED, RunStatus.SUBMITTED)
    return _wait(context, context.backend.submit(context.plan))


def _resubmit(context: RunContext, *, dry_run: bool) -> SubmitResult:
    run = context.run
    if run.status in (RunStatus.PLANNED, RunStatus.SUBMITTED, RunStatus.CANCELLED):
        advice = {
            RunStatus.PLANNED: "nothing has run yet; submit it without --resubmit-failed",
            RunStatus.SUBMITTED: "its tasks have not started yet; check it with raman status",
            RunStatus.CANCELLED: "plan a new run instead",
        }
        raise SubmitRefusedError(run.run_id, run.status, advice[run.status])
    view = refresh(context)
    stopped = [
        t for t in view.tasks if t.state.status not in IN_FLIGHT and not t.state.has_done_marker
    ]
    if view.run.status in _DONE_WITH or not stopped:
        return SubmitResult(SubmitStatus.NOTHING_TO_DO, view.run, tasks=view.tasks)
    capped = [t.task_index for t in stopped if t.state.attempts >= MAX_ATTEMPTS]
    if capped:
        raise MaxAttemptsError(run.run_id, capped, MAX_ATTEMPTS)
    indices = tuple(t.task_index for t in stopped)
    if dry_run:
        preview = context.backend.submit(context.plan, dry_run=True)
        record = SubmissionRecord.model_validate(preview.model_dump() | {"task_indices": indices})
        return SubmitResult(SubmitStatus.DRY_RUN, view.run, record, view.tasks)
    if view.run.status is RunStatus.FAILED:
        context.catalog.transition(run.run_id, RunStatus.FAILED, RunStatus.SUBMITTED)
    return _wait(context, context.backend.resubmit(run.run_id, indices))


def _wait(context: RunContext, record: SubmissionRecord) -> SubmitResult:
    run_dir = context.backend.runs_root / record.run_id
    append_submission(run_dir, record)
    failures = context.backend.wait(record.run_id)
    view = refresh(context)
    return SubmitResult(SubmitStatus.FINISHED, view.run, record, view.tasks, failures)


class CancelStatus(StrEnum):
    """What a cancel did."""

    CANCELLED = "cancelled"
    NOTHING_TO_DO = "nothing_to_do"


@dataclass(frozen=True)
class CancelResult:
    """The run after a cancel."""

    status: CancelStatus
    run: RunRow


def cancel_run(run_id: str, loaded: LoadedSettings) -> CancelResult:
    """Stop the run's live tasks on this host and mark it cancelled.

    A run that is complete, collected or already cancelled is left as it is. Raises the
    errors of ``open_run``.
    """
    context = open_run(run_id, loaded)
    run = context.run
    if run.status in (*_DONE_WITH, RunStatus.CANCELLED):
        return CancelResult(CancelStatus.NOTHING_TO_DO, run)
    context.backend.cancel(run_id)
    context.catalog.transition(run_id, run.status, RunStatus.CANCELLED)
    return CancelResult(CancelStatus.CANCELLED, context.reread())


def run_local(config: RunConfig, loaded: LoadedSettings) -> tuple[PlanResult, SubmitResult | None]:
    """Plan ``config`` and, if the plan is valid, submit it and wait (local backend only).

    The submit result is None when planning failed. Raises LocalBackendOnlyError when the
    profile's backend is not local, before planning anything.
    """
    settings = loaded.settings
    if settings.backend != "local":
        raise LocalBackendOnlyError("run", settings.profile, settings.backend)
    planned = plan_run(config, loaded)
    if planned.plan is None:
        return planned, None
    return planned, submit_run(SubmitRequest(planned.plan.run_id), loaded)
