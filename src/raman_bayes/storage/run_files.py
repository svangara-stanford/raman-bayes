"""The JSON files of a run folder: ``run_plan.json`` and ``submission.json`` (DESIGN.md §7, §8.1).

The plan is written once when the run is planned. ``submission.json`` is the list of real
submissions in attempt order, which the CLI driver extends; each write replaces the whole
file atomically.
"""

from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from raman_bayes.domain.errors import (
    InvalidRunPlanError,
    InvalidSubmissionsError,
    RunPlanNotFoundError,
)
from raman_bayes.domain.run import RunPlan, SubmissionRecord
from raman_bayes.storage.atomic import write_text_atomically

__all__ = [
    "RUN_PLAN_FILE",
    "SUBMISSION_FILE",
    "append_submission",
    "read_plan",
    "read_submissions",
    "write_plan",
]

RUN_PLAN_FILE = "run_plan.json"
SUBMISSION_FILE = "submission.json"
_SUBMISSIONS = TypeAdapter(list[SubmissionRecord])


def write_plan(run_dir: Path, plan: RunPlan) -> Path:
    """Create ``run_dir`` and write ``plan`` to its ``run_plan.json``; return the file.

    Raises FileExistsError when ``run_dir`` already exists: a plan is never overwritten.
    """
    run_dir.mkdir(parents=True)
    path = run_dir / RUN_PLAN_FILE
    write_text_atomically(path, plan.model_dump_json(indent=2))
    return path


def read_plan(run_dir: Path) -> RunPlan:
    """Return the plan in ``run_dir/run_plan.json``, checked as on construction.

    Raises RunPlanNotFoundError when there is no plan file, and InvalidRunPlanError when it is
    not a valid plan or belongs to another run.
    """
    path = run_dir / RUN_PLAN_FILE
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise RunPlanNotFoundError(path) from None
    try:
        plan = RunPlan.model_validate_json(text)
    except ValidationError as exc:
        raise InvalidRunPlanError(path, str(exc)) from exc
    if plan.run_id != run_dir.name:
        raise InvalidRunPlanError(path, f"it is the plan of run {plan.run_id}")
    return plan


def read_submissions(run_dir: Path) -> tuple[SubmissionRecord, ...]:
    """Return the run's real submissions in attempt order; empty before the first.

    Raises InvalidSubmissionsError when the file is there but is not a list of records.
    """
    path = run_dir / SUBMISSION_FILE
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ()
    try:
        return tuple(_SUBMISSIONS.validate_json(text))
    except ValidationError as exc:
        raise InvalidSubmissionsError(path, str(exc)) from exc


def append_submission(run_dir: Path, record: SubmissionRecord) -> None:
    """Add ``record`` to the end of ``submission.json``, rewriting the file atomically."""
    records = [*read_submissions(run_dir), record]
    text = _SUBMISSIONS.dump_json(records, indent=2).decode("utf-8")
    write_text_atomically(run_dir / SUBMISSION_FILE, text)
