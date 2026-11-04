"""Register a run folder copied from elsewhere (DESIGN.md §8.5 ``raman runs import``).

The folder is registered in place, so it must already be ``runs_root/<run_id>`` of the
profile in use, and its dataset must be in the catalog with the content hash the run was
planned on. Task states come from the folder's markers alone, whatever backend ran it: a
done marker is done, anything else unknown. The run then takes the legal §7.4 path to the
state those imply: planned if nothing ran, complete if every task is done (collected when
the folder also holds complete results), failed otherwise.
"""

import json
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import yaml

from raman_bayes.config.settings import LoadedSettings
from raman_bayes.domain.errors import (
    DatasetHashMismatchError,
    DatasetNotIngestedError,
    InvalidRunFolderError,
    InvalidRunPlanError,
    RunFolderPlaceError,
)
from raman_bayes.domain.run import RunPlan, RunStatus, TaskState, TaskStatus, config_hash
from raman_bayes.storage.catalog import Catalog, RunRecord, RunRow
from raman_bayes.storage.merge import MANIFEST_FILE, RESULTS_DIR
from raman_bayes.storage.run_files import read_plan, read_submissions
from raman_bayes.storage.shards import read_done, task_dir

__all__ = ["ImportResult", "ImportStatus", "import_run"]


class ImportStatus(StrEnum):
    """What runs import did."""

    IMPORTED = "imported"
    EXISTS = "exists"


@dataclass(frozen=True)
class ImportResult:
    """The run as the catalog now has it."""

    status: ImportStatus
    run: RunRow


def import_run(folder: Path, loaded: LoadedSettings) -> ImportResult:
    """Register the run in ``folder``; a run already in the catalog is left as it is.

    Raises RunFolderPlaceError when ``folder`` is not ``runs_root/<name>``,
    RunPlanNotFoundError or InvalidRunFolderError for a missing or unreadable plan,
    DatasetNotIngestedError and DatasetHashMismatchError.
    """
    paths = loaded.settings.paths
    expected = paths.runs_root / folder.name
    if folder.resolve() != expected.resolve():
        raise RunFolderPlaceError(folder, expected)
    try:
        plan = read_plan(expected)
    except InvalidRunPlanError as exc:
        raise InvalidRunFolderError(folder, str(exc)) from exc
    catalog = Catalog(paths.catalog)
    existing = catalog.get_run(plan.run_id)
    if existing is not None:
        return ImportResult(ImportStatus.EXISTS, existing)
    dataset = catalog.get_dataset(plan.dataset_id)
    if dataset is None:
        raise DatasetNotIngestedError(plan.run_id, plan.dataset_id)
    if dataset.content_hash != plan.dataset_hash:
        raise DatasetHashMismatchError(plan.run_id, plan.dataset_id)
    _register(catalog, plan)
    _advance(catalog, plan, expected)
    return ImportResult(ImportStatus.IMPORTED, catalog.require_run(plan.run_id))


def _register(catalog: Catalog, plan: RunPlan) -> None:
    # The original YAML stays where the run was planned; the recipe it held is in the plan.
    text = yaml.safe_dump(plan.priors.model_dump(mode="json"), sort_keys=False)
    catalog.add_priors_spec(plan.priors, plan.priors_hash, text, plan.created_at)
    record = RunRecord(
        run_id=plan.run_id,
        dataset_id=plan.dataset_id,
        priors_hash=plan.priors_hash,
        engine=plan.engine,
        engine_version=plan.engine_version,
        config_hash=config_hash(plan),
        git_sha=None,
        backend=plan.backend,
        profile=plan.profile,
        seed=plan.seed,
        created_at=plan.created_at,
    )
    catalog.create_run(record, plan.tasks)


def _advance(catalog: Catalog, plan: RunPlan, run_dir: Path) -> None:
    attempts = Counter(i for r in read_submissions(run_dir) for i in r.task_indices)
    states = {
        t.task_index: _state(run_dir, t.task_index, attempts[t.task_index]) for t in plan.tasks
    }
    if not any(s.has_done_marker for s in states.values()):
        return  # nothing ran: the run stays planned
    catalog.transition(plan.run_id, RunStatus.PLANNED, RunStatus.SUBMITTED)
    entered = catalog.refresh(plan.run_id, states)
    results = run_dir / RESULTS_DIR
    if RunStatus.COMPLETE in entered and _complete_results(results / MANIFEST_FILE):
        catalog.mark_collected(plan.run_id, str(results))


def _state(run_dir: Path, index: int, attempts: int) -> TaskState:
    done = read_done(task_dir(run_dir, index))
    if done is None:
        return TaskState(status=TaskStatus.UNKNOWN, attempts=attempts, has_done_marker=False)
    return TaskState(
        status=TaskStatus.DONE,
        attempts=attempts,
        has_done_marker=True,
        n_ok=done.n_ok,
        n_failed=done.n_failed,
    )


def _complete_results(manifest: Path) -> bool:
    try:
        text = manifest.read_text(encoding="utf-8")
    except FileNotFoundError:
        return False
    return json.loads(text).get("partial") is False
