"""One run as the CLI driver works on it: its catalog row, its plan and its backend (§7.3).

A run is executed by the backend it was planned for, configured from the profile in use; in
Phase 1 that must be the local backend.
"""

from dataclasses import dataclass
from datetime import timedelta
from functools import partial

from raman_bayes.app.run_task import run_task_in_worker
from raman_bayes.config.settings import LoadedSettings
from raman_bayes.domain.errors import BackendNotSupportedError, RunNotInCatalogError
from raman_bayes.domain.run import RunPlan
from raman_bayes.orchestration.local import LocalBackend
from raman_bayes.storage.catalog import Catalog, RunRow
from raman_bayes.storage.run_files import read_plan

__all__ = ["RunContext", "open_run"]


@dataclass(frozen=True)
class RunContext:
    """A run's catalog, its row as last read, its plan and the backend that executes it."""

    catalog: Catalog
    run: RunRow
    plan: RunPlan
    backend: LocalBackend

    def reread(self) -> RunRow:
        """Return the run's row as the catalog has it now."""
        row = self.catalog.get_run(self.run.run_id)
        if row is None:
            raise RunNotInCatalogError(self.run.run_id, self.catalog.path)
        return row


def open_run(run_id: str, loaded: LoadedSettings) -> RunContext:
    """Return the run ``run_id`` with the backend this profile runs it on.

    Raises RunNotInCatalogError for an unknown run, the run_plan errors of ``read_plan``, and
    BackendNotSupportedError when the run was planned for another backend than the profile
    uses, or for one that is not supported yet.
    """
    settings = loaded.settings
    catalog = Catalog(settings.paths.catalog)
    run = catalog.get_run(run_id)
    if run is None:
        raise RunNotInCatalogError(run_id, settings.paths.catalog)
    plan = read_plan(settings.paths.runs_root / run_id)
    if plan.backend != settings.backend or plan.backend != "local":
        raise BackendNotSupportedError(run_id, plan.backend, settings.profile, settings.backend)
    lock_stale = timedelta(minutes=settings.lock_stale_minutes)
    runner = partial(run_task_in_worker, lock_stale=lock_stale)
    backend = LocalBackend(settings.paths.runs_root, settings.local.workers, lock_stale, runner)
    return RunContext(catalog, run, plan, backend)
