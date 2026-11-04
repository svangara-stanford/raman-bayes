"""Refresh a run from its backend and report its tasks (DESIGN.md §7.4, §10 ``status``).

A refresh merges ``poll`` into the catalog's task rows and applies every run transition that
implies, which may be several at once (submitted, running, complete). ``watch`` refreshes until
the run has stopped: complete, collected, cancelled or failed.
"""

import time
from dataclasses import dataclass

from raman_bayes.app.runs import RunContext, open_run
from raman_bayes.config.settings import LoadedSettings
from raman_bayes.domain.run import IN_FLIGHT, RunStatus, TaskStatus
from raman_bayes.storage.catalog import RunRow, TaskRow

__all__ = ["STOPPED", "WATCH_INTERVAL_S", "RunView", "refresh", "refresh_run", "watch_run"]

WATCH_INTERVAL_S = 5.0
STOPPED = frozenset(
    {RunStatus.COMPLETE, RunStatus.COLLECTED, RunStatus.CANCELLED, RunStatus.FAILED}
)


@dataclass(frozen=True)
class RunView:
    """A run and its tasks after a refresh, and the statuses the refresh moved it through."""

    run: RunRow
    tasks: tuple[TaskRow, ...]
    entered: tuple[RunStatus, ...]

    @property
    def failures(self) -> tuple[TaskRow, ...]:
        """The tasks that have stopped without finishing."""
        return tuple(
            t
            for t in self.tasks
            if t.state.status is not TaskStatus.DONE and t.state.status not in IN_FLIGHT
        )


def refresh(context: RunContext) -> RunView:
    """Poll the run's backend, store the task states and apply the transitions they imply."""
    run_id = context.run.run_id
    entered = context.catalog.refresh(run_id, context.backend.poll(run_id))
    return RunView(context.reread(), tuple(context.catalog.tasks(run_id)), entered)


def refresh_run(run_id: str, loaded: LoadedSettings) -> RunView:
    """Refresh run ``run_id``; raises the errors of ``open_run``."""
    return refresh(open_run(run_id, loaded))


def watch_run(run_id: str, loaded: LoadedSettings) -> RunView:
    """Refresh run ``run_id`` every WATCH_INTERVAL_S seconds until it has stopped.

    The view returned lists every status entered while watching.
    """
    context = open_run(run_id, loaded)
    entered: list[RunStatus] = []
    while True:
        view = refresh(context)
        entered.extend(view.entered)
        if view.run.status in STOPPED:
            return RunView(view.run, view.tasks, tuple(entered))
        time.sleep(WATCH_INTERVAL_S)
