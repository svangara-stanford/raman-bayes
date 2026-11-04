"""Collect a run (DESIGN.md §7.4, §7.5, §8): merge its shards into ``results/``.

The run is refreshed first, so a run whose tasks have all finished is complete. Collect refuses
a run that is not complete unless ``partial`` is asked for; a partial collect merges the tasks
that have finished, marks the manifest partial and changes no status. When the profile sets
``archive_root``, the results are also copied to ``archive_root/<run_id>/results``, which
becomes the run's ``result_uri`` (scratch space is purged; the archive is not). The catalog is
written last, after the result files.
"""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from raman_bayes.app.runs import open_run
from raman_bayes.app.status import refresh
from raman_bayes.config.settings import LoadedSettings
from raman_bayes.domain.errors import CollectRefusedError, NothingToCollectError
from raman_bayes.domain.run import RunStatus
from raman_bayes.domain.run_manifest import RunManifest, build_manifest
from raman_bayes.storage.catalog import RunRow
from raman_bayes.storage.merge import RESULTS_DIR, copy_results, merge_tasks, write_results
from raman_bayes.storage.shards import read_done, task_dir

__all__ = ["CollectRequest", "CollectResult", "CollectStatus", "collect_run"]


class CollectStatus(StrEnum):
    """What collect did."""

    COLLECTED = "collected"
    PARTIAL = "partial"
    NOTHING_TO_DO = "nothing_to_do"


@dataclass(frozen=True)
class CollectRequest:
    """``raman collect RUN [--partial]``."""

    run_id: str
    partial: bool = False


@dataclass(frozen=True)
class CollectResult:
    """The run after collect, and what was written; nothing is written for nothing to do."""

    status: CollectStatus
    run: RunRow
    manifest: RunManifest | None = None
    files: tuple[Path, ...] = ()
    archive_dir: Path | None = None

    @property
    def results_dir(self) -> Path | None:
        """The folder the result files were written to."""
        return self.files[0].parent if self.files else None


def collect_run(request: CollectRequest, loaded: LoadedSettings) -> CollectResult:
    """Merge the run's finished tasks into ``results/`` and, unless partial, mark it collected.

    Raises the errors of ``open_run``, CollectRefusedError for an incomplete run without
    ``partial``, NothingToCollectError when no task has finished, and the merge and manifest
    errors (ShardMismatchError, PartSchemaError, DuplicateRowsError, MixedCodeError).
    """
    context = open_run(request.run_id, loaded)
    run = refresh(context).run
    if run.status is RunStatus.COLLECTED:
        return CollectResult(CollectStatus.NOTHING_TO_DO, run)
    if run.status is not RunStatus.COMPLETE and not request.partial:
        raise CollectRefusedError(run.run_id, run.status)
    run_dir = context.backend.runs_root / run.run_id
    markers = {
        t.task_index: done
        for t in context.plan.tasks
        if (done := read_done(task_dir(run_dir, t.task_index))) is not None
    }
    if not markers:
        raise NothingToCollectError(run.run_id)
    tables = merge_tasks(run_dir, markers)
    manifest = build_manifest(context.plan, markers, tables.fits)
    files = write_results(run_dir / RESULTS_DIR, tables, manifest.model_dump_json(indent=2))
    archive_root = loaded.settings.paths.archive_root
    archive = archive_root / run.run_id / RESULTS_DIR if archive_root is not None else None
    if archive is not None:
        copy_results(run_dir / RESULTS_DIR, archive)
    if request.partial:
        return CollectResult(CollectStatus.PARTIAL, run, manifest, files, archive)
    context.catalog.mark_collected(run.run_id, str(archive or run_dir / RESULTS_DIR))
    return CollectResult(CollectStatus.COLLECTED, context.reread(), manifest, files, archive)
