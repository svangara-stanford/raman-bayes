"""Read-only views of the catalog for ``datasets list/show`` and ``runs list/show`` (§8.5)."""

from dataclasses import dataclass
from pathlib import Path

from raman_bayes.domain.errors import DatasetStoreError
from raman_bayes.domain.qc import QCCounts, qc_counts
from raman_bayes.storage.catalog import Catalog, DatasetRow, RunRow, TaskRow
from raman_bayes.storage.zarr_dataset import open_dataset

__all__ = ["DatasetView", "find_run", "list_datasets", "list_runs", "show_dataset"]


@dataclass(frozen=True)
class DatasetView:
    """One dataset for ``datasets show``: its catalog row and the QC counts from its store.

    ``qc`` is None, with a warning saying why, when the store cannot be read from here (it may
    live on another machine).
    """

    row: DatasetRow
    qc: QCCounts | None
    warnings: tuple[str, ...] = ()


def list_datasets(catalog: Path) -> list[DatasetRow]:
    """Return every registered dataset; empty when the catalog does not exist yet."""
    return Catalog(catalog).list_datasets()


def show_dataset(catalog: Path, dataset_id: str) -> DatasetView | None:
    """Return one dataset with its QC counts, or None when it is not registered."""
    row = Catalog(catalog).get_dataset(dataset_id)
    if row is None:
        return None
    try:
        flags = open_dataset(Path(row.store_uri)).qc_flags()
    except (DatasetStoreError, OSError) as exc:
        return DatasetView(row, None, (f"QC flags not available: {exc}",))
    return DatasetView(row, qc_counts(flags))


def list_runs(catalog: Path) -> list[RunRow]:
    """Return every run; empty when the catalog does not exist yet."""
    return Catalog(catalog).list_runs()


def find_run(catalog: Path, run_id: str) -> tuple[RunRow, list[TaskRow]] | None:
    """Return one run and its tasks, or None when there is no such run."""
    cat = Catalog(catalog)
    run = cat.get_run(run_id)
    return None if run is None else (run, cat.tasks(run_id))
