"""Read-only views of the catalog for ``datasets list/show`` and ``runs list/show`` (§8.5)."""

from pathlib import Path

from raman_bayes.storage.catalog import Catalog, DatasetRow, RunRow, TaskRow

__all__ = ["find_dataset", "find_run", "list_datasets", "list_runs"]


def list_datasets(catalog: Path) -> list[DatasetRow]:
    """Return every registered dataset; empty when the catalog does not exist yet."""
    return Catalog(catalog).list_datasets()


def find_dataset(catalog: Path, dataset_id: str) -> DatasetRow | None:
    """Return one dataset, or None when it is not registered."""
    return Catalog(catalog).get_dataset(dataset_id)


def list_runs(catalog: Path) -> list[RunRow]:
    """Return every run; empty when the catalog does not exist yet."""
    return Catalog(catalog).list_runs()


def find_run(catalog: Path, run_id: str) -> tuple[RunRow, list[TaskRow]] | None:
    """Return one run and its tasks, or None when there is no such run."""
    cat = Catalog(catalog)
    run = cat.get_run(run_id)
    return None if run is None else (run, cat.tasks(run_id))
