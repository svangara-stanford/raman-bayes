"""Merge a run's shard parts into ``results/`` (DESIGN.md §8.1, §8.2).

Each task's parts are first checked against its ``_DONE.json`` (the same files, the same
hash), then read with their explicit schemas: peaks parts must equal PEAKS_SCHEMA, fits parts
FITS_SCHEMA plus float32 ``diag_*`` columns, which are unioned across parts. A fit or peak may
appear only once. Both tables are sorted by ``spectrum_id`` (stably, so each spectrum keeps its
recipe order) and written in row groups of 64 Ki rows, with ``run_manifest.json`` and
``schema_version`` as key-value metadata.
"""

import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from raman_bayes.domain.errors import DuplicateRowsError, PartSchemaError, ShardMismatchError
from raman_bayes.domain.markers import DoneMarker
from raman_bayes.domain.results import DIAG_PREFIX, PEAKS_SCHEMA, fits_schema, is_fits_schema
from raman_bayes.storage.atomic import write_atomically, write_text_atomically
from raman_bayes.storage.shards import summarize_parts, task_dir

__all__ = [
    "FITS_FILE",
    "MANIFEST_FILE",
    "PEAKS_FILE",
    "RESULTS_DIR",
    "RESULTS_SCHEMA_VERSION",
    "ROW_GROUP_ROWS",
    "MergedTables",
    "copy_results",
    "merge_tasks",
    "write_results",
]

RESULTS_DIR = "results"
PEAKS_FILE = "peaks.parquet"
FITS_FILE = "fits.parquet"
MANIFEST_FILE = "run_manifest.json"
RESULTS_SCHEMA_VERSION = "1.0"
ROW_GROUP_ROWS = 65536
_FITS_KEY = ("spectrum_id", "window_id")
_PEAKS_KEY = ("spectrum_id", "window_id", "peak_id")


@dataclass(frozen=True)
class MergedTables:
    """The merged fits and peaks of a run, sorted by spectrum id."""

    fits: pa.Table
    peaks: pa.Table


def merge_tasks(run_dir: Path, markers: Mapping[int, DoneMarker]) -> MergedTables:
    """Merge the parts of the tasks in ``markers`` (task index to done marker; not empty).

    Raises ShardMismatchError when a task folder's parts are not those its marker recorded,
    PartSchemaError for a part without its explicit schema, and DuplicateRowsError when a fit
    or peak appears twice.
    """
    fits: list[pa.Table] = []
    peaks: list[pa.Table] = []
    for index in sorted(markers):
        folder = task_dir(run_dir, index)
        _check_parts(folder, markers[index])
        for name in markers[index].parts:
            table = pq.read_table(folder / name)
            schema = table.schema.remove_metadata()
            if name.startswith("peaks-"):
                if not schema.equals(PEAKS_SCHEMA):
                    raise PartSchemaError(folder / name, "peaks")
                peaks.append(table)
            elif is_fits_schema(schema):
                fits.append(table)
            else:
                raise PartSchemaError(folder / name, "fits")
    merged = MergedTables(fits=_sorted(_union_fits(fits)), peaks=_sorted(pa.concat_tables(peaks)))
    _check_unique(merged.fits, _FITS_KEY, "fit")
    _check_unique(merged.peaks, _PEAKS_KEY, "peak")
    return merged


def write_results(results_dir: Path, tables: MergedTables, manifest_json: str) -> tuple[Path, ...]:
    """Write ``peaks.parquet``, ``fits.parquet`` and ``run_manifest.json``, each atomically.

    Returns the three paths, sorted by name.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        b"run_manifest.json": manifest_json.encode("utf-8"),
        b"schema_version": RESULTS_SCHEMA_VERSION.encode("utf-8"),
    }
    for name, table in ((PEAKS_FILE, tables.peaks), (FITS_FILE, tables.fits)):
        _write_table(results_dir / name, table.replace_schema_metadata(metadata))
    write_text_atomically(results_dir / MANIFEST_FILE, manifest_json)
    return tuple(results_dir / n for n in sorted((PEAKS_FILE, FITS_FILE, MANIFEST_FILE)))


def copy_results(results_dir: Path, to: Path) -> None:
    """Copy the result files of ``results_dir`` into ``to``, each atomically."""
    to.mkdir(parents=True, exist_ok=True)
    for name in sorted((PEAKS_FILE, FITS_FILE, MANIFEST_FILE)):
        with (results_dir / name).open("rb") as source:
            write_atomically(to / name, partial(shutil.copyfileobj, source))


def _write_table(path: Path, table: pa.Table) -> None:
    write_atomically(
        path, lambda stream: pq.write_table(table, stream, row_group_size=ROW_GROUP_ROWS)
    )


def _check_parts(folder: Path, marker: DoneMarker) -> None:
    found = summarize_parts(folder)
    if found.parts != marker.parts:
        raise ShardMismatchError(
            folder, f"holds parts {list(found.parts)}, not {list(marker.parts)}"
        )
    if found.shard_hash != marker.shard_hash:
        raise ShardMismatchError(folder, "its parts changed since the task finished")


def _union_fits(tables: Sequence[pa.Table]) -> pa.Table:
    keys = {
        n.removeprefix(DIAG_PREFIX)
        for t in tables
        for n in t.column_names
        if n.startswith(DIAG_PREFIX)
    }
    schema = fits_schema(keys)
    aligned = [
        pa.Table.from_arrays(
            [
                t.column(f.name) if f.name in t.column_names else pa.nulls(t.num_rows, f.type)
                for f in schema
            ],
            schema=schema,
        )
        for t in tables
    ]
    return pa.concat_tables(aligned)


def _sorted(table: pa.Table) -> pa.Table:
    # Parts carry their own dictionaries; unify them so the sort's take has one per column.
    unified = table.unify_dictionaries().combine_chunks()
    return unified.sort_by([("spectrum_id", "ascending")])


def _check_unique(table: pa.Table, key: Sequence[str], kind: str) -> None:
    seen: set[tuple[object, ...]] = set()
    for row in zip(*(table.column(k).to_pylist() for k in key), strict=True):
        if row in seen:
            raise DuplicateRowsError(kind, dict(zip(key, row, strict=True)))
        seen.add(row)
