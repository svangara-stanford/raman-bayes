"""One task's shard folder, ``shards/task=NNNNN/`` (DESIGN.md §7.2, §8.1).

Sub-batch k is two Parquet parts, ``fits-<k>.parquet`` and then ``peaks-<k>.parquet``, each
written atomically with its explicit schema. A sub-batch counts as written only when both parts
read back with their schemas and hold the expected rows; anything else is written again, so a
crash between or inside the two files costs at most that sub-batch. ``_DONE.json`` records the
finished task, with a hash over its parts that collect checks.
"""

import hashlib
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import ValidationError

from raman_bayes.domain.errors import InvalidDoneMarkerError
from raman_bayes.domain.markers import DoneMarker
from raman_bayes.domain.results import PEAKS_SCHEMA, is_fits_schema
from raman_bayes.storage.atomic import write_atomically, write_text_atomically

__all__ = [
    "DONE_FILE",
    "SHARDS_DIR",
    "PartPaths",
    "PartRows",
    "PartSummary",
    "part_is_valid",
    "part_paths",
    "read_done",
    "summarize_parts",
    "task_dir",
    "write_done",
    "write_part",
]

SHARDS_DIR = "shards"
DONE_FILE = "_DONE.json"
_PART_GLOBS = ("fits-*.parquet", "peaks-*.parquet")


def task_dir(run_dir: Path, task_index: int) -> Path:
    """Return ``run_dir/shards/task=NNNNN``, the only folder task ``task_index`` writes."""
    return run_dir / SHARDS_DIR / f"task={task_index:05d}"


@dataclass(frozen=True)
class PartPaths:
    """The two files of one sub-batch."""

    fits: Path
    peaks: Path


def part_paths(folder: Path, index: int) -> PartPaths:
    """Return the parts of sub-batch ``index`` in the task folder ``folder``."""
    return PartPaths(folder / f"fits-{index:03d}.parquet", folder / f"peaks-{index:03d}.parquet")


@dataclass(frozen=True)
class PartRows:
    """What a sub-batch's parts must hold for its spectra.

    One fits row per spectrum and window, one peaks row per spectrum and recipe peak, where
    ``n_peaks`` counts the peaks of every window.
    """

    spectrum_ids: tuple[int, ...]
    n_windows: int
    n_peaks: int


def write_part(folder: Path, index: int, fits: pa.Table, peaks: pa.Table) -> PartPaths:
    """Write sub-batch ``index``: ``fits`` and then ``peaks``, each atomically."""
    paths = part_paths(folder, index)
    write_atomically(paths.fits, lambda stream: pq.write_table(fits, stream))
    write_atomically(paths.peaks, lambda stream: pq.write_table(peaks, stream))
    return paths


def part_is_valid(folder: Path, index: int, rows: PartRows) -> bool:
    """Return whether both parts of sub-batch ``index`` exist and hold exactly ``rows``."""
    paths = part_paths(folder, index)
    n = len(rows.spectrum_ids)
    return _holds(paths.fits, is_fits_schema, n * rows.n_windows, rows) and _holds(
        paths.peaks, PEAKS_SCHEMA.equals, n * rows.n_peaks, rows
    )


def _holds(path: Path, schema_ok: Callable[[pa.Schema], bool], n_rows: int, rows: PartRows) -> bool:
    try:
        parquet = pq.ParquetFile(path)
        if not schema_ok(parquet.schema_arrow.remove_metadata()):
            return False
        if parquet.metadata.num_rows != n_rows:
            return False
        ids = parquet.read(columns=["spectrum_id"]).column(0)
    except (pa.ArrowException, OSError):
        return False
    return set(ids.to_pylist()) == set(rows.spectrum_ids)


@dataclass(frozen=True)
class PartSummary:
    """The parts in a task folder, in name order, their row counts and their hash."""

    parts: tuple[str, ...]
    n_fits: int
    n_ok: int
    n_failed: int
    n_skipped_qc: int
    n_peaks: int
    shard_hash: str


def summarize_parts(folder: Path) -> PartSummary:
    """Count the fits rows by status and the peaks rows of every part in ``folder``.

    ``shard_hash`` is the SHA-256 of, for each part in name order, its name, a newline, its
    own SHA-256 hex digest and a newline. Staging files are not parts and are ignored.
    """
    parts = sorted(p for pattern in _PART_GLOBS for p in folder.glob(pattern))
    statuses: Counter[str] = Counter()
    n_peaks = 0
    digest = hashlib.sha256()
    for path in parts:
        with path.open("rb") as stream:
            file_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        digest.update(f"{path.name}\n{file_hash}\n".encode())
        if path.name.startswith("peaks-"):
            n_peaks += pq.read_metadata(path).num_rows
        else:
            statuses.update(pq.read_table(path, columns=["status"]).column(0).to_pylist())
    return PartSummary(
        parts=tuple(p.name for p in parts),
        n_fits=statuses.total(),
        n_ok=statuses["ok"],
        n_failed=statuses["failed"],
        n_skipped_qc=statuses["skipped_qc"],
        n_peaks=n_peaks,
        shard_hash=digest.hexdigest(),
    )


def write_done(folder: Path, marker: DoneMarker) -> None:
    """Write ``folder/_DONE.json`` atomically."""
    write_text_atomically(folder / DONE_FILE, marker.model_dump_json(indent=2))


def read_done(folder: Path) -> DoneMarker | None:
    """Return ``folder/_DONE.json``, or None when the task has not finished.

    Raises InvalidDoneMarkerError when the file is there but is not a done marker.
    """
    path = folder / DONE_FILE
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    try:
        return DoneMarker.model_validate_json(text)
    except ValidationError as exc:
        raise InvalidDoneMarkerError(path, str(exc)) from exc
