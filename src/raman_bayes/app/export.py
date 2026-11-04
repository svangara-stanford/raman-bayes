"""Export a collected run's peaks (DESIGN.md §8.4) to ``runs/<run_id>/exports/``.

Only a collected run is exported; its peaks come from the catalog's ``result_uri`` (the
archive copy when there is one) and the window and peak order from its plan. Legacy CSV gives
``legacy-<window>.csv`` per window; the flat CSV gives ``peaks.csv``, or
``peaks-<window>.csv`` for one window. Each file is written atomically.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from raman_bayes.config.settings import LoadedSettings
from raman_bayes.domain.errors import (
    ExportRefusedError,
    ResultsMissingError,
    RunNotInCatalogError,
    UnknownWindowError,
)
from raman_bayes.domain.priors import Window
from raman_bayes.domain.run import RunStatus
from raman_bayes.io.results_csv import LegacyOptions, flat_csv, legacy_csv
from raman_bayes.storage.atomic import write_text_atomically
from raman_bayes.storage.catalog import Catalog, RunRow
from raman_bayes.storage.merge import PEAKS_FILE
from raman_bayes.storage.run_files import read_plan

__all__ = ["EXPORTS_DIR", "ExportRequest", "ExportResult", "LegacyOptions", "export_run"]

EXPORTS_DIR = "exports"


@dataclass(frozen=True)
class ExportRequest:
    """``raman export RUN --format F [--window W] ...``; ``legacy`` applies to legacy-csv."""

    run_id: str
    format: Literal["legacy-csv", "csv"]
    window: str | None = None
    legacy: LegacyOptions = field(default_factory=LegacyOptions)


@dataclass(frozen=True)
class ExportResult:
    """The run exported and the files written, in window order."""

    run: RunRow
    files: tuple[Path, ...]


def export_run(request: ExportRequest, loaded: LoadedSettings) -> ExportResult:
    """Write the export files of a collected run and return them.

    Raises RunNotInCatalogError, ExportRefusedError for a run that is not collected,
    UnknownWindowError, and ResultsMissingError when its result files are gone.
    """
    paths = loaded.settings.paths
    run = Catalog(paths.catalog).get_run(request.run_id)
    if run is None:
        raise RunNotInCatalogError(request.run_id, paths.catalog)
    if run.status is not RunStatus.COLLECTED or run.result_uri is None:
        raise ExportRefusedError(run.run_id, run.status)
    run_dir = paths.runs_root / run.run_id
    windows = _windows(read_plan(run_dir).priors.windows, request.window)
    source = Path(run.result_uri) / PEAKS_FILE
    try:
        peaks = pq.read_table(source)
    except FileNotFoundError:
        raise ResultsMissingError(source) from None
    out = run_dir / EXPORTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for name, text in _render(request, peaks, windows):
        write_text_atomically(out / name, text)
        files.append(out / name)
    return ExportResult(run, tuple(files))


def _windows(windows: tuple[Window, ...], chosen: str | None) -> tuple[Window, ...]:
    if chosen is None:
        return windows
    picked = tuple(w for w in windows if w.id == chosen)
    if not picked:
        raise UnknownWindowError(chosen, [w.id for w in windows])
    return picked


def _render(
    request: ExportRequest, peaks: pa.Table, windows: tuple[Window, ...]
) -> list[tuple[str, str]]:
    if request.format == "legacy-csv":
        return [(f"legacy-{w.id}.csv", legacy_csv(peaks, w, request.legacy)) for w in windows]
    if request.window is None:
        return [("peaks.csv", flat_csv(peaks))]
    chosen = pc.equal(peaks.column("window_id").cast(pa.string()), request.window)
    return [(f"peaks-{request.window}.csv", flat_csv(peaks.filter(chosen)))]
