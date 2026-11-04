"""Collected peaks as CSV (DESIGN.md §8.4): the legacy per-window layout and the flat table.

The legacy layout has one row per spectrum: its grid position (or µm), then for each peak of
the window, in recipe order, ``<peak> center``, ``<peak> height``, ``<peak> width`` and
``<peak> confidence``. Its defaults reproduce the old pipeline: the area median under
"height", absent peaks filled with 0 and grid indices. A peak counts as present when its
presence is at least ``min_presence`` and it has posterior summaries; otherwise its center,
height and width are ``absent_fill``, while its confidence is still the presence probability
(``absent_fill`` only when the fit gave none). The flat table is every peaks column, with
nulls as empty fields. float32 values are written in their shortest exact form, float64 in
Python's ``repr``; files use LF line endings and no BOM.
"""

import csv
import io
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pyarrow as pa

from raman_bayes.domain.priors import Window

__all__ = ["LegacyOptions", "flat_csv", "legacy_csv", "legacy_header"]

_QUANTITIES = ("center", "height", "width", "confidence")


@dataclass(frozen=True)
class LegacyOptions:
    """``--min-presence``, ``--height-source``, ``--absent-fill`` and ``--coords``."""

    min_presence: float = 0.90
    height_source: Literal["area", "height"] = "area"
    absent_fill: float = 0.0
    coords: Literal["index", "um"] = "index"


def legacy_header(window: Window) -> list[str]:
    """Return ``x, y`` and the four columns of each of the window's peaks, in recipe order."""
    return ["x", "y", *(f"{p.id} {q}" for p in window.peaks for q in _QUANTITIES)]


def legacy_csv(peaks: pa.Table, window: Window, options: LegacyOptions) -> str:
    """Return the legacy CSV of ``window`` from a peaks table sorted by spectrum id."""
    rows = [r for r in peaks.to_pylist() if r["window_id"] == window.id]
    by_spectrum: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_spectrum.setdefault(row["spectrum_id"], {})[row["peak_id"]] = row
    lines = [legacy_header(window)]
    for found in by_spectrum.values():
        first = next(iter(found.values()))
        cells = _coords(first, options.coords)
        for peak in window.peaks:
            cells.extend(_peak_cells(found.get(peak.id), options))
        lines.append(cells)
    return _csv(lines)


def flat_csv(peaks: pa.Table) -> str:
    """Return every column of ``peaks`` as CSV, nulls as empty fields."""
    kinds = [f.type for f in peaks.schema]
    lines: list[list[str]] = [peaks.schema.names]
    for row in peaks.to_pylist():
        lines.append([_cell(v, k) for v, k in zip(row.values(), kinds, strict=True)])
    return _csv(lines)


def _coords(row: dict[str, Any], coords: str) -> list[str]:
    if coords == "um":
        return [repr(row["x_um"]), repr(row["y_um"])]
    return [str(row["ix"]), str(row["iy"])]


def _peak_cells(row: dict[str, Any] | None, options: LegacyOptions) -> list[str]:
    fill = format(options.absent_fill, "g")
    if row is None:
        return [fill] * len(_QUANTITIES)
    presence = row["presence_prob"]
    confidence = _float32(presence) if presence is not None else fill
    if presence is None or presence < options.min_presence or row["center_median"] is None:
        return [fill, fill, fill, confidence]
    values = (row["center_median"], row[f"{options.height_source}_median"], row["fwhm_median"])
    return [*(_float32(v) if v is not None else fill for v in values), confidence]


def _cell(value: Any, kind: pa.DataType) -> str:
    if value is None:
        return ""
    if kind == pa.float32():
        return _float32(value)
    if kind == pa.float64():
        return repr(float(value))
    return str(value)


def _float32(value: float) -> str:
    return str(np.float32(value))


def _csv(lines: Sequence[Sequence[str]]) -> str:
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="\n").writerows(lines)
    return buffer.getvalue()
