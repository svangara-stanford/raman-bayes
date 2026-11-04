"""Fit results (DESIGN.md §6) and the explicit Parquet schemas they are written with (§8.2).

Every shard and merged file is written with PEAKS_SCHEMA / FITS_SCHEMA so an all-null column
stays float32 instead of pyarrow's inferred ``null`` type. FITS_SCHEMA holds the fixed columns
only; merge appends the engine-defined ``diag_*`` columns (float32). File-level key-value
metadata (``run_manifest.json``, ``schema_version``) is attached at write time, not here.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

import pyarrow as pa

__all__ = [
    "DIAG_PREFIX",
    "FITS_SCHEMA",
    "PEAKS_SCHEMA",
    "SUMMARY_QUANTITIES",
    "SUMMARY_STATS",
    "PeakPosterior",
    "PosteriorSummary",
    "WindowFitResult",
    "fits_schema",
    "is_fits_schema",
]


@dataclass(frozen=True)
class PosteriorSummary:
    """Summary statistics of one posterior quantity.

    ``at_bound_frac`` is the fraction of draws at a prior bound, when the engine reports it.
    """

    mean: float
    median: float
    sd: float
    q025: float
    q975: float
    at_bound_frac: float | None = None


@dataclass(frozen=True)
class PeakPosterior:
    """One peak's posterior. The summaries are None when the peak is absent."""

    peak_id: str
    presence_prob: float
    center: PosteriorSummary | None
    fwhm: PosteriorSummary | None
    area: PosteriorSummary | None
    height: PosteriorSummary | None
    eta: PosteriorSummary | None


@dataclass(frozen=True)
class WindowFitResult:
    """One fit of one spectrum in one window. ``diagnostics`` keys are engine-defined."""

    spectrum_id: int
    window_id: str
    status: Literal["ok", "failed", "skipped_qc"]
    peaks: tuple[PeakPosterior, ...]
    diagnostics: Mapping[str, float]
    runtime_s: float
    engine: str
    engine_version: str
    error: str | None = None


_DICT_STR = pa.dictionary(pa.int32(), pa.string())
SUMMARY_QUANTITIES = ("center", "fwhm", "area", "height", "eta")
SUMMARY_STATS = ("mean", "median", "sd", "q025", "q975")
DIAG_PREFIX = "diag_"

PEAKS_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("run_id", _DICT_STR, nullable=False),
        pa.field("dataset_id", _DICT_STR, nullable=False),
        pa.field("spectrum_id", pa.uint32(), nullable=False),
        pa.field("ix", pa.int32(), nullable=False),
        pa.field("iy", pa.int32(), nullable=False),
        pa.field("iz", pa.int32(), nullable=False),
        pa.field("x_um", pa.float64(), nullable=False),
        pa.field("y_um", pa.float64(), nullable=False),
        pa.field("z_um", pa.float64(), nullable=False),
        pa.field("window_id", _DICT_STR, nullable=False),
        pa.field("end_member_id", _DICT_STR),
        pa.field("peak_id", _DICT_STR, nullable=False),
        pa.field("presence_prob", pa.float32()),
        *(pa.field(f"{q}_{s}", pa.float32()) for q in SUMMARY_QUANTITIES for s in SUMMARY_STATS),
        pa.field("center_at_bound_frac", pa.float32()),
        pa.field("fwhm_at_bound_frac", pa.float32()),
        pa.field("fit_status", _DICT_STR, nullable=False),
    ]
)

FITS_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("run_id", _DICT_STR, nullable=False),
        pa.field("spectrum_id", pa.uint32(), nullable=False),
        pa.field("window_id", _DICT_STR, nullable=False),
        pa.field("status", _DICT_STR, nullable=False),
        pa.field("runtime_s", pa.float64(), nullable=False),
        pa.field("noise_sd", pa.float32()),
        pa.field("engine", _DICT_STR, nullable=False),
        pa.field("engine_version", _DICT_STR, nullable=False),
        pa.field("error", pa.string()),
    ]
)


def fits_schema(diagnostics: Iterable[str]) -> pa.Schema:
    """Return FITS_SCHEMA followed by one nullable float32 ``diag_<key>`` column per key, sorted."""
    extra = [pa.field(f"{DIAG_PREFIX}{key}", pa.float32()) for key in sorted(set(diagnostics))]
    return pa.schema([*FITS_SCHEMA, *extra])


def is_fits_schema(schema: pa.Schema) -> bool:
    """Return whether ``schema`` is FITS_SCHEMA followed only by float32 ``diag_*`` columns."""
    n = len(FITS_SCHEMA)
    if len(schema) < n or not pa.schema(list(schema)[:n]).equals(FITS_SCHEMA):
        return False
    return all(
        f.name.startswith(DIAG_PREFIX) and f.type == pa.float32() and f.nullable
        for f in list(schema)[n:]
    )
