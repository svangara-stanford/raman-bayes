"""Arrow tables of fit results in the explicit §8.2 schemas (DESIGN.md §8.2, §15 #6).

The fits table has a row per fit; engine diagnostics become float32 ``diag_*`` columns. The
peaks table has a row for every peak of the recipe whatever the fit's outcome, so a peak an
engine reports absent, does not report, or that a failed or skipped fit never reached, is a
row of nulls, never of zeros.
"""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import pyarrow as pa

from raman_bayes.domain.priors import PriorsSpec
from raman_bayes.domain.results import (
    DIAG_PREFIX,
    PEAKS_SCHEMA,
    SUMMARY_QUANTITIES,
    SUMMARY_STATS,
    PeakPosterior,
    WindowFitResult,
    fits_schema,
)

__all__ = ["FitRecord", "SpectrumPlace", "fits_table", "peaks_table"]


@dataclass(frozen=True)
class FitRecord:
    """One fit and the ``noise_sd`` of the spectrum it fitted; None when it was not prepared."""

    result: WindowFitResult
    noise_sd: float | None


@dataclass(frozen=True)
class SpectrumPlace:
    """A spectrum's grid indices and coordinates in µm."""

    ix: int
    iy: int
    iz: int
    x_um: float
    y_um: float
    z_um: float


def fits_table(run_id: str, records: Sequence[FitRecord]) -> pa.Table:
    """Return one FITS_SCHEMA row per record, in order, plus a column per diagnostic key."""
    keys = sorted({key for record in records for key in record.result.diagnostics})
    rows = [
        {
            "run_id": run_id,
            "spectrum_id": r.result.spectrum_id,
            "window_id": r.result.window_id,
            "status": r.result.status,
            "runtime_s": r.result.runtime_s,
            "noise_sd": r.noise_sd,
            "engine": r.result.engine,
            "engine_version": r.result.engine_version,
            "error": r.result.error,
            **{f"{DIAG_PREFIX}{key}": r.result.diagnostics.get(key) for key in keys},
        }
        for r in records
    ]
    return pa.Table.from_pylist(rows, schema=fits_schema(keys))


def peaks_table(
    run_id: str,
    dataset_id: str,
    priors: PriorsSpec,
    results: Sequence[WindowFitResult],
    places: Mapping[int, SpectrumPlace],
) -> pa.Table:
    """Return one PEAKS_SCHEMA row per result and recipe peak, in result then recipe order.

    ``places`` gives each result's spectrum coordinates. Peaks are matched by id; a peak the
    result does not report has null posteriors.
    """
    peaks = {w.id: w.peaks for w in priors.windows}
    end_member = {peak: m.id for m in priors.end_members for peak in m.peaks}
    rows = []
    for result in results:
        reported = {p.peak_id: p for p in result.peaks}
        fixed = {
            "run_id": run_id,
            "dataset_id": dataset_id,
            "spectrum_id": result.spectrum_id,
            **asdict(places[result.spectrum_id]),
            "window_id": result.window_id,
            "fit_status": result.status,
        }
        for peak in peaks[result.window_id]:
            posterior = reported.get(peak.id)
            rows.append(
                fixed
                | {"end_member_id": end_member.get(peak.id), "peak_id": peak.id}
                | _posterior_columns(posterior)
            )
    return pa.Table.from_pylist(rows, schema=PEAKS_SCHEMA)


def _posterior_columns(posterior: PeakPosterior | None) -> dict[str, Any]:
    columns: dict[str, Any] = {
        "presence_prob": posterior.presence_prob if posterior else None,
    }
    for quantity in SUMMARY_QUANTITIES:
        summary = getattr(posterior, quantity) if posterior else None
        for stat in SUMMARY_STATS:
            columns[f"{quantity}_{stat}"] = getattr(summary, stat) if summary else None
    for quantity in ("center", "fwhm"):
        summary = getattr(posterior, quantity) if posterior else None
        columns[f"{quantity}_at_bound_frac"] = summary.at_bound_frac if summary else None
    return columns
