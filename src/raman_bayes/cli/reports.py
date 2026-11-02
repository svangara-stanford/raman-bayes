"""Text-mode reports for commands whose results take more than a line (DESIGN.md §10).

Under ``--json`` the same results go out as envelope data instead; these functions only shape
what a person reads.
"""

from collections.abc import Sequence

from raman_bayes.app.ingest import IngestPreview, ScanOutcome, ScanStatus
from raman_bayes.domain.dataset import AxisSpec

__all__ = ["ingest_report"]


def ingest_report(outcomes: Sequence[ScanOutcome]) -> str:
    """One paragraph per scan folder: what was ingested, skipped, previewed or refused."""
    return "\n".join(_outcome_lines(o) for o in outcomes)


def _outcome_lines(outcome: ScanOutcome) -> str:
    manifest = outcome.manifest
    if outcome.status is ScanStatus.INGESTED and manifest is not None:
        return (
            f"{outcome.folder.name}: ingested {manifest.n_spectra} spectra as "
            f"{manifest.dataset_id} into {outcome.store_uri}"
        )
    if outcome.status is ScanStatus.EXISTS and manifest is not None:
        return f"{outcome.folder.name}: already ingested as {manifest.dataset_id}; nothing to do"
    if outcome.preview is not None:
        return _preview(outcome.folder.name, outcome.preview)
    return f"{outcome.folder}: not ingested"


def _preview(name: str, preview: IngestPreview) -> str:
    info = preview.info
    grid, acq, instrument = info.grid, info.acq, info.instrument
    lines = [
        f"{name}: dry run, nothing written",
        f"  header: acquired {info.acquired_at:%Y-%m-%d %H:%M:%S}, {info.duration_s:g} s; "
        f"{instrument.system}, {instrument.detector}, {instrument.positioner}",
        f"  channel: exposure_s {acq.exposure_s:g}, n_accum {acq.n_accum}, "
        f"ccd_temp_c {acq.ccd_temp_c:g}, central_pixel {acq.central_pixel}, "
        f"head_model {acq.head_model}",
        f"  dimensions: n_wn {info.n_wn}, n_series {info.n_series}",
        f"  grid: {grid.nx} x {grid.ny} x {grid.nz}; x {_axis(grid.x)}; y {_axis(grid.y)}; "
        f"z {_axis(grid.z)}; allow_partial {str(grid.allow_partial).lower()}",
        f"  files: {preview.n_files} of {preview.n_positions} grid positions have a spectrum",
        f"  first {len(preview.first_files)} files:",
        *(f"    {file}" for file in preview.first_files),
    ]
    return "\n".join(lines)


def _axis(axis: AxisSpec) -> str:
    if axis.step_um == 0:
        return f"at {axis.start_um:g} um"
    return f"{axis.start_um:g} to {axis.stop_um:g} um, step {axis.step_um:g}"
