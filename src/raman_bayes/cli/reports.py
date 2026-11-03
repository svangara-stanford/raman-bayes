"""Text-mode reports for commands whose results take more than a line (DESIGN.md §10).

Under ``--json`` the same results go out as envelope data instead; these functions only shape
what a person reads.
"""

from collections.abc import Sequence
from pathlib import Path

from raman_bayes.app.ingest import IngestPreview, ScanOutcome, ScanStatus
from raman_bayes.domain.dataset import AxisSpec
from raman_bayes.domain.qc import QCCounts
from raman_bayes.domain.run import RunPlan

__all__ = ["ingest_report", "plan_report", "qc_summary"]

_TASKS_SHOWN = 20  # a Slurm plan may have hundreds; the JSON envelope lists them all


def ingest_report(outcomes: Sequence[ScanOutcome]) -> str:
    """Describe each scan folder: what was ingested, skipped, previewed or refused."""
    return "\n".join(_outcome_lines(o) for o in outcomes)


def _outcome_lines(outcome: ScanOutcome) -> str:
    manifest = outcome.manifest
    if outcome.status is ScanStatus.INGESTED and manifest is not None:
        qc = f"; {qc_summary(outcome.qc)}" if outcome.qc is not None else ""
        return (
            f"{outcome.folder.name}: ingested {manifest.n_spectra} spectra as "
            f"{manifest.dataset_id} into {outcome.store_uri}{qc}"
        )
    if outcome.status is ScanStatus.EXISTS and manifest is not None:
        return f"{outcome.folder.name}: already ingested as {manifest.dataset_id}; nothing to do"
    if outcome.preview is not None:
        return _preview(outcome.folder.name, outcome.preview)
    return f"{outcome.folder.name}: not ingested"


def qc_summary(counts: QCCounts) -> str:
    """Describe the QC counts in one line, naming only the flags that occur."""
    if counts.n_flagged == 0:
        return "no QC flags"
    present = ", ".join(f"{name} {n}" for name, n in counts.flags.items() if n)
    return f"QC flags on {counts.n_flagged} of {counts.n_spectra} spectra: {present}"


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


def plan_report(plan: RunPlan, run_plan: Path) -> str:
    """Describe a planned run: its identity, engine, backend and seed, and its first 20 tasks."""
    spec = plan.priors.spec
    lines = [
        f"planned run {plan.run_id}: {len(plan.tasks)} tasks of up to {plan.spectra_per_task} "
        f"spectra, {plan.n_spectra} spectra in all",
        f"  dataset {plan.dataset_id}; priors {spec.name} {spec.version} ({spec.status}); "
        f"engine {plan.engine} {plan.engine_version}; backend {plan.backend}; seed {plan.seed}",
        f"  plan written to {run_plan}",
        *(
            f"  task {t.task_index}: spectra {t.spec_start}-{t.spec_stop - 1}"
            for t in plan.tasks[:_TASKS_SHOWN]
        ),
    ]
    if len(plan.tasks) > _TASKS_SHOWN:
        lines.append(f"  ... and {len(plan.tasks) - _TASKS_SHOWN} more tasks")
    return "\n".join(lines)
