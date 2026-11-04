"""Text-mode reports for commands whose results take more than a line (DESIGN.md §10).

Under ``--json`` the same results go out as envelope data instead; these functions only shape
what a person reads.
"""

from collections.abc import Sequence
from pathlib import Path

from raman_bayes.app.collect import CollectResult, CollectStatus
from raman_bayes.app.doctor import DoctorReport
from raman_bayes.app.ingest import IngestPreview, ScanOutcome, ScanStatus
from raman_bayes.app.run_task import RunTaskStatus, TaskOutcome
from raman_bayes.app.status import RunView
from raman_bayes.app.submit import SubmitResult, SubmitStatus
from raman_bayes.domain.dataset import AxisSpec
from raman_bayes.domain.qc import QCCounts
from raman_bayes.domain.run import RunPlan, RunStatus, TaskState, TaskStatus

__all__ = [
    "collect_report",
    "doctor_report",
    "ingest_report",
    "plan_report",
    "qc_summary",
    "status_report",
    "submit_report",
    "task_report",
]

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


def task_report(run: str, outcome: TaskOutcome) -> str:
    """Describe what run-task did with one task: its fits, or who holds it."""
    head = f"task {outcome.task.task_index} of run {run}"
    marker, holder = outcome.marker, outcome.holder
    if marker is None:  # only a task held by another process has no done marker
        who = (
            f"host {holder.host}, pid {holder.pid}, since {holder.started_at:%Y-%m-%d %H:%M:%S %Z}"
            if holder is not None
            else "its lock is being written"
        )
        return f"{head}: already running ({who}); nothing to do"
    fits = (
        f"{marker.n_fits} fits ({marker.n_ok} ok, {marker.n_failed} failed, "
        f"{marker.n_skipped_qc} skipped_qc)"
    )
    if outcome.status is RunTaskStatus.ALREADY_DONE:
        return f"{head}: already done, {fits}; nothing to do"
    return (
        f"{head}: done, {fits}; {outcome.n_parts_written} sub-batch(es) written, "
        f"{outcome.n_parts_kept} kept"
    )


def submit_report(result: SubmitResult, *, resubmit: bool) -> str:
    """Describe a submit: the preview, why there was nothing to do, or how the run ended."""
    run = result.run
    head = f"run {run.run_id}"
    if result.status is SubmitStatus.NOTHING_TO_DO:
        if run.status in (RunStatus.COMPLETE, RunStatus.COLLECTED):
            return f"{head} is {run.status.value}; nothing to submit"
        if resubmit:
            return f"{head} is {run.status.value} and has no stopped tasks to resubmit"
        return (
            f"{head} is already submitted ({run.status.value}); see raman status, or retry "
            "stopped tasks with --resubmit-failed"
        )
    record = result.record
    assert record is not None
    tasks = ", ".join(str(i) for i in record.task_indices)
    if result.status is SubmitStatus.DRY_RUN:
        return f"{head}: dry run, would submit attempt {record.attempt} of tasks {tasks}"
    lines = [
        f"{head}: attempt {record.attempt} of tasks {tasks} finished; run is {run.status.value}"
    ]
    lines += [f"  task {i} stopped: {e}" for i, e in sorted(result.failures.items())]
    return "\n".join(lines)


def status_report(view: RunView, *, failures: bool) -> str:
    """Describe a run and its first 20 tasks; with ``failures``, why each stopped task stopped."""
    run = view.run
    moved = f" (now {', '.join(s.value for s in view.entered)})" if view.entered else ""
    lines = [f"run {run.run_id}: {run.status.value}{moved}; dataset {run.dataset_id}"]
    lines += [f"  task {t.task_index}: {_task_line(t.state)}" for t in view.tasks[:_TASKS_SHOWN]]
    if len(view.tasks) > _TASKS_SHOWN:
        lines.append(f"  ... and {len(view.tasks) - _TASKS_SHOWN} more tasks")
    if failures:
        lines.append(f"failures: {len(view.failures)}")
        lines += [
            f"  task {t.task_index}: {t.state.status.value}: {t.state.error or 'no error recorded'}"
            for t in view.failures
        ]
    return "\n".join(lines)


def _task_line(state: TaskState) -> str:
    attempts = f"{state.attempts} attempt{'' if state.attempts == 1 else 's'}"
    if state.status is TaskStatus.DONE:
        return f"done, {state.n_ok} ok, {state.n_failed} failed ({attempts})"
    return f"{state.status.value} ({attempts})"


def collect_report(result: CollectResult) -> str:
    """Describe a collect: where the results went, and whether they cover the whole run."""
    run = result.run
    manifest = result.manifest
    if result.status is CollectStatus.NOTHING_TO_DO or manifest is None:
        return f"run {run.run_id} is already {run.status.value}; nothing to do"
    scope = (
        f"{len(manifest.shard_hashes)} of {manifest.n_tasks} tasks (partial; run is still "
        f"{run.status.value})"
        if manifest.partial
        else f"all {manifest.n_tasks} tasks; run is {run.status.value}"
    )
    lines = [
        f"run {run.run_id}: collected {scope}",
        f"  {manifest.n_spectra} spectra, {manifest.n_failed} failed fits, "
        f"{manifest.n_skipped_qc} skipped_qc",
        f"  results in {result.results_dir}",
    ]
    if result.archive_dir is not None:
        lines.append(f"  copied to {result.archive_dir}")
    return "\n".join(lines)


def doctor_report(report: DoctorReport) -> str:
    """List every check with its status, after the profile and backend it ran with."""
    settings = report.settings
    lines = [f"profile {settings.profile} ({report.source}), backend {settings.backend}"]
    lines += [f"{c.status.value:<5} {c.name}: {c.detail}" for c in report.checks]
    return "\n".join(lines)
