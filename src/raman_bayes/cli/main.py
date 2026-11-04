"""The ``raman`` command tree (DESIGN.md §10).

Help is plain Click (``rich_markup_mode=None``): Typer forces a Rich terminal when
GITHUB_ACTIONS, FORCE_COLOR or PY_COLORS is set, which would put ANSI into piped ``--help``.
Command output goes through ``output.make_console``, which is colour-free off a TTY.
"""

from collections.abc import Callable
from datetime import timedelta
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, NoReturn, TypeVar

import typer

from raman_bayes.app import catalog_views, priors_csv, validate
from raman_bayes.app.collect import CollectRequest, CollectStatus, collect_run
from raman_bayes.app.doctor import CheckStatus, DoctorReport, run_doctor
from raman_bayes.app.export import ExportRequest, LegacyOptions, export_run
from raman_bayes.app.ingest import IngestRequest, ScanOutcome, ScanStatus
from raman_bayes.app.ingest import ingest as ingest_scans
from raman_bayes.app.plan import RUN_PLAN_FILE, plan_run
from raman_bayes.app.run_task import RunTaskStatus, TaskOutcome, TaskRequest
from raman_bayes.app.run_task import run_task as run_one_task
from raman_bayes.app.runs_import import ImportStatus, import_run
from raman_bayes.app.status import RunView, refresh_run, watch_run
from raman_bayes.app.submit import (
    CancelStatus,
    SubmitRequest,
    SubmitResult,
    SubmitStatus,
    cancel_run,
    run_local,
    submit_run,
)
from raman_bayes.cli.output import CliEnvelope, ExitCode, emit
from raman_bayes.cli.reports import (
    collect_report,
    doctor_report,
    ingest_report,
    plan_report,
    qc_summary,
    status_report,
    submit_report,
    task_report,
)
from raman_bayes.config.settings import LoadedSettings, load_settings, requested_profile
from raman_bayes.domain.errors import (
    ConfigError,
    DatasetStoreError,
    InputError,
    RamanBayesError,
    TaskLockedError,
)
from raman_bayes.domain.run import SEED_LIMIT, RunConfig, RunPlan, RunStatus
from raman_bayes.logging import setup_logging

JsonOpt = Annotated[
    bool,
    typer.Option("--json", help="Write one JSON envelope to stdout."),
]
ProfileOpt = Annotated[
    str | None,
    typer.Option("--profile", help="Profile name or path (default: $RAMAN_PROFILE, else local)."),
]
RunArg = Annotated[str, typer.Argument(metavar="RUN", help="Run id.")]
_T = TypeVar("_T")


def _typer(help_: str) -> typer.Typer:
    return typer.Typer(
        help=help_,
        no_args_is_help=True,
        rich_markup_mode=None,
        pretty_exceptions_enable=False,
    )


app = _typer("Bayesian Raman peak-detection pipeline.")
datasets_app = _typer("Inspect ingested datasets.")
priors_app = _typer("Convert priors between YAML and the legacy CSV format.")
runs_app = _typer("Inspect and register runs.")
app.add_typer(datasets_app, name="datasets")
app.add_typer(priors_app, name="priors")
app.add_typer(runs_app, name="runs")


def _invalid(
    command: str, message: str, errors: tuple[str, ...], warnings: tuple[str, ...]
) -> CliEnvelope:
    return CliEnvelope(
        command=command,
        exit_code=ExitCode.VALIDATION,
        ok=False,
        message=message,
        warnings=warnings,
        errors=errors,
    )


def _finish(envelope: CliEnvelope, as_json: bool) -> NoReturn:
    emit(envelope, as_json=as_json)
    raise typer.Exit(code=envelope.exit_code)


def _succeed(
    command: str,
    message: str,
    data: dict[str, Any],
    as_json: bool,
    warnings: tuple[str, ...] = (),
) -> NoReturn:
    envelope = CliEnvelope(
        command=command,
        exit_code=ExitCode.OK,
        ok=True,
        message=message,
        data=data,
        warnings=warnings,
    )
    _finish(envelope, as_json)


def _infrastructure(command: str, problem: str, error: Exception, as_json: bool) -> NoReturn:
    envelope = CliEnvelope(
        command=command,
        exit_code=ExitCode.BACKEND,
        ok=False,
        message=f"raman {command}: {problem}",
        errors=(str(error),),
    )
    _finish(envelope, as_json)


def _settings(command: str, profile: str | None, as_json: bool) -> LoadedSettings:
    try:
        return load_settings(profile)
    except ConfigError as exc:
        _finish(_invalid(command, "cannot load settings", (str(exc),), ()), as_json)


class ExportFormat(StrEnum):
    """``export --format`` choices."""

    LEGACY_CSV = "legacy-csv"
    CSV = "csv"


class HeightSource(StrEnum):
    """Posterior written in the legacy CSV 'height' column."""

    AREA = "area"
    HEIGHT = "height"


class Coords(StrEnum):
    """Coordinates written in the legacy CSV: grid indices or micrometres."""

    INDEX = "index"
    UM = "um"


# ---------------------------------------------------------------- datasets


@app.command()
def ingest(
    path: Annotated[Path, typer.Argument(help="Scan folder, or a parent of several.")],
    sample: Annotated[str | None, typer.Option(help="Sample name for the manifest.")] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print parsed sections, grid and file check.")
    ] = False,
    archive: Annotated[
        bool, typer.Option("--archive", help="Tar raw files to archive_root.")
    ] = False,
    allow_partial_grid: Annotated[
        bool, typer.Option("--allow-partial-grid", help="Record missing positions in grid_mask.")
    ] = False,
    allow_axis_mismatch: Annotated[
        bool, typer.Option("--allow-axis-mismatch", help="Flag, don't refuse, axis mismatches.")
    ] = False,
    as_job: Annotated[bool, typer.Option("--as-job", help="Run as a Slurm job.")] = False,
    profile: ProfileOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Pack a Nanofinder export into dataset.zarr."""
    loaded = _settings("ingest", profile, as_json)
    refused = _slurm_only({"--archive": archive, "--as-job": as_job}, loaded)
    if refused:
        _finish(_invalid("ingest", "cannot ingest with these options", refused, ()), as_json)
    request = IngestRequest(
        path=path,
        sample=sample,
        allow_partial_grid=allow_partial_grid,
        allow_axis_mismatch=allow_axis_mismatch,
        dry_run=dry_run,
    )
    outcomes = ingest_scans(request, loaded.settings.paths)
    _finish(_ingest_envelope(outcomes), as_json)


def _slurm_only(flags: dict[str, bool], loaded: LoadedSettings) -> tuple[str, ...]:
    # Archiving and running as a job need the Slurm backend, which is not available yet.
    backend = loaded.settings.backend
    reason = (
        f"needs the slurm_array backend; profile {loaded.settings.profile!r} uses {backend}"
        if backend == "local"
        else "is not supported yet"
    )
    return tuple(f"{flag} {reason}" for flag, given in flags.items() if given)


def _ingest_envelope(outcomes: tuple[ScanOutcome, ...]) -> CliEnvelope:
    statuses = {o.status for o in outcomes}
    if ScanStatus.FAILED in statuses:
        code = ExitCode.VALIDATION
    elif statuses & {ScanStatus.INGESTED, ScanStatus.DRY_RUN}:
        code = ExitCode.OK
    else:
        code = ExitCode.NOTHING_TO_DO
    return CliEnvelope(
        command="ingest",
        exit_code=code,
        ok=code is ExitCode.OK,
        message=ingest_report(outcomes),
        data={"scans": [o.as_json() for o in outcomes]},
        warnings=tuple(w for o in outcomes for w in o.warnings),
        errors=tuple(e for o in outcomes for e in o.errors),
    )


@datasets_app.command("list")
def datasets_list(profile: ProfileOpt = None, as_json: JsonOpt = False) -> None:
    """List datasets in the catalog."""
    catalog = _settings("datasets list", profile, as_json).settings.paths.catalog
    rows = catalog_views.list_datasets(catalog)
    lines = [f"{r.dataset_id}  {r.n_spectra} spectra  {r.sample or '-'}" for r in rows]
    data = {
        "catalog": str(catalog),
        "datasets": [{k: v for k, v in r.as_json().items() if k != "manifest"} for r in rows],
    }
    _succeed("datasets list", "\n".join(lines) or f"no datasets in {catalog}", data, as_json)


@datasets_app.command("show")
def datasets_show(
    dataset: Annotated[str, typer.Argument(metavar="DATASET", help="Dataset id.")],
    profile: ProfileOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Show one dataset's manifest and QC flag counts."""
    catalog = _settings("datasets show", profile, as_json).settings.paths.catalog
    view = catalog_views.show_dataset(catalog, dataset)
    if view is None:
        error = f"no dataset {dataset!r} in {catalog}"
        _finish(_invalid("datasets show", "unknown dataset", (error,), ()), as_json)
    message = view.row.manifest.model_dump_json(indent=2)
    if view.qc is not None:
        message = f"{message}\n{qc_summary(view.qc)}"
    data = view.row.as_json() | {"qc": view.qc.model_dump(mode="json") if view.qc else None}
    _succeed("datasets show", message, data, as_json, view.warnings)


# ---------------------------------------------------------------- priors


@app.command("validate-priors")
def validate_priors(
    file: Annotated[Path, typer.Argument(help="priors.yaml to validate.")],
    profile: ProfileOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Validate a priors recipe; exit 2 on error."""
    _settings(
        "validate-priors", profile, as_json
    )  # every command resolves --profile, so a bad one fails alike
    result = validate.validate_priors_file(file)
    if result.spec is None:
        message = f"{file}: invalid priors ({len(result.errors)} error(s))"
        _finish(_invalid("validate-priors", message, result.errors, result.warnings), as_json)
    spec = result.spec.spec
    envelope = CliEnvelope(
        command="validate-priors",
        exit_code=ExitCode.OK,
        ok=True,
        message=f"{file}: valid {spec.name} {spec.version} ({spec.status}), "
        f"priors_hash {result.priors_hash}",
        data={
            "priors_hash": result.priors_hash,
            "name": spec.name,
            "version": spec.version,
            "status": spec.status,
        },
        warnings=result.warnings,
    )
    _finish(envelope, as_json)


@priors_app.command("import-csv")
def priors_import_csv(
    file: Annotated[Path, typer.Argument(help="Legacy priors CSV.")],
    window_id: Annotated[str, typer.Option("--window-id", help="Id of the new window.")],
    range_cm1: Annotated[
        tuple[float, float],
        typer.Option("--range", metavar="LO HI", help="Window range in cm^-1."),
    ],
    profile: ProfileOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Convert a legacy CSV into a priors YAML window fragment."""
    _settings(
        "priors import-csv", profile, as_json
    )  # every command resolves --profile, so a bad one fails alike
    result = priors_csv.import_window(file, window_id, range_cm1)
    if result.window is None or result.fragment is None:
        message = f"{file}: cannot import ({len(result.errors)} error(s))"
        _finish(_invalid("priors import-csv", message, result.errors, result.warnings), as_json)
    envelope = CliEnvelope(
        command="priors import-csv",
        exit_code=ExitCode.OK,
        ok=True,
        message=result.fragment.rstrip("\n"),
        data={"window": result.window.model_dump(mode="json"), "fragment": result.fragment},
        warnings=result.warnings,
    )
    _finish(envelope, as_json)


@priors_app.command("export-csv")
def priors_export_csv(
    file: Annotated[Path, typer.Argument(help="priors.yaml to export from.")],
    window_id: Annotated[str, typer.Option("--window-id", help="Window to export.")],
    profile: ProfileOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Write one window of a priors recipe in the legacy CSV format."""
    _settings(
        "priors export-csv", profile, as_json
    )  # every command resolves --profile, so a bad one fails alike
    result = priors_csv.export_window(file, window_id)
    if result.csv is None:
        message = f"{file}: cannot export window {window_id!r} ({len(result.errors)} error(s))"
        _finish(_invalid("priors export-csv", message, result.errors, result.warnings), as_json)
    envelope = CliEnvelope(
        command="priors export-csv",
        exit_code=ExitCode.OK,
        ok=True,
        message=result.csv.rstrip("\n"),
        data={"window_id": window_id, "csv": result.csv},
        warnings=result.warnings,
    )
    _finish(envelope, as_json)


# ---------------------------------------------------------------- runs


@app.command()
def plan(
    dataset: Annotated[str, typer.Option("--dataset", help="Dataset id.")],
    priors: Annotated[Path, typer.Option("--priors", help="priors.yaml.")],
    engine: Annotated[
        str | None, typer.Option("--engine", help="Engine name (default: the profile's).")
    ] = None,
    spectra_per_task: Annotated[
        int | None,
        typer.Option("--spectra-per-task", min=1, help="Spectra per task, used as given."),
    ] = None,
    seed: Annotated[
        int | None,
        typer.Option("--seed", min=0, max=SEED_LIMIT - 1, help="Run seed (default: random)."),
    ] = None,
    profile: ProfileOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Write run_plan.json and record the run; print its tasks."""
    loaded = _settings("plan", profile, as_json)
    config = RunConfig(
        dataset_id=dataset,
        priors_path=priors,
        engine=engine or loaded.settings.engine.name,
        spectra_per_task=spectra_per_task,
        seed=seed,
        profile=requested_profile(profile),
    )
    try:
        result = plan_run(config, loaded)
    except DatasetStoreError as exc:
        _infrastructure("plan", "storage problem", exc, as_json)
    if result.plan is None or result.run_dir is None:
        _finish(_invalid("plan", "cannot plan this run", result.errors, result.warnings), as_json)
    envelope = CliEnvelope(
        command="plan",
        exit_code=ExitCode.OK,
        ok=True,
        message=plan_report(result.plan, result.run_dir / RUN_PLAN_FILE),
        data=_plan_data(result.plan, result.run_dir),
        warnings=result.warnings,
    )
    _finish(envelope, as_json)


def _plan_data(planned: RunPlan, run_dir: Path) -> dict[str, Any]:
    return {
        "run_id": planned.run_id,
        "run_dir": str(run_dir),
        "run_plan": str(run_dir / RUN_PLAN_FILE),
        "dataset_id": planned.dataset_id,
        "priors_hash": planned.priors_hash,
        "engine": planned.engine,
        "engine_version": planned.engine_version,
        "backend": planned.backend,
        "seed": planned.seed,
        "spectra_per_task": planned.spectra_per_task,
        "n_tasks": len(planned.tasks),
        "tasks": [t.model_dump() for t in planned.tasks],
    }


@app.command()
def submit(
    run: RunArg,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Render, don't submit.")] = False,
    confirm: Annotated[bool, typer.Option("--confirm", help="Submit without asking.")] = False,
    resubmit_failed: Annotated[
        bool, typer.Option("--resubmit-failed", help="Resubmit tasks with no _DONE marker.")
    ] = False,
    profile: ProfileOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Submit a planned run to its backend; on the local backend, wait for it."""
    if dry_run and confirm:
        refused = ("--dry-run and --confirm cannot be used together",)
        _finish(_invalid("submit", "cannot submit", refused, ()), as_json)
    loaded = _settings("submit", profile, as_json)
    request = SubmitRequest(run, dry_run=dry_run, resubmit_failed=resubmit_failed)
    result = _driver("submit", lambda: submit_run(request, loaded), as_json)
    _finish(_submit_envelope("submit", result, resubmit_failed, {}), as_json)


def _driver(command: str, call: Callable[[], _T], as_json: bool) -> _T:
    try:
        return call()
    except InputError as exc:
        _finish(_invalid(command, f"cannot {command} this run", (str(exc),), ()), as_json)
    except TaskLockedError as exc:
        envelope = CliEnvelope(
            command=command,
            exit_code=ExitCode.NOTHING_TO_DO,
            ok=False,
            message=f"raman {command}: another process is running this run's tasks",
            errors=(str(exc),),
        )
        _finish(envelope, as_json)
    except (RamanBayesError, OSError) as exc:
        _infrastructure(command, "storage or backend problem", exc, as_json)


def _submit_envelope(
    command: str, result: SubmitResult, resubmit: bool, extra: dict[str, Any]
) -> CliEnvelope:
    stopped = result.run.status in (RunStatus.FAILED, RunStatus.CANCELLED)
    if result.status is SubmitStatus.NOTHING_TO_DO:
        code = ExitCode.NOTHING_TO_DO
    else:
        code = ExitCode.BACKEND if stopped else ExitCode.OK
    data: dict[str, Any] = {
        **extra,
        "run_id": result.run.run_id,
        "status": result.status.value,
        "run_status": result.run.status.value,
        "record": result.record.model_dump(mode="json") if result.record else None,
        "tasks": [t.as_json() for t in result.tasks],
        "failures": [{"task_index": i, "error": e} for i, e in sorted(result.failures.items())],
    }
    return CliEnvelope(
        command=command,
        exit_code=code,
        ok=code is ExitCode.OK,
        message=submit_report(result, resubmit=resubmit),
        data=data,
    )


@app.command("run")
def run_(
    dataset: Annotated[str, typer.Option("--dataset", help="Dataset id.")],
    priors: Annotated[Path, typer.Option("--priors", help="priors.yaml.")],
    profile: ProfileOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Plan, submit and wait (local backend only)."""
    loaded = _settings("run", profile, as_json)
    config = RunConfig(
        dataset_id=dataset,
        priors_path=priors,
        engine=loaded.settings.engine.name,
        profile=requested_profile(profile),
    )
    planned, result = _driver("run", lambda: run_local(config, loaded), as_json)
    if result is None or planned.plan is None:
        _finish(_invalid("run", "cannot plan this run", planned.errors, planned.warnings), as_json)
    extra = {"run_dir": str(planned.run_dir), "n_tasks": len(planned.plan.tasks)}
    _finish(_submit_envelope("run", result, False, extra), as_json)


@app.command("run-task")
def run_task(
    run: Annotated[str, typer.Option("--run", help="Run id.")],
    task: Annotated[int, typer.Option("--task", min=0, help="Task index.")],
    workers: Annotated[
        int | None, typer.Option("--workers", min=1, help="Worker processes for this task.")
    ] = None,
    profile: ProfileOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Run one task of a plan (invoked by backends)."""
    setup_logging()
    loaded = _settings("run-task", profile, as_json)
    request = TaskRequest(
        run_dir=loaded.settings.paths.runs_root / run,
        task_index=task,
        workers=workers or 1,
        lock_stale=timedelta(minutes=loaded.settings.lock_stale_minutes),
    )
    try:
        outcome = run_one_task(request)
    except InputError as exc:
        _finish(_invalid("run-task", "cannot run this task", (str(exc),), ()), as_json)
    except (RamanBayesError, OSError) as exc:
        _infrastructure("run-task", "task stopped", exc, as_json)
    _finish(_task_envelope(run, outcome), as_json)


def _task_envelope(run: str, outcome: TaskOutcome) -> CliEnvelope:
    code = ExitCode.OK if outcome.status is RunTaskStatus.DONE else ExitCode.NOTHING_TO_DO
    task = outcome.task
    data = {
        "run_id": run,
        "task_index": task.task_index,
        "status": outcome.status.value,
        "spec_start": task.spec_start,
        "spec_stop": task.spec_stop,
        "n_parts_written": outcome.n_parts_written,
        "n_parts_kept": outcome.n_parts_kept,
        "done": outcome.marker.model_dump(mode="json") if outcome.marker else None,
        "lock": outcome.holder.model_dump(mode="json") if outcome.holder else None,
    }
    return CliEnvelope(
        command="run-task",
        exit_code=code,
        ok=code is ExitCode.OK,
        message=task_report(run, outcome),
        data=data,
    )


@app.command()
def status(
    run: RunArg,
    watch: Annotated[bool, typer.Option("--watch", help="Refresh until done.")] = False,
    failures: Annotated[bool, typer.Option("--failures", help="List failed tasks.")] = False,
    profile: ProfileOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Show task states from _DONE markers and sacct."""
    loaded = _settings("status", profile, as_json)
    view = _driver("status", lambda: (watch_run if watch else refresh_run)(run, loaded), as_json)
    _finish(_status_envelope(view, failures), as_json)


def _status_envelope(view: RunView, failures: bool) -> CliEnvelope:
    data = {
        "run": view.run.as_json(),
        "entered": [s.value for s in view.entered],
        "tasks": [t.as_json() for t in view.tasks],
        "failures": [t.as_json() for t in view.failures],
    }
    return CliEnvelope(
        command="status",
        exit_code=ExitCode.OK,
        ok=True,
        message=status_report(view, failures=failures),
        data=data,
    )


@app.command()
def cancel(
    run: RunArg,
    confirm: Annotated[bool, typer.Option("--confirm", help="Required to cancel.")] = False,
    profile: ProfileOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Cancel a run."""
    if not confirm:
        refused = ("cancelling stops the run for good; add --confirm to cancel it",)
        _finish(_invalid("cancel", "not cancelled", refused, ()), as_json)
    loaded = _settings("cancel", profile, as_json)
    result = _driver("cancel", lambda: cancel_run(run, loaded), as_json)
    if result.status is CancelStatus.NOTHING_TO_DO:
        code, message = ExitCode.NOTHING_TO_DO, f"run {run} is already {result.run.status.value}"
    else:
        code, message = ExitCode.OK, f"run {run} cancelled"
    envelope = CliEnvelope(
        command="cancel",
        exit_code=code,
        ok=code is ExitCode.OK,
        message=message,
        data={"run": result.run.as_json()},
    )
    _finish(envelope, as_json)


@app.command()
def collect(
    run: RunArg,
    partial: Annotated[bool, typer.Option("--partial", help="Collect an incomplete run.")] = False,
    as_job: Annotated[bool, typer.Option("--as-job", help="Run as a Slurm job.")] = False,
    profile: ProfileOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Merge shards into results/*.parquet."""
    loaded = _settings("collect", profile, as_json)
    refused = _slurm_only({"--as-job": as_job}, loaded)
    if refused:
        _finish(_invalid("collect", "cannot collect with these options", refused, ()), as_json)
    request = CollectRequest(run, partial=partial)
    result = _driver("collect", lambda: collect_run(request, loaded), as_json)
    code = ExitCode.NOTHING_TO_DO if result.status is CollectStatus.NOTHING_TO_DO else ExitCode.OK
    data = {
        "run_id": result.run.run_id,
        "status": result.status.value,
        "run_status": result.run.status.value,
        "results_dir": str(result.results_dir) if result.results_dir else None,
        "files": [str(f) for f in result.files],
        "archive_dir": str(result.archive_dir) if result.archive_dir else None,
        "manifest": result.manifest.model_dump(mode="json") if result.manifest else None,
    }
    envelope = CliEnvelope(
        command="collect",
        exit_code=code,
        ok=code is ExitCode.OK,
        message=collect_report(result),
        data=data,
    )
    _finish(envelope, as_json)


@app.command()
def export(
    run: RunArg,
    export_format: Annotated[ExportFormat, typer.Option("--format", help="Output format.")],
    window: Annotated[
        str | None, typer.Option("--window", help="Window id (default: all).")
    ] = None,
    min_presence: Annotated[
        float, typer.Option("--min-presence", min=0.0, max=1.0, help="Presence threshold.")
    ] = 0.90,
    height_source: Annotated[
        HeightSource, typer.Option("--height-source", help="Column written as height.")
    ] = HeightSource.AREA,
    absent_fill: Annotated[
        float, typer.Option("--absent-fill", help="Value for absent peaks.")
    ] = 0.0,
    coords: Annotated[Coords, typer.Option("--coords", help="Grid indices or µm.")] = Coords.INDEX,
    profile: ProfileOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Export results as legacy CSV or flattened CSV."""
    loaded = _settings("export", profile, as_json)
    legacy = LegacyOptions(
        min_presence=min_presence,
        height_source=height_source.value,
        absent_fill=absent_fill,
        coords=coords.value,
    )
    request = ExportRequest(run, export_format.value, window, legacy)
    result = _driver("export", lambda: export_run(request, loaded), as_json)
    files = [str(f) for f in result.files]
    _succeed("export", "\n".join(files), {"run_id": run, "files": files}, as_json)


@runs_app.command("list")
def runs_list(profile: ProfileOpt = None, as_json: JsonOpt = False) -> None:
    """List runs in the catalog."""
    catalog = _settings("runs list", profile, as_json).settings.paths.catalog
    rows = catalog_views.list_runs(catalog)
    lines = [f"{r.run_id}  {r.status.value}  {r.dataset_id}  {r.engine}" for r in rows]
    data = {"catalog": str(catalog), "runs": [r.as_json() for r in rows]}
    _succeed("runs list", "\n".join(lines) or f"no runs in {catalog}", data, as_json)


@runs_app.command("show")
def runs_show(run: RunArg, profile: ProfileOpt = None, as_json: JsonOpt = False) -> None:
    """Show one run."""
    catalog = _settings("runs show", profile, as_json).settings.paths.catalog
    found = catalog_views.find_run(catalog, run)
    if found is None:
        error = f"no run {run!r} in {catalog}"
        _finish(_invalid("runs show", "unknown run", (error,), ()), as_json)
    row, tasks = found
    task_lines = [f"  task {t.task_index}: {t.state.status.value}" for t in tasks]
    message = "\n".join([f"{row.run_id}  {row.status.value}  {row.dataset_id}", *task_lines])
    data = {"run": row.as_json(), "tasks": [t.as_json() for t in tasks]}
    _succeed("runs show", message, data, as_json)


@runs_app.command("import")
def runs_import(
    directory: Annotated[Path, typer.Argument(metavar="DIR", help="Run folder copied in.")],
    profile: ProfileOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Register a run folder copied from elsewhere."""
    loaded = _settings("runs import", profile, as_json)
    result = _driver("runs import", lambda: import_run(directory, loaded), as_json)
    imported = result.status is ImportStatus.IMPORTED
    code = ExitCode.OK if imported else ExitCode.NOTHING_TO_DO
    state = result.run.status.value
    message = (
        f"registered run {result.run.run_id} ({state}) from {directory}"
        if imported
        else f"run {result.run.run_id} is already in the catalog ({state}); nothing to do"
    )
    data = {
        "run_id": result.run.run_id,
        "status": result.status.value,
        "run_status": state,
        "run": result.run.as_json(),
    }
    envelope = CliEnvelope(
        command="runs import", exit_code=code, ok=imported, message=message, data=data
    )
    _finish(envelope, as_json)


# ---------------------------------------------------------------- environment


@app.command()
def doctor(profile: ProfileOpt = None, as_json: JsonOpt = False) -> None:
    """Check environment, paths, ${VAR} resolution, pixi_bin, quotas and Slurm limits."""
    report = run_doctor(_settings("doctor", profile, as_json))
    code = ExitCode.BACKEND if report.failed else ExitCode.OK
    envelope = CliEnvelope(
        command="doctor",
        exit_code=code,
        ok=code is ExitCode.OK,
        message=doctor_report(report),
        data=_doctor_data(report),
        warnings=_checks(report, CheckStatus.WARN),
        errors=_checks(report, CheckStatus.FAIL),
    )
    _finish(envelope, as_json)


def _checks(report: DoctorReport, status: CheckStatus) -> tuple[str, ...]:
    return tuple(f"{c.name}: {c.detail}" for c in report.checks if c.status is status)


def _doctor_data(report: DoctorReport) -> dict[str, Any]:
    settings = report.settings
    return {
        "profile": settings.profile,
        "profile_source": report.source,
        "backend": settings.backend,
        "paths": settings.paths.model_dump(mode="json"),
        "engines": list(report.engines),
        "checks": [
            {"name": c.name, "status": c.status.value, "detail": c.detail} for c in report.checks
        ],
    }
