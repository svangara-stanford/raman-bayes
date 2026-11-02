"""The ``raman`` command tree (DESIGN.md §10). Commands not built yet are stubs that exit 3.

Help is plain Click (``rich_markup_mode=None``): Typer forces a Rich terminal when
GITHUB_ACTIONS, FORCE_COLOR or PY_COLORS is set, which would put ANSI into piped ``--help``.
Command output goes through ``output.make_console``, which is colour-free off a TTY.
"""

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer

from raman_bayes.app import catalog_views, priors_csv, validate
from raman_bayes.cli.output import CliEnvelope, ExitCode, emit
from raman_bayes.config.settings import LoadedSettings, load_settings
from raman_bayes.domain.errors import ConfigError

JsonOpt = Annotated[
    bool,
    typer.Option("--json", help="Write one JSON envelope to stdout."),
]
ProfileOpt = Annotated[
    str | None,
    typer.Option("--profile", help="Profile name or path (default: $RAMAN_PROFILE, else local)."),
]
RunArg = Annotated[str, typer.Argument(metavar="RUN", help="Run id.")]


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


def _stub(command: str, as_json: bool) -> NoReturn:
    envelope = CliEnvelope(
        command=command,
        exit_code=ExitCode.NOTHING_TO_DO,
        ok=False,
        message=f"raman {command}: not implemented yet",
    )
    emit(envelope, as_json=as_json)
    raise typer.Exit(code=envelope.exit_code)


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


def _succeed(command: str, message: str, data: dict[str, Any], as_json: bool) -> NoReturn:
    envelope = CliEnvelope(
        command=command, exit_code=ExitCode.OK, ok=True, message=message, data=data
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
    _stub("ingest", as_json)


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
    """Show one dataset's manifest."""
    catalog = _settings("datasets show", profile, as_json).settings.paths.catalog
    row = catalog_views.find_dataset(catalog, dataset)
    if row is None:
        error = f"no dataset {dataset!r} in {catalog}"
        _finish(_invalid("datasets show", "unknown dataset", (error,), ()), as_json)
    message = row.manifest.model_dump_json(indent=2)
    _succeed("datasets show", message, row.as_json(), as_json)


# ---------------------------------------------------------------- priors


@app.command("validate-priors")
def validate_priors(
    file: Annotated[Path, typer.Argument(help="priors.yaml to validate.")],
    profile: ProfileOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Validate a priors recipe; exit 2 on error."""
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
    engine: Annotated[str, typer.Option("--engine", help="Engine name.")] = "mock",
    spectra_per_task: Annotated[
        int | None,
        typer.Option("--spectra-per-task", min=1, help="Override the benchmarked task size."),
    ] = None,
    profile: ProfileOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Write run_plan.json; print tasks, core-hour estimate and sbatch preview."""
    _stub("plan", as_json)


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
    """Submit a planned run to its backend."""
    _stub("submit", as_json)


@app.command("run")
def run_(
    dataset: Annotated[str, typer.Option("--dataset", help="Dataset id.")],
    priors: Annotated[Path, typer.Option("--priors", help="priors.yaml.")],
    profile: ProfileOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Plan, submit and wait (local backend only)."""
    _stub("run", as_json)


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
    _stub("run-task", as_json)


@app.command()
def status(
    run: RunArg,
    watch: Annotated[bool, typer.Option("--watch", help="Refresh until done.")] = False,
    failures: Annotated[bool, typer.Option("--failures", help="List failed tasks.")] = False,
    profile: ProfileOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Show task states from _DONE markers and sacct."""
    _stub("status", as_json)


@app.command()
def cancel(
    run: RunArg,
    confirm: Annotated[bool, typer.Option("--confirm", help="Required to cancel.")] = False,
    profile: ProfileOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Cancel a run."""
    _stub("cancel", as_json)


@app.command()
def collect(
    run: RunArg,
    partial: Annotated[bool, typer.Option("--partial", help="Collect an incomplete run.")] = False,
    as_job: Annotated[bool, typer.Option("--as-job", help="Run as a Slurm job.")] = False,
    profile: ProfileOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Merge shards into results/*.parquet."""
    _stub("collect", as_json)


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
    _stub("export", as_json)


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
    _stub("runs import", as_json)


# ---------------------------------------------------------------- environment


@app.command()
def doctor(profile: ProfileOpt = None, as_json: JsonOpt = False) -> None:
    """Check environment, paths, ${VAR} resolution, pixi_bin, quotas and Slurm limits."""
    _stub("doctor", as_json)
