"""``raman runs import DIR`` (DESIGN.md §8.5, C14): register a run folder copied in place."""

import shutil
from collections.abc import Callable
from pathlib import Path

from typer.testing import CliRunner

from raman_bayes.app.run_task import TaskRequest, run_task
from raman_bayes.cli.main import app
from raman_bayes.cli.output import CliEnvelope, ExitCode
from raman_bayes.config.settings import PathsSettings
from raman_bayes.domain.priors_rules import priors_hash
from raman_bayes.domain.run import RunStatus, TaskStatus, config_hash
from raman_bayes.domain.run_manifest import build_manifest
from raman_bayes.storage.catalog import Catalog
from raman_bayes.storage.merge import merge_tasks, write_results
from raman_bayes.storage.run_files import read_plan
from raman_bayes.storage.shards import read_done, task_dir

WritePlan = Callable[..., Path]
runner = CliRunner()


def _import(directory: Path) -> tuple[int, CliEnvelope]:
    result = runner.invoke(app, ["runs", "import", str(directory), "--json"])
    (line,) = result.stdout.splitlines()
    return result.exit_code, CliEnvelope.model_validate_json(line)


def _run_all(run_dir: Path) -> None:
    for task in read_plan(run_dir).tasks:
        run_task(TaskRequest(run_dir, task.task_index))


def test_runs_import_registers_run(
    mini_store: Path, write_plan: WritePlan, raman_paths: PathsSettings
) -> None:
    run_dir = write_plan(mini_store)
    _run_all(run_dir)
    code, env = _import(run_dir)
    assert code == ExitCode.OK, env.errors
    plan = read_plan(run_dir)
    catalog = Catalog(raman_paths.catalog)
    run = catalog.get_run(plan.run_id)
    assert run is not None
    assert run.status is RunStatus.COMPLETE
    assert (run.dataset_id, run.priors_hash, run.config_hash, run.seed) == (
        plan.dataset_id,
        priors_hash(plan.priors),
        config_hash(plan),
        plan.seed,
    )
    assert (run.engine, run.engine_version, run.backend, run.profile) == (
        "mock",
        "0.1.0",
        "local",
        "local",
    )
    tasks = catalog.tasks(plan.run_id)
    assert [t.state.status for t in tasks] == [TaskStatus.DONE, TaskStatus.DONE]
    done = read_done(task_dir(run_dir, 0))
    assert done is not None
    assert (tasks[0].state.n_ok, tasks[0].state.n_failed) == (done.n_ok, done.n_failed)
    assert env.data is not None
    assert (env.data["run_id"], env.data["run_status"]) == (plan.run_id, "complete")


def test_runs_import_of_a_collected_folder_is_collected(
    mini_store: Path, write_plan: WritePlan, raman_paths: PathsSettings
) -> None:
    run_dir = write_plan(mini_store)
    _run_all(run_dir)
    plan = read_plan(run_dir)
    markers = {t.task_index: read_done(task_dir(run_dir, t.task_index)) for t in plan.tasks}
    done = {i: m for i, m in markers.items() if m is not None}
    tables = merge_tasks(run_dir, done)
    write_results(
        run_dir / "results", tables, build_manifest(plan, done, tables.fits).model_dump_json()
    )
    assert _import(run_dir)[0] == ExitCode.OK
    run = Catalog(raman_paths.catalog).get_run(plan.run_id)
    assert run is not None
    assert (run.status, run.result_uri) == (RunStatus.COLLECTED, str(run_dir / "results"))


def test_runs_import_of_unfinished_folders(
    mini_store: Path, write_plan: WritePlan, raman_paths: PathsSettings
) -> None:
    untouched = write_plan(mini_store)
    half = write_plan(mini_store)
    run_task(TaskRequest(half, 0))
    for run_dir, expected in ((untouched, RunStatus.PLANNED), (half, RunStatus.FAILED)):
        assert _import(run_dir)[0] == ExitCode.OK
        run = Catalog(raman_paths.catalog).get_run(run_dir.name)
        assert run is not None
        assert run.status is expected


def test_runs_import_existing_exits_3(mini_store: Path, write_plan: WritePlan) -> None:
    run_dir = write_plan(mini_store)
    assert _import(run_dir)[0] == ExitCode.OK
    code, env = _import(run_dir)
    assert code == ExitCode.NOTHING_TO_DO
    assert "already" in (env.message or "")


def test_runs_import_invalid_dir_exits_2(
    mini_store: Path, write_plan: WritePlan, raman_paths: PathsSettings, tmp_path: Path
) -> None:
    empty = raman_paths.runs_root / "0190b3d4-0000-7000-8000-000000000000"
    empty.mkdir(parents=True)
    code, env = _import(empty)
    assert code == ExitCode.VALIDATION
    assert "no run plan" in env.errors[0]

    run_dir = write_plan(mini_store)
    (run_dir / "run_plan.json").write_text("{}", encoding="utf-8")
    code, env = _import(run_dir)
    assert code == ExitCode.VALIDATION
    assert "not a valid run plan" in env.errors[0]

    good = write_plan(mini_store)
    outside = tmp_path / "elsewhere" / good.name
    shutil.copytree(good, outside)
    code, env = _import(outside)
    assert code == ExitCode.VALIDATION
    assert str(raman_paths.runs_root / good.name) in env.errors[0]
    assert Catalog(raman_paths.catalog).get_run(good.name) is None


def test_runs_import_unknown_dataset_exits_2(
    synthetic_store: Callable[..., Path], write_plan: WritePlan, raman_paths: PathsSettings
) -> None:
    run_dir = write_plan(synthetic_store(4), spectra_per_task=4)
    code, env = _import(run_dir)
    assert code == ExitCode.VALIDATION
    assert "ingest" in env.errors[0]
    assert Catalog(raman_paths.catalog).get_run(run_dir.name) is None


def test_runs_import_of_a_dataset_that_changed_exits_2(
    mini_store: Path, write_plan: WritePlan
) -> None:
    run_dir = write_plan(mini_store, dataset_hash="0" * 64)
    code, env = _import(run_dir)
    assert code == ExitCode.VALIDATION
    assert "content hash" in env.errors[0]
