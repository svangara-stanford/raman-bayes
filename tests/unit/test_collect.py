"""``raman collect`` (DESIGN.md §7.4, §7.5, §8): merge a run's shards into ``results/``."""

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from raman_bayes.app.collect import CollectRequest, CollectStatus, collect_run
from raman_bayes.app.plan import plan_run
from raman_bayes.app.run_task import TaskRequest, run_task
from raman_bayes.cli.main import app
from raman_bayes.cli.output import CliEnvelope, ExitCode
from raman_bayes.config.settings import PathsSettings, load_settings
from raman_bayes.domain.errors import CollectRefusedError
from raman_bayes.domain.run import RunConfig, RunStatus
from raman_bayes.storage import shards
from raman_bayes.storage.catalog import Catalog

PRIORS = Path(__file__).parents[1] / "fixtures" / "priors" / "design_example.yaml"
RESULT_FILES = ["fits.parquet", "peaks.parquet", "run_manifest.json"]
runner = CliRunner()


@pytest.fixture
def run_id(mini_store: Path, raman_paths: PathsSettings) -> str:
    """Plan the mini dataset in two tasks and mark the run submitted; run no task yet."""
    config = RunConfig(
        dataset_id=mini_store.name.removesuffix(".zarr"),
        priors_path=PRIORS,
        engine="mock",
        spectra_per_task=16,
        seed=3,
        profile="local",
    )
    planned = plan_run(config, load_settings())
    assert planned.plan is not None, planned.errors
    run = planned.plan.run_id
    Catalog(raman_paths.catalog).transition(run, RunStatus.PLANNED, RunStatus.SUBMITTED)
    return run


def _run_tasks(paths: PathsSettings, run: str, *tasks: int) -> None:
    for task in tasks:
        run_task(TaskRequest(paths.runs_root / run, task))


def _cli(*argv: str) -> tuple[int, CliEnvelope]:
    result = runner.invoke(app, [*argv, "--json"])
    (line,) = result.stdout.splitlines()
    return result.exit_code, CliEnvelope.model_validate_json(line)


def _run_row(paths: PathsSettings, run: str) -> tuple[RunStatus, str | None]:
    row = Catalog(paths.catalog).get_run(run)
    assert row is not None
    return row.status, row.result_uri


def test_collect_refuses_incomplete_run(run_id: str, raman_paths: PathsSettings) -> None:
    _run_tasks(raman_paths, run_id, 0)
    with pytest.raises(CollectRefusedError, match="--partial"):
        collect_run(CollectRequest(run_id), load_settings())
    code, env = _cli("collect", run_id)
    assert code == ExitCode.VALIDATION
    assert "running" in env.errors[0]
    assert not (raman_paths.runs_root / run_id / "results").exists()


def test_collect_partial_marks_manifest_keeps_status(
    run_id: str, raman_paths: PathsSettings
) -> None:
    _run_tasks(raman_paths, run_id, 0)
    result = collect_run(CollectRequest(run_id, partial=True), load_settings())
    assert result.status is CollectStatus.PARTIAL
    assert result.manifest is not None
    assert result.manifest.partial is True
    assert set(result.manifest.shard_hashes) == {0}
    results = raman_paths.runs_root / run_id / "results"
    assert sorted(p.name for p in results.iterdir()) == RESULT_FILES
    assert json.loads((results / "run_manifest.json").read_text("utf-8"))["partial"] is True
    assert pq.read_metadata(results / "peaks.parquet").num_rows == 16 * 4
    assert _run_row(raman_paths, run_id) == (RunStatus.RUNNING, None)


def test_collect_complete_moves_to_collected(run_id: str, raman_paths: PathsSettings) -> None:
    _run_tasks(raman_paths, run_id, 0, 1)
    result = collect_run(CollectRequest(run_id), load_settings())
    assert result.status is CollectStatus.COLLECTED
    assert result.manifest is not None
    assert (result.manifest.partial, result.manifest.n_tasks) == (False, 2)
    results = raman_paths.runs_root / run_id / "results"
    assert result.results_dir == results
    peaks = pq.read_table(results / "peaks.parquet")
    assert peaks.num_rows == 25 * 2 * 2
    assert peaks.column("spectrum_id").to_pylist() == sorted(
        peaks.column("spectrum_id").to_pylist()
    )
    assert _run_row(raman_paths, run_id) == (RunStatus.COLLECTED, str(results))
    for task in (0, 1):  # collect only reads the shards
        assert shards.read_done(shards.task_dir(raman_paths.runs_root / run_id, task)) is not None


def test_collect_twice_exits_3(run_id: str, raman_paths: PathsSettings) -> None:
    _run_tasks(raman_paths, run_id, 0, 1)
    code, env = _cli("collect", run_id)
    assert code == ExitCode.OK, env.errors
    assert env.data is not None
    assert (env.data["status"], env.data["run_status"]) == ("collected", "collected")
    assert env.data["files"] == [
        str(raman_paths.runs_root / run_id / "results" / name) for name in RESULT_FILES
    ]
    assert env.data["manifest"]["n_tasks"] == 2
    code, env = _cli("collect", run_id)
    assert code == ExitCode.NOTHING_TO_DO
    assert "already collected" in (env.message or "")


def test_collect_copies_to_archive_root_when_set(
    run_id: str, raman_paths: PathsSettings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_root = tmp_path / "oak" / "raman"
    monkeypatch.setenv("RAMAN_PATHS__ARCHIVE_ROOT", str(archive_root))
    _run_tasks(raman_paths, run_id, 0, 1)
    result = collect_run(CollectRequest(run_id), load_settings())
    archived = archive_root / run_id / "results"
    assert result.archive_dir == archived
    results = raman_paths.runs_root / run_id / "results"
    for name in RESULT_FILES:
        assert (archived / name).read_bytes() == (results / name).read_bytes()
    assert list(archived.glob("*.tmp")) == []
    assert _run_row(raman_paths, run_id) == (RunStatus.COLLECTED, str(archived))


def test_cli_collect_as_job_on_local_exits_2(run_id: str, raman_paths: PathsSettings) -> None:
    _run_tasks(raman_paths, run_id, 0, 1)
    code, env = _cli("collect", run_id, "--as-job")
    assert code == ExitCode.VALIDATION
    assert "--as-job" in env.errors[0]
    assert not (raman_paths.runs_root / run_id / "results").exists()


def test_cli_collect_partial_with_no_finished_task_exits_2(run_id: str) -> None:
    code, env = _cli("collect", run_id, "--partial")
    assert code == ExitCode.VALIDATION
    assert "no task has finished" in env.errors[0]


def test_cli_collect_of_tasks_that_ran_different_code_exits_4(
    run_id: str, raman_paths: PathsSettings
) -> None:
    _run_tasks(raman_paths, run_id, 0, 1)
    folder = shards.task_dir(raman_paths.runs_root / run_id, 1)
    marker = shards.read_done(folder)
    assert marker is not None
    shards.write_done(folder, marker.model_copy(update={"package_version": "9.9.9"}))
    code, env = _cli("collect", run_id)
    assert code == ExitCode.BACKEND
    assert "9.9.9" in env.errors[0]
    assert _run_row(raman_paths, run_id) == (RunStatus.COMPLETE, None)


def test_cli_collect_text_report(
    run_id: str, raman_paths: PathsSettings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAMAN_PATHS__ARCHIVE_ROOT", str(tmp_path / "archive"))
    _run_tasks(raman_paths, run_id, 0)
    result = runner.invoke(app, ["collect", run_id, "--partial"])
    assert result.exit_code == 0, result.output
    assert f"run {run_id}: collected 1 of 2 tasks (partial; run is still running)" in result.stdout
    assert "16 spectra" not in result.stdout  # the manifest counts the plan's spectra
    assert "25 spectra, 0 failed fits, 0 skipped_qc" in result.stdout
    assert f"copied to {tmp_path / 'archive' / run_id / 'results'}" in result.stdout
