"""``raman plan``: envelope, text report and exit codes (DESIGN.md §7.1, §10)."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from raman_bayes.app.ingest import IngestRequest, ingest
from raman_bayes.cli.main import app
from raman_bayes.cli.output import CliEnvelope, ExitCode
from raman_bayes.config.settings import PathsSettings
from raman_bayes.domain.run import RunPlan

FIXTURE = Path(__file__).parents[1] / "fixtures" / "priors" / "design_example.yaml"
ESC = "\x1b"
REPO_REFS = ("DESIGN.md", "§", "schemas/", "configs/", "docs/", "Phase")
runner = CliRunner()


@pytest.fixture
def dataset_id(mini_scan: Path, raman_paths: PathsSettings) -> str:
    (outcome,) = ingest(IngestRequest(path=mini_scan, allow_partial_grid=True), raman_paths)
    assert outcome.manifest is not None
    return outcome.manifest.dataset_id


def _argv(dataset_id: str, *extra: str) -> list[str]:
    return ["plan", "--dataset", dataset_id, "--priors", str(FIXTURE), *extra]


def test_cli_plan_json_lists_tasks(dataset_id: str, raman_paths: PathsSettings) -> None:
    result = runner.invoke(
        app, [*_argv(dataset_id, "--spectra-per-task", "16", "--seed", "7"), "--json"]
    )
    assert result.exit_code == ExitCode.OK, result.output
    (line,) = result.stdout.splitlines()
    env = CliEnvelope.model_validate_json(line)
    assert env.command == "plan"
    assert env.data is not None
    assert env.data["tasks"] == [
        {"task_index": 0, "spec_start": 0, "spec_stop": 16},
        {"task_index": 1, "spec_start": 16, "spec_stop": 25},
    ]
    assert (env.data["n_tasks"], env.data["spectra_per_task"], env.data["seed"]) == (2, 16, 7)
    assert (env.data["engine"], env.data["engine_version"], env.data["backend"]) == (
        "mock",
        "0.1.0",
        "local",
    )
    run_plan = Path(env.data["run_plan"])
    assert run_plan == raman_paths.runs_root / env.data["run_id"] / "run_plan.json"
    assert RunPlan.model_validate_json(run_plan.read_text("utf-8")).run_id == env.data["run_id"]


def test_cli_plan_text_report(dataset_id: str) -> None:
    result = runner.invoke(app, _argv(dataset_id, "--spectra-per-task", "16"))
    assert result.exit_code == ExitCode.OK, result.output
    out = result.stdout
    assert "2 tasks" in out
    assert "task 0: spectra 0-15" in out
    assert "task 1: spectra 16-24" in out
    assert ESC not in result.output
    for ref in REPO_REFS:
        assert ref not in result.output


def test_cli_plan_engine_defaults_to_profile(
    dataset_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAMAN_ENGINE__NAME", "bayes_v9")
    result = runner.invoke(app, [*_argv(dataset_id), "--json"])
    assert result.exit_code == ExitCode.VALIDATION
    assert "bayes_v9" in CliEnvelope.model_validate_json(result.stdout).errors[0]
    result = runner.invoke(app, [*_argv(dataset_id, "--engine", "mock"), "--json"])
    assert result.exit_code == ExitCode.OK


@pytest.mark.usefixtures("raman_paths")
def test_cli_plan_seed_out_of_range_is_a_usage_error() -> None:
    result = runner.invoke(app, _argv("D", "--seed", str(2**63)))
    assert result.exit_code == 2
    assert "--seed" in result.output
