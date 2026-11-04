"""Every command's ``--json`` output and exit code (DESIGN.md §10, §11 I7; plan §3.3)."""

import errno
import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path

import jsonschema
import pytest
import typer
from typer.testing import CliRunner, Result

from raman_bayes.app import catalog_views
from raman_bayes.cli.main import app
from raman_bayes.cli.output import CliEnvelope, ExitCode
from raman_bayes.config.settings import PathsSettings
from raman_bayes.domain.errors import DatasetStoreError
from raman_bayes.storage import shards, task_lock

ROOT = Path(__file__).resolve().parents[2]
PRIORS = ROOT / "tests" / "fixtures" / "priors" / "design_example.yaml"
LEGACY = ROOT / "configs" / "priors" / "legacy" / "nmc_lithiation.csv"
SCHEMA = json.loads((ROOT / "schemas" / "cli" / "envelope.schema.json").read_text("utf-8"))
VALIDATOR = jsonschema.Draft202012Validator(SCHEMA)
ESC = "\x1b"
# The §10 commands, copied by hand.
SECTION_10 = {
    "ingest", "datasets list", "datasets show", "validate-priors", "priors import-csv",
    "priors export-csv", "plan", "submit", "run", "run-task", "status", "cancel", "collect",
    "export", "runs list", "runs show", "runs import", "doctor",
}  # fmt: skip
runner = CliRunner()


def _envelope(result: Result) -> CliEnvelope:
    (line,) = result.stdout.splitlines()
    payload = json.loads(line)
    VALIDATOR.validate(payload)
    return CliEnvelope.model_validate(payload)


def _json(*argv: str) -> tuple[int, CliEnvelope]:
    result = runner.invoke(app, [*argv, "--json"])
    return result.exit_code, _envelope(result)


def test_every_command_json_validates_against_schema(
    mini_scan: Path, raman_paths: PathsSettings
) -> None:
    VALIDATOR.check_schema(SCHEMA)
    seen: dict[str, int] = {}

    def run(*argv: str) -> CliEnvelope:
        code, envelope = _json(*argv)
        assert envelope.exit_code == code
        seen[envelope.command] = code
        return envelope

    (scan,) = run("ingest", str(mini_scan), "--allow-partial-grid").data["scans"]  # type: ignore[index]
    dataset = scan["dataset_id"]
    run("datasets", "list")
    run("datasets", "show", dataset)
    run("validate-priors", str(PRIORS))
    run("priors", "import-csv", str(LEGACY), "--window-id", "lithiation", "--range", "350", "700")
    run("priors", "export-csv", str(PRIORS), "--window-id", "lithiation")
    planned = run("plan", "--dataset", dataset, "--priors", str(PRIORS), "--spectra-per-task", "16")
    run_id = planned.data["run_id"]  # type: ignore[index]
    run("submit", run_id)
    run("status", run_id)
    run("run-task", "--run", run_id, "--task", "0")
    run("collect", run_id)
    run("export", run_id, "--format", "csv")
    run("runs", "list")
    run("runs", "show", run_id)
    run("runs", "import", str(raman_paths.runs_root / run_id))
    run("cancel", run_id, "--confirm")
    run("run", "--dataset", dataset, "--priors", str(PRIORS))
    run("doctor")
    assert set(seen) == SECTION_10
    assert seen["run-task"] == seen["runs import"] == seen["cancel"] == ExitCode.NOTHING_TO_DO
    assert {seen[c] for c in SECTION_10 - {"run-task", "runs import", "cancel"}} == {0}


def _ingested_twice(mini_scan: Path, _paths: PathsSettings) -> list[str]:
    assert runner.invoke(app, ["ingest", str(mini_scan), "--allow-partial-grid"]).exit_code == 0
    return ["ingest", str(mini_scan), "--allow-partial-grid"]


def _planned(mini_scan: Path) -> str:
    result = runner.invoke(app, ["ingest", str(mini_scan), "--allow-partial-grid", "--json"])
    dataset = _envelope(result).data["scans"][0]["dataset_id"]  # type: ignore[index]
    argv = ["plan", "--dataset", dataset, "--priors", str(PRIORS), "--json"]
    return str(_envelope(runner.invoke(app, argv)).data["run_id"])  # type: ignore[index]


def _held_elsewhere(mini_scan: Path, paths: PathsSettings) -> list[str]:
    run_id = _planned(mini_scan)
    folder = shards.task_dir(paths.runs_root / run_id, 0)
    folder.mkdir(parents=True)
    holder = {"host": "elsewhere.example", "pid": 7, "started_at": "2026-11-04T09:00:00Z"}
    (folder / task_lock.LOCK_FILE).write_text(json.dumps(holder), encoding="utf-8")
    return ["run-task", "--run", run_id, "--task", "0"]


def _store_gone(mini_scan: Path, paths: PathsSettings) -> list[str]:
    # plan, not submit: planning against a missing store writes nothing, so it can be repeated.
    result = runner.invoke(app, ["ingest", str(mini_scan), "--allow-partial-grid", "--json"])
    dataset = _envelope(result).data["scans"][0]["dataset_id"]  # type: ignore[index]
    shutil.rmtree(paths.data_root / f"{dataset}.zarr")
    return ["plan", "--dataset", dataset, "--priors", str(PRIORS)]


Scenario = Callable[[Path, PathsSettings], list[str]]
SCENARIOS: list[tuple[str, Scenario, ExitCode]] = [
    ("ok", lambda _scan, _paths: ["validate-priors", str(PRIORS)], ExitCode.OK),
    (
        "invalid input",
        lambda _scan, _paths: ["validate-priors", "missing.yaml"],
        ExitCode.VALIDATION,
    ),
    ("usage error", lambda _scan, _paths: ["plan", "--dataset", "D"], ExitCode.VALIDATION),
    ("already ingested", _ingested_twice, ExitCode.NOTHING_TO_DO),
    ("already running", _held_elsewhere, ExitCode.NOTHING_TO_DO),
    ("store missing", _store_gone, ExitCode.BACKEND),
]


@pytest.mark.parametrize(("scenario", "argv", "code"), SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_exit_codes_per_section_10(
    mini_scan: Path, raman_paths: PathsSettings, scenario: str, argv: Scenario, code: ExitCode
) -> None:
    args = argv(mini_scan, raman_paths)
    text = runner.invoke(app, args)
    assert text.exit_code == code, (scenario, text.output)
    exit_code, envelope = _json(*args)
    assert exit_code == envelope.exit_code == code
    assert envelope.ok is (code is ExitCode.OK)


def test_usage_error_under_json_emits_envelope_exit_2() -> None:
    code, envelope = _json("plan", "--dataset", "D")
    assert code == ExitCode.VALIDATION
    assert (envelope.command, envelope.ok) == ("plan", False)
    assert "--priors" in envelope.errors[0]
    code, envelope = _json("runs", "show", "R", "--watch")
    assert (code, envelope.command) == (ExitCode.VALIDATION, "runs show")
    assert "--watch" in envelope.errors[0]
    code, envelope = _json("nonesuch")
    assert (code, envelope.command) == (ExitCode.VALIDATION, "raman")
    plain = runner.invoke(app, ["plan", "--dataset", "D"])
    assert plain.exit_code == ExitCode.VALIDATION
    assert plain.stdout == ""
    assert "--priors" in plain.stderr
    assert "Usage:" in plain.stderr


@pytest.mark.parametrize(
    "error",
    [OSError(errno.EIO, os.strerror(errno.EIO)), DatasetStoreError(Path("d.zarr"), "unreadable")],
    ids=["OSError", "RamanBayesError"],
)
@pytest.mark.usefixtures("raman_paths")
def test_unhandled_exception_exits_4(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    def broken(*_args: object) -> None:
        raise error

    monkeypatch.setattr(catalog_views, "list_datasets", broken)
    code, envelope = _json("datasets", "list")
    assert code == ExitCode.BACKEND
    assert (envelope.command, envelope.ok) == ("datasets list", False)
    assert envelope.errors == (str(error),)
    assert "storage or backend problem" in (envelope.message or "")
    plain = runner.invoke(app, ["datasets", "list"])
    assert plain.exit_code == ExitCode.BACKEND
    assert str(error) in plain.stderr


@pytest.mark.usefixtures("raman_paths")
def test_any_other_exception_is_a_bug_with_a_traceback(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*_args: object) -> None:
        raise ZeroDivisionError

    monkeypatch.setattr(catalog_views, "list_datasets", broken)
    result = runner.invoke(app, ["datasets", "list", "--json"])
    assert result.exit_code == 1
    assert isinstance(result.exception, ZeroDivisionError)
    assert result.stdout == ""


@pytest.mark.usefixtures("raman_paths")
def test_no_ansi_when_not_a_tty() -> None:
    colour = CliRunner(env={"GITHUB_ACTIONS": "true", "FORCE_COLOR": "1", "PY_COLORS": "1"})
    for argv in (
        ["--help"],
        ["validate-priors", str(PRIORS)],
        ["validate-priors", "missing.yaml"],
        ["plan", "--dataset", "D"],
        ["status", "0190b3d4-0000-7000-8000-000000000000"],
        ["doctor"],
    ):
        result = colour.invoke(app, argv)
        assert result.exit_code in (0, 2), (argv, result.output)
        assert ESC not in result.stdout + result.stderr, argv


@pytest.mark.usefixtures("raman_paths")
def test_abort_under_json_prints_aborted_and_exits_1(monkeypatch: pytest.MonkeyPatch) -> None:
    def abort(*_args: object) -> None:
        raise typer.Abort

    monkeypatch.setattr(catalog_views, "list_datasets", abort)
    result = runner.invoke(app, ["datasets", "list", "--json"])
    assert result.exit_code == 1
    assert result.stdout == ""
    assert "Aborted!" in result.stderr


@pytest.mark.usefixtures("raman_paths")
def test_main_without_standalone_mode_returns_the_exit_code() -> None:
    root = typer.main.get_command(app)
    assert root.main(["validate-priors", str(PRIORS)], standalone_mode=False) == ExitCode.OK
    missing = ["validate-priors", "missing.yaml", "--json"]
    assert root.main(missing, standalone_mode=False) == ExitCode.VALIDATION
