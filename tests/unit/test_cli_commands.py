"""The §10 command surface: all 18 commands registered; those not built yet are stubs (exit 3)."""

import json
from pathlib import Path
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from raman_bayes.cli.main import app
from raman_bayes.cli.output import CliEnvelope, ExitCode

ROOT = Path(__file__).resolve().parents[2]
ESC = "\x1b"
# User-visible text must not cite repo paths or internal docs; installed users have neither.
REPO_REFS = ("DESIGN.md", "§", "schemas/", "configs/", "docs/", "tests/", "Phase")

# Minimal valid argv for every DESIGN.md §10 command, keyed by its command path.
COMMANDS: dict[str, list[str]] = {
    "ingest": ["ingest", "scans/NMC811_v0"],
    "datasets list": ["datasets", "list"],
    "datasets show": ["datasets", "show", "NMC811_v0-0123456789"],
    "validate-priors": ["validate-priors", "priors.yaml"],
    "priors import-csv": [
        "priors",
        "import-csv",
        "legacy.csv",
        "--window-id",
        "lithiation",
        "--range",
        "350",
        "700",
    ],
    "priors export-csv": ["priors", "export-csv", "priors.yaml", "--window-id", "lithiation"],
    "plan": ["plan", "--dataset", "D", "--priors", "priors.yaml"],
    "submit": ["submit", "RUN"],
    "run": ["run", "--dataset", "D", "--priors", "priors.yaml"],
    "run-task": ["run-task", "--run", "RUN", "--task", "0"],
    "status": ["status", "RUN"],
    "cancel": ["cancel", "RUN", "--confirm"],
    "collect": ["collect", "RUN"],
    "export": ["export", "RUN", "--format", "csv"],
    "runs list": ["runs", "list"],
    "runs show": ["runs", "show", "RUN"],
    "runs import": ["runs", "import", "copied_run/"],
    "doctor": ["doctor"],
}
NAMES = list(COMMANDS)
# Commands built so far; each one has its own test module.
IMPLEMENTED = {
    "ingest",
    "plan",
    "validate-priors",
    "priors import-csv",
    "priors export-csv",
    "datasets list",
    "datasets show",
    "runs list",
    "runs show",
}
STUBS = [name for name in NAMES if name not in IMPLEMENTED]

runner = CliRunner()


def _schema_properties() -> set[str]:
    schema = json.loads((ROOT / "schemas/cli/envelope.schema.json").read_text(encoding="utf-8"))
    return set(schema["properties"])


def _assert_no_repo_refs(text: str) -> None:
    for ref in REPO_REFS:
        assert ref not in text, f"user-visible text cites {ref!r}"


def _leaf_commands(group: Any, prefix: str = "") -> set[str]:
    # Typer 0.27 vendors Click (typer._click), so walk groups by their ``commands`` mapping.
    leaves: set[str] = set()
    for name, cmd in group.commands.items():
        path = f"{prefix}{name}"
        if hasattr(cmd, "commands"):
            leaves |= _leaf_commands(cmd, f"{path} ")
        else:
            leaves.add(path)
    return leaves


# ---------------------------------------------------------------- registration


def test_exactly_the_section_10_commands_are_registered() -> None:
    root = typer.main.get_command(app)
    assert len(COMMANDS) == 18
    assert _leaf_commands(root) == set(COMMANDS)


def test_root_help_lists_top_level_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for top in {name.split()[0] for name in NAMES}:
        assert top in result.stdout
    assert ESC not in result.output
    _assert_no_repo_refs(result.output)


# ---------------------------------------------------------------- every command


@pytest.mark.parametrize("name", STUBS)
def test_stub_exits_nothing_to_do(name: str) -> None:
    result = runner.invoke(app, COMMANDS[name])
    assert result.exit_code == ExitCode.NOTHING_TO_DO, result.output
    assert f"raman {name}: not implemented yet" in result.stdout
    assert result.stderr == ""
    assert ESC not in result.output
    _assert_no_repo_refs(result.output)


@pytest.mark.parametrize("name", STUBS)
def test_stub_json_envelope(name: str) -> None:
    result = runner.invoke(app, [*COMMANDS[name], "--json"])
    assert result.exit_code == ExitCode.NOTHING_TO_DO, result.output
    assert result.stderr == ""
    lines = result.stdout.splitlines()
    assert len(lines) == 1
    env = CliEnvelope.model_validate_json(lines[0])
    assert env.command == name
    assert env.exit_code is ExitCode.NOTHING_TO_DO
    assert env.ok is False
    assert set(json.loads(lines[0])) == _schema_properties()
    assert env.message is not None
    _assert_no_repo_refs(env.message)


@pytest.mark.parametrize("name", NAMES)
def test_help_lists_shared_options(name: str) -> None:
    result = runner.invoke(app, [*name.split(), "--help"])
    assert result.exit_code == 0
    assert "--json" in result.stdout
    assert "--profile" in result.stdout
    assert ESC not in result.output
    _assert_no_repo_refs(result.output)


@pytest.mark.parametrize("name", NAMES)
def test_profile_is_accepted(name: str) -> None:
    # Not a Click usage error: under --json a parsed command always writes one envelope.
    result = runner.invoke(app, [*COMMANDS[name], "--profile", "sherlock", "--json"])
    (line,) = result.stdout.splitlines()
    assert CliEnvelope.model_validate_json(line).command == name, result.output


# ---------------------------------------------------------------- specific options


@pytest.mark.parametrize(("workers", "code"), [("4", 3), ("abc", 2), ("0", 2)])
def test_run_task_workers(workers: str, code: int) -> None:
    result = runner.invoke(app, [*COMMANDS["run-task"], "--workers", workers])
    assert result.exit_code == code, result.output


@pytest.mark.parametrize("missing", ["--run", "--task"])
def test_run_task_requires_run_and_task(missing: str) -> None:
    argv = COMMANDS["run-task"]
    i = argv.index(missing)
    result = runner.invoke(app, argv[:i] + argv[i + 2 :])
    assert result.exit_code == 2


def test_export_csv_requires_window_id() -> None:
    result = runner.invoke(app, ["priors", "export-csv", "priors.yaml"])
    assert result.exit_code == 2
    assert "--window-id" in result.output


def test_import_csv_range_takes_two_numbers() -> None:
    argv = ["priors", "import-csv", "legacy.csv", "--window-id", "w", "--range", "350"]
    assert runner.invoke(app, argv).exit_code == 2


def test_export_rejects_unknown_format() -> None:
    assert runner.invoke(app, ["export", "RUN", "--format", "xlsx"]).exit_code == 2


def test_export_accepts_all_section_8_4_options() -> None:
    argv = [
        "export",
        "RUN",
        "--format",
        "legacy-csv",
        "--window",
        "carbon",
        "--min-presence",
        "0.9",
        "--height-source",
        "height",
        "--absent-fill",
        "0",
        "--coords",
        "um",
    ]
    assert runner.invoke(app, argv).exit_code == ExitCode.NOTHING_TO_DO


def test_ingest_accepts_all_flags() -> None:
    # Not a Click usage error: under --json a parsed command always writes one envelope.
    argv = [
        *COMMANDS["ingest"],
        "--sample",
        "S1",
        "--dry-run",
        "--archive",
        "--allow-partial-grid",
        "--allow-axis-mismatch",
        "--as-job",
        "--json",
    ]
    (line,) = runner.invoke(app, argv).stdout.splitlines()
    assert CliEnvelope.model_validate_json(line).command == "ingest"
