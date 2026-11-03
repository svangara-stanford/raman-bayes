"""``raman ingest``: envelopes and exit codes (DESIGN.md §10), and the B8 refusals."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from raman_bayes.cli.main import app
from raman_bayes.cli.output import CliEnvelope, ExitCode
from raman_bayes.config.settings import PathsSettings

ESC = "\x1b"
REPO_REFS = ("DESIGN.md", "§", "schemas/", "configs/", "docs/", "tests/fixtures", "Phase")
runner = CliRunner()


def _json(*argv: str) -> tuple[int, CliEnvelope]:
    result = runner.invoke(app, ["ingest", *argv, "--json"])
    (line,) = result.stdout.splitlines()
    return result.exit_code, CliEnvelope.model_validate_json(line)


def _no_repo_refs(text: str) -> None:
    for ref in REPO_REFS:
        assert ref not in text, ref


def test_cli_ingest_json_and_exit_codes(mini_scan: Path, raman_paths: PathsSettings) -> None:
    code, env = _json(str(mini_scan), "--allow-partial-grid", "--sample", "S1")
    assert code == ExitCode.OK, env
    assert env.command == "ingest"
    assert env.data is not None
    (scan,) = env.data["scans"]
    assert scan["status"] == "ingested"
    assert scan["folder"] == str(mini_scan)
    assert scan["dataset_id"].startswith("NMC811_v0-")
    assert scan["store_uri"] == str(raman_paths.data_root / f"{scan['dataset_id']}.zarr")
    assert scan["n_spectra"] == 25
    assert scan["manifest"]["sample"] == "S1"

    code, env = _json(str(mini_scan), "--allow-partial-grid")
    assert code == ExitCode.NOTHING_TO_DO
    assert env.ok is False

    code, env = _json(str(mini_scan.parent / "absent"))
    assert code == ExitCode.VALIDATION
    assert env.errors

    for message in (env.message or "", *env.errors):
        _no_repo_refs(message)


def test_cli_ingest_count_mismatch_exits_2(mini_scan: Path, raman_paths: PathsSettings) -> None:
    code, env = _json(str(mini_scan))
    assert code == ExitCode.VALIDATION
    assert env.data is not None
    assert env.data["scans"][0]["status"] == "failed"
    assert "25 of 1250" in env.errors[0]
    assert not raman_paths.data_root.exists()


@pytest.mark.parametrize("flag", ["--archive", "--as-job"])
def test_cli_ingest_archive_and_as_job_exit_2_on_local(
    flag: str, mini_scan: Path, raman_paths: PathsSettings
) -> None:
    code, env = _json(str(mini_scan), "--allow-partial-grid", flag)
    assert code == ExitCode.VALIDATION
    assert flag in env.errors[0]
    assert not raman_paths.data_root.exists()


def test_cli_ingest_dry_run_human_output(mini_scan: Path, raman_paths: PathsSettings) -> None:
    result = runner.invoke(app, ["ingest", str(mini_scan), "--allow-partial-grid", "--dry-run"])
    assert result.exit_code == ExitCode.OK, result.output
    out = result.stdout
    assert "NMC811_v0" in out
    assert "50 x 25 x 1" in out
    assert "25 of 1250" in out
    assert "NMC811_v0-z0y1x4.txt" in out  # the 10th file
    assert "NMC811_v0-z0y2x0.txt" not in out  # the 11th is not listed
    assert "n_accum" in out
    assert ESC not in result.output
    _no_repo_refs(result.output)
    assert not raman_paths.data_root.exists()


@pytest.mark.usefixtures("raman_paths")
def test_cli_ingest_human_failure_goes_to_stderr(mini_scan: Path) -> None:
    result = runner.invoke(app, ["ingest", str(mini_scan)])
    assert result.exit_code == ExitCode.VALIDATION
    assert result.stdout == ""
    assert "25 of 1250" in result.stderr
    _no_repo_refs(result.output)
