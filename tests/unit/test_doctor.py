"""``raman doctor`` on a local profile (DESIGN.md §7.5, §10)."""

import json
import os
import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from raman_bayes.app.ingest import IngestRequest, ingest
from raman_bayes.cli.main import app
from raman_bayes.cli.output import CliEnvelope, ExitCode
from raman_bayes.config.settings import PathsSettings

ROOT = Path(__file__).resolve().parents[2]
runner = CliRunner()


def _doctor(*argv: str) -> tuple[int, CliEnvelope]:
    result = runner.invoke(app, ["doctor", *argv, "--json"])
    (line,) = result.stdout.splitlines()
    return result.exit_code, CliEnvelope.model_validate_json(line)


def _checks(env: CliEnvelope) -> dict[str, dict[str, str]]:
    assert env.data is not None
    return {c["name"]: c for c in env.data["checks"]}


def test_doctor_local_ok_exits_0(raman_paths: PathsSettings) -> None:
    code, env = _doctor()
    assert code == ExitCode.OK, env.errors
    assert env.ok
    assert {c["status"] for c in _checks(env).values()} <= {"ok", "skipped"}
    assert not raman_paths.catalog.exists()  # doctor writes nothing


def test_doctor_reports_profile_paths_and_engines(raman_paths: PathsSettings) -> None:
    _, env = _doctor()
    assert env.data is not None
    assert (env.data["profile"], env.data["profile_source"]) == ("local", "package:local.yaml")
    assert env.data["paths"]["runs_root"] == str(raman_paths.runs_root)
    assert env.data["paths"]["catalog"] == str(raman_paths.catalog)
    assert env.data["paths"]["archive_root"] is None
    assert env.data["engines"] == ["mock"]
    assert env.data["backend"] == "local"
    checks = _checks(env)
    assert checks["engine"]["status"] == "ok"
    assert "mock" in checks["engine"]["detail"]
    assert checks["paths.runs_root"]["status"] == "ok"
    assert "created on first use" in checks["paths.runs_root"]["detail"]
    result = runner.invoke(app, ["doctor"])
    assert "profile local" in result.stdout
    assert "ok    engine" in result.stdout


def test_doctor_unresolved_var_exits_2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SCRATCH", raising=False)
    monkeypatch.chdir(ROOT)  # the site profiles are found under configs/profiles/
    code, env = _doctor("--profile", "sherlock")
    assert code == ExitCode.VALIDATION
    assert "SCRATCH" in env.errors[0]


@pytest.mark.usefixtures("raman_paths")
def test_doctor_skips_slurm_checks_on_local() -> None:
    checks = _checks(_doctor()[1])
    for name in ("slurm.pixi_bin", "slurm.limits", "quotas"):
        assert checks[name]["status"] == "skipped"
        assert "local" in checks[name]["detail"]


def test_doctor_warns_dataset_older_than_60_days(
    mini_scan: Path, raman_paths: PathsSettings
) -> None:
    old = datetime.now(UTC) - timedelta(days=61)
    (outcome,) = ingest(
        IngestRequest(path=mini_scan, allow_partial_grid=True), raman_paths, now=old
    )
    assert outcome.manifest is not None
    code, env = _doctor()
    assert code == ExitCode.OK
    assert env.warnings
    assert outcome.manifest.dataset_id in env.warnings[0]
    assert "61 days" in env.warnings[0]
    assert _checks(env)["datasets.age"]["status"] == "warn"


def test_doctor_fails_on_a_path_it_cannot_write(raman_paths: PathsSettings) -> None:
    raman_paths.runs_root.parent.mkdir(parents=True)
    os.chmod(raman_paths.runs_root.parent, 0o500)
    try:
        code, env = _doctor()
    finally:
        os.chmod(raman_paths.runs_root.parent, 0o700)
    assert code == ExitCode.BACKEND
    assert _checks(env)["paths.runs_root"]["status"] == "fail"
    assert any("runs_root" in e for e in env.errors)


@pytest.mark.usefixtures("raman_paths")
def test_doctor_fails_when_the_profile_engine_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAMAN_ENGINE__NAME", "nonesuch")
    code, env = _doctor()
    assert code == ExitCode.BACKEND
    assert "nonesuch" in _checks(env)["engine"]["detail"]


@pytest.mark.usefixtures("raman_paths")
def test_doctor_json_validates() -> None:
    result = runner.invoke(app, ["doctor", "--json"])
    schema = json.loads((ROOT / "schemas" / "cli" / "envelope.schema.json").read_text("utf-8"))
    payload = json.loads(result.stdout)
    assert set(payload) == set(schema["properties"])
    assert CliEnvelope.model_validate(payload).command == "doctor"
    for check in payload["data"]["checks"]:
        assert set(check) == {"name", "status", "detail"}
        assert check["status"] in {"ok", "warn", "fail", "skipped"}


def test_doctor_fails_on_a_catalog_of_another_version(raman_paths: PathsSettings) -> None:
    raman_paths.catalog.parent.mkdir(parents=True)
    with closing(sqlite3.connect(raman_paths.catalog)) as conn:
        conn.execute("PRAGMA user_version = 7")
    code, env = _doctor()
    assert code == ExitCode.BACKEND
    assert "schema version 7" in _checks(env)["catalog"]["detail"]


@pytest.mark.parametrize(("pixi_bin", "status"), [("/bin/sh", "ok"), ("/no/such/pixi", "fail")])
@pytest.mark.usefixtures("raman_paths")
def test_doctor_checks_pixi_bin_on_slurm(tmp_path: Path, pixi_bin: str, status: str) -> None:
    profile = tmp_path / "cluster.yaml"
    profile.write_text(
        "profile: cluster\nbackend: slurm_array\nslurm:\n  partitions: [normal]\n"
        "  cpus_per_task: 1\n  mem_per_cpu: 1G\n  target_task_minutes: 60\n"
        f"  max_array_tasks: 10\n  max_concurrent: 10\n  pixi_bin: {pixi_bin}\n",
        encoding="utf-8",
    )
    code, env = _doctor("--profile", str(profile))
    checks = _checks(env)
    assert checks["slurm.pixi_bin"]["status"] == status
    assert pixi_bin in checks["slurm.pixi_bin"]["detail"]
    assert checks["slurm.limits"]["status"] == "skipped"
    assert code == (ExitCode.OK if status == "ok" else ExitCode.BACKEND)
