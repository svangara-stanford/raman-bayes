"""``raman datasets list/show`` and ``runs list/show`` against a catalog (§8.5, §10)."""

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from raman_bayes.cli.main import app
from raman_bayes.cli.output import CliEnvelope, ExitCode
from raman_bayes.domain.dataset import DatasetManifest
from raman_bayes.domain.priors import PriorsSpec
from raman_bayes.domain.run import TaskSpec
from raman_bayes.storage.catalog import Catalog, RunRecord

NOW = datetime(2026, 11, 2, 12, 0, tzinfo=UTC)
FIXTURE = Path(__file__).parents[1] / "fixtures" / "priors" / "design_example.yaml"
RUN_ID = "0190a1b2-c3d4-7e5f-8a9b-0c1d2e3f4a5b"
DATASET_ID = "NMC811_v0-0123456789"

runner = CliRunner()


@pytest.fixture
def catalog_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in list(os.environ):
        if name.upper().startswith("RAMAN_"):
            monkeypatch.delenv(name)
    path = tmp_path / "state" / "catalog.sqlite"
    monkeypatch.setenv("RAMAN_PATHS__CATALOG", str(path))
    monkeypatch.chdir(tmp_path)
    return path


def _manifest() -> DatasetManifest:
    point = {"start_um": 0.0, "stop_um": 0.0, "step_um": 0.0}
    fields: dict[str, Any] = {
        "scan": "NMC811_v0",
        "acquired_at": datetime(2026, 5, 28, 15, 47, 23),
        "duration_s": 2090.0,
        "instrument": {"system": "s", "detector": "d", "positioner": "p", "raw_stage": ""},
        "acq": {
            "exposure_s": 0.2,
            "n_accum": 5,
            "ccd_temp_c": -72.0,
            "central_pixel": 505,
            "head_model": "DU920P_BEX2D",
        },
        "n_wn": 1024,
        "n_series": 1,
        "grid": {"nx": 1, "ny": 1, "nz": 1, "x": point, "y": point, "z": point},
        "dataset_id": DATASET_ID,
        "sample": None,
        "n_spectra": 1,
        "content_hash": "ab" * 32,
        "created_at": NOW,
    }
    return DatasetManifest.model_validate(fields)


def _populate(path: Path) -> None:
    catalog = Catalog(path)
    catalog.add_dataset(_manifest(), path.parent / f"{DATASET_ID}.zarr")
    text = FIXTURE.read_text(encoding="utf-8")
    catalog.add_priors_spec(PriorsSpec.model_validate(yaml.safe_load(text)), "cd" * 32, text, NOW)
    record = RunRecord(
        run_id=RUN_ID,
        dataset_id=DATASET_ID,
        priors_hash="cd" * 32,
        engine="mock",
        engine_version="0.1.0",
        config_hash="ef" * 32,
        git_sha=None,
        backend="local",
        profile="local",
        seed=7,
        created_at=NOW,
    )
    catalog.create_run(record, (TaskSpec(task_index=0, spec_start=0, spec_stop=1),))


def _json(argv: list[str]) -> tuple[int, CliEnvelope]:
    result = runner.invoke(app, [*argv, "--json"])
    (line,) = result.stdout.splitlines()
    return result.exit_code, CliEnvelope.model_validate_json(line)


def test_cli_datasets_list_empty_exits_0(catalog_path: Path) -> None:
    result = runner.invoke(app, ["datasets", "list"])
    assert result.exit_code == 0, result.output
    assert "no datasets" in result.stdout
    assert not catalog_path.exists()


def test_cli_datasets_list_empty_json(catalog_path: Path) -> None:
    code, env = _json(["datasets", "list"])
    assert code == ExitCode.OK
    assert env.data == {"catalog": str(catalog_path), "datasets": []}
    assert not catalog_path.exists()


def test_cli_datasets_list_and_show(catalog_path: Path) -> None:
    _populate(catalog_path)
    code, env = _json(["datasets", "list"])
    assert code == ExitCode.OK
    assert env.data is not None
    (row,) = env.data["datasets"]
    assert row["dataset_id"] == DATASET_ID
    assert row["n_spectra"] == 1
    code, env = _json(["datasets", "show", DATASET_ID])
    assert code == ExitCode.OK
    assert env.data is not None
    assert DatasetManifest.model_validate(env.data["manifest"]) == _manifest()
    assert env.data["store_uri"] == str(catalog_path.parent / f"{DATASET_ID}.zarr")
    human = runner.invoke(app, ["datasets", "list"])
    assert DATASET_ID in human.stdout


def test_cli_datasets_show_unknown_exits_2(catalog_path: Path) -> None:
    _populate(catalog_path)
    code, env = _json(["datasets", "show", "nope-0000000000"])
    assert code == ExitCode.VALIDATION
    (error,) = env.errors
    assert "'nope-0000000000'" in error
    human = runner.invoke(app, ["datasets", "show", "nope-0000000000"])
    assert human.exit_code == 2
    assert human.stdout == ""


def test_cli_runs_list_and_show_json(catalog_path: Path) -> None:
    _populate(catalog_path)
    code, env = _json(["runs", "list"])
    assert code == ExitCode.OK
    assert env.data is not None
    (row,) = env.data["runs"]
    assert (row["run_id"], row["status"], row["dataset_id"]) == (RUN_ID, "planned", DATASET_ID)
    code, env = _json(["runs", "show", RUN_ID])
    assert code == ExitCode.OK
    assert env.data is not None
    assert env.data["run"]["seed"] == 7
    assert env.data["run"]["completed_at"] is None
    assert env.data["tasks"] == [
        {
            "task_index": 0,
            "spec_start": 0,
            "spec_stop": 1,
            "status": "pending",
            "attempts": 0,
            "slurm_job_id": None,
            "n_ok": None,
            "n_failed": None,
            "error": None,
        }
    ]


@pytest.mark.usefixtures("catalog_path")
def test_cli_runs_show_unknown_exits_2() -> None:
    code, env = _json(["runs", "show", "0190a1b2-0000-7000-8000-000000000000"])
    assert code == ExitCode.VALIDATION
    assert "0190a1b2-0000-7000-8000-000000000000" in env.errors[0]


@pytest.mark.usefixtures("catalog_path")
def test_cli_bad_profile_exits_2_with_envelope() -> None:
    code, env = _json(["runs", "list", "--profile", "no_such_profile"])
    assert code == ExitCode.VALIDATION
    assert "no_such_profile" in env.errors[0]
