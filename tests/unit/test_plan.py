"""``app.plan``: RunConfig -> run_plan.json + catalog rows (DESIGN.md §5 rule 8, §7.1, §8.5)."""

import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from raman_bayes.app import plan as plan_module
from raman_bayes.app.ingest import IngestRequest, ingest
from raman_bayes.app.plan import PlanResult, plan_run
from raman_bayes.cli.main import app
from raman_bayes.cli.output import CliEnvelope, ExitCode
from raman_bayes.config.settings import PathsSettings, load_settings
from raman_bayes.domain.priors import Window
from raman_bayes.domain.priors_rules import priors_hash
from raman_bayes.domain.run import RunConfig, RunPlan, RunStatus, TaskStatus, config_hash
from raman_bayes.engine.mock import MockEngine
from raman_bayes.ports.engine import EngineConfig
from raman_bayes.storage.catalog import Catalog

FIXTURE = Path(__file__).parents[1] / "fixtures" / "priors" / "design_example.yaml"
NOW = datetime(2026, 11, 2, 12, 0, tzinfo=UTC)
runner = CliRunner()


@pytest.fixture
def dataset_id(mini_scan: Path, raman_paths: PathsSettings) -> str:
    (outcome,) = ingest(IngestRequest(path=mini_scan, allow_partial_grid=True), raman_paths)
    assert outcome.manifest is not None
    return outcome.manifest.dataset_id


def _priors(tmp_path: Path, edit: dict[str, Any] | None = None) -> Path:
    data = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    for key, value in (edit or {}).items():
        if key == "windows+":
            data["windows"].append(value)
        elif key == "smoothing":
            data["preprocessing"]["smoothing"] = value
        else:
            data["spec"][key] = value
    path = tmp_path / "priors.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _plan(dataset: str, priors: Path, **config: Any) -> PlanResult:
    fields: dict[str, Any] = {"engine": "mock", "profile": "local", "spectra_per_task": 16}
    run_config = RunConfig(dataset_id=dataset, priors_path=priors, **(fields | config))
    return plan_run(run_config, load_settings(), now=NOW)


def _cli(*argv: str) -> tuple[int, CliEnvelope]:
    result = runner.invoke(app, ["plan", *argv, "--json"])
    (line,) = result.stdout.splitlines()
    return result.exit_code, CliEnvelope.model_validate_json(line)


def _assert_nothing_planned(paths: PathsSettings) -> None:
    assert not paths.runs_root.exists()
    assert Catalog(paths.catalog).list_runs() == []


def _window(window_id: str, lo: float, hi: float, center: tuple[float, float]) -> dict[str, Any]:
    peak = {"id": f"{window_id}_p", "center": {"mean": sum(center) / 2, "bounds": list(center)}}
    return {
        "id": window_id,
        "range_cm1": [lo, hi],
        "peaks": [peak | {"fwhm": {"bounds": [10, 40]}}],
    }


# ---------------------------------------------------------------- a valid plan


def test_plan_fixture_size_16_gives_2_tasks(dataset_id: str, raman_paths: PathsSettings) -> None:
    result = _plan(dataset_id, FIXTURE)
    assert result.errors == ()
    plan = result.plan
    assert plan is not None
    assert (plan.n_spectra, plan.spectra_per_task) == (25, 16)
    assert [(t.task_index, t.spec_start, t.spec_stop) for t in plan.tasks] == [
        (0, 0, 16),
        (1, 16, 25),
    ]
    assert (plan.engine, plan.engine_version, plan.backend) == ("mock", "0.1.0", "local")
    assert plan.engine_config == {"n_iter": 6000, "burn_in": 3000, "thin": 5, "options": {}}
    assert plan.dataset_id == dataset_id
    assert plan.dataset_uri == str(raman_paths.data_root / f"{dataset_id}.zarr")
    assert plan.created_at == NOW


def test_derived_size_on_local_is_one_chunk_for_the_fixture(dataset_id: str) -> None:
    result = _plan(dataset_id, FIXTURE, spectra_per_task=None)
    assert result.plan is not None
    assert result.plan.spectra_per_task == 1024  # ceil(25 / 8 workers) rounded up to 1024
    assert len(result.plan.tasks) == 1


def test_plan_writes_run_plan_and_catalog_rows(dataset_id: str, raman_paths: PathsSettings) -> None:
    result = _plan(dataset_id, FIXTURE)
    plan = result.plan
    assert plan is not None
    assert result.run_dir == raman_paths.runs_root / plan.run_id
    stored = RunPlan.model_validate_json((result.run_dir / "run_plan.json").read_text("utf-8"))
    assert stored == plan
    assert sorted(p.name for p in result.run_dir.iterdir()) == ["run_plan.json"]
    catalog = Catalog(raman_paths.catalog)
    run = catalog.get_run(plan.run_id)
    assert run is not None
    assert run.status is RunStatus.PLANNED
    assert (run.dataset_id, run.priors_hash, run.engine, run.engine_version) == (
        dataset_id,
        plan.priors_hash,
        "mock",
        "0.1.0",
    )
    assert (run.backend, run.profile, run.seed) == ("local", "local", plan.seed)
    assert run.config_hash == config_hash(plan)
    assert run.created_at == NOW
    tasks = catalog.tasks(plan.run_id)
    assert [(t.task_index, t.spec_start, t.spec_stop) for t in tasks] == [(0, 0, 16), (1, 16, 25)]
    assert all(t.state.status is TaskStatus.PENDING and t.state.attempts == 0 for t in tasks)


def test_config_hash_same_when_only_execution_settings_differ(
    dataset_id: str, raman_paths: PathsSettings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _plan(dataset_id, FIXTURE).plan
    monkeypatch.setenv("RAMAN_PATHS__RUNS_ROOT", str(tmp_path / "elsewhere"))
    monkeypatch.setenv("RAMAN_LOCAL__WORKERS", "3")
    second = _plan(dataset_id, FIXTURE).plan
    assert first is not None
    assert second is not None
    assert first.resolved_config != second.resolved_config
    catalog = Catalog(raman_paths.catalog)
    rows = [catalog.get_run(p.run_id) for p in (first, second)]
    assert [r.config_hash for r in rows if r is not None] == [config_hash(first)] * 2


def test_plan_freezes_absolute_resolved_config(dataset_id: str) -> None:
    result = _plan(dataset_id, FIXTURE)
    assert result.plan is not None
    resolved = result.plan.resolved_config
    assert resolved == load_settings().resolved_config()
    assert resolved["profile_source"] == "package:local.yaml"
    paths = resolved["paths"]
    assert isinstance(paths, dict)
    for key in ("data_root", "runs_root", "catalog"):
        value = paths[key]
        assert isinstance(value, str)
        assert Path(value).is_absolute(), key
    assert result.plan.profile == "local"


def test_plan_registers_priors_spec(dataset_id: str, raman_paths: PathsSettings) -> None:
    result = _plan(dataset_id, FIXTURE)
    assert result.plan is not None
    with closing(sqlite3.connect(raman_paths.catalog)) as conn:
        rows = conn.execute(
            "SELECT priors_hash, name, version, status, yaml_text FROM priors_spec"
        ).fetchall()
    assert rows == [
        (
            priors_hash(result.plan.priors),
            "nmc_commercial",
            "1.0.0",
            "approved",
            FIXTURE.read_text(encoding="utf-8"),
        )
    ]
    second = _plan(dataset_id, FIXTURE)  # same recipe, second run: still one priors_spec row
    assert second.plan is not None
    with closing(sqlite3.connect(raman_paths.catalog)) as conn:
        assert conn.execute("SELECT count(*) FROM priors_spec").fetchone() == (1,)


# ---------------------------------------------------------------- seeds


def test_seed_drawn_and_recorded_when_absent(dataset_id: str, raman_paths: PathsSettings) -> None:
    first, second = _plan(dataset_id, FIXTURE), _plan(dataset_id, FIXTURE)
    assert first.plan is not None
    assert second.plan is not None
    assert 0 <= first.plan.seed < 2**63
    assert first.plan.seed != second.plan.seed
    run = Catalog(raman_paths.catalog).get_run(first.plan.run_id)
    assert run is not None
    assert run.seed == first.plan.seed


def test_explicit_seed_kept(dataset_id: str) -> None:
    result = _plan(dataset_id, FIXTURE, seed=2**63 - 1)
    assert result.plan is not None
    assert result.plan.seed == 2**63 - 1


# ---------------------------------------------------------------- refusals


def test_unknown_dataset_exits_2(raman_paths: PathsSettings) -> None:
    code, env = _cli("--dataset", "NMC811_v0-0000000000", "--priors", str(FIXTURE))
    assert code == ExitCode.VALIDATION
    assert "NMC811_v0-0000000000" in env.errors[0]
    _assert_nothing_planned(raman_paths)


def test_rule8_window_under_32_channels_exits_2(
    dataset_id: str, tmp_path: Path, raman_paths: PathsSettings
) -> None:
    priors = _priors(tmp_path, {"windows+": _window("narrow", 800, 850, (815, 835))})
    code, env = _cli("--dataset", dataset_id, "--priors", str(priors))
    assert code == ExitCode.VALIDATION
    (error,) = env.errors
    assert "narrow" in error
    assert "32" in error
    _assert_nothing_planned(raman_paths)


def test_rule8_margin_outside_axis_exits_2(
    dataset_id: str, tmp_path: Path, raman_paths: PathsSettings
) -> None:
    priors = _priors(tmp_path, {"windows+": _window("edge", 1840, 1940, (1870, 1910))})
    code, env = _cli("--dataset", dataset_id, "--priors", str(priors))
    assert code == ExitCode.VALIDATION
    (error,) = env.errors
    assert "edge" in error
    assert "1965" in error
    _assert_nothing_planned(raman_paths)


def test_rule8_segment_shorter_than_smoothing_window_exits_2(
    dataset_id: str, tmp_path: Path, raman_paths: PathsSettings
) -> None:
    smoothing = {"method": "savgol", "window": 301, "polyorder": 3}
    priors = _priors(tmp_path, {"smoothing": smoothing})
    code, env = _cli("--dataset", dataset_id, "--priors", str(priors))
    assert code == ExitCode.VALIDATION
    (error,) = env.errors  # lithiation's ~200-channel segment, not carbon's ~357
    assert "lithiation" in error
    assert "301" in error
    _assert_nothing_planned(raman_paths)


def test_invalid_priors_refuse_plan(
    dataset_id: str, tmp_path: Path, raman_paths: PathsSettings
) -> None:
    priors = _priors(tmp_path, {"status": "final"})
    result = _plan(dataset_id, priors)
    assert result.plan is None
    assert "status" in result.errors[0]
    _assert_nothing_planned(raman_paths)


def test_unknown_engine_refuses_plan(dataset_id: str, raman_paths: PathsSettings) -> None:
    result = _plan(dataset_id, FIXTURE, engine="bayes_v9")
    assert result.plan is None
    assert "bayes_v9" in result.errors[0]
    _assert_nothing_planned(raman_paths)


def test_engine_validate_messages_refuse_plan(
    dataset_id: str, raman_paths: PathsSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    class PickyEngine(MockEngine):
        def validate(self, window: Window, cfg: EngineConfig) -> list[str]:  # noqa: ARG002
            return (
                [f"window {window.id}: too wide for this engine"] if window.id == "carbon" else []
            )

    monkeypatch.setattr(plan_module, "load_engine", lambda _name: PickyEngine())
    result = _plan(dataset_id, FIXTURE)
    assert result.plan is None
    assert result.errors == ("window carbon: too wide for this engine",)
    _assert_nothing_planned(raman_paths)


def test_explicit_size_over_slurm_cap_refused(
    dataset_id: str, raman_paths: PathsSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAMAN_BACKEND", "slurm_array")
    slurm = (
        '{"partitions": ["normal"], "cpus_per_task": 1, "mem_per_cpu": "4G", '
        '"target_task_minutes": 60, "max_array_tasks": 3, "max_concurrent": 1, '
        '"pixi_bin": "/opt/pixi"}'
    )
    monkeypatch.setenv("RAMAN_SLURM", slurm)
    result = _plan(dataset_id, FIXTURE, spectra_per_task=8)  # 4 tasks > 3
    assert result.plan is None
    assert "4 tasks" in result.errors[0]
    _assert_nothing_planned(raman_paths)


def test_draft_priors_plan_warns(dataset_id: str, tmp_path: Path) -> None:
    result = _plan(dataset_id, _priors(tmp_path, {"status": "draft"}))
    assert result.plan is not None
    assert result.plan.priors.spec.status == "draft"
    assert any("draft" in w for w in result.warnings)
