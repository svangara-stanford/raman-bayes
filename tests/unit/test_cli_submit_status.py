"""``raman submit``, ``status``, ``cancel`` and ``run`` on the local backend (§7.4, §10)."""

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from raman_bayes.cli.main import app
from raman_bayes.cli.output import CliEnvelope, ExitCode
from raman_bayes.config.settings import PathsSettings
from raman_bayes.domain.run import RunStatus
from raman_bayes.storage import shards, task_lock
from raman_bayes.storage.catalog import Catalog

PRIORS = Path(__file__).parents[1] / "fixtures" / "priors" / "design_example.yaml"
OTHER_HOST = {"host": "elsewhere.example", "pid": 4242, "started_at": "2026-11-03T11:00:00Z"}
runner = CliRunner()


def _cli(*argv: str) -> tuple[int, CliEnvelope]:
    result = runner.invoke(app, [*argv, "--json"])
    (line,) = result.stdout.splitlines()
    return result.exit_code, CliEnvelope.model_validate_json(line)


@pytest.fixture
def dataset_id(mini_store: Path) -> str:
    return mini_store.name.removesuffix(".zarr")


@pytest.fixture
def planned(dataset_id: str) -> str:
    """Plan the mini dataset in two tasks through the CLI; return the run id."""
    code, env = _cli(
        "plan", "--dataset", dataset_id, "--priors", str(PRIORS), "--spectra-per-task", "16"
    )
    assert code == ExitCode.OK, env.errors
    assert env.data is not None
    return str(env.data["run_id"])


def _move(paths: PathsSettings, run_id: str, *statuses: RunStatus) -> None:
    catalog = Catalog(paths.catalog)
    for status in statuses:
        catalog.transition(run_id, _status(paths, run_id), status)


def _status(paths: PathsSettings, run_id: str) -> RunStatus:
    run = Catalog(paths.catalog).get_run(run_id)
    assert run is not None
    return run.status


# ---------------------------------------------------------------- submit


def test_cli_submit_planned_run_completes(planned: str, raman_paths: PathsSettings) -> None:
    code, env = _cli("submit", planned)
    assert code == ExitCode.OK, env.errors
    assert env.data is not None
    assert (env.data["status"], env.data["run_status"]) == ("finished", "complete")
    assert env.data["record"]["attempt"] == 1
    assert env.data["record"]["task_indices"] == [0, 1]
    assert [t["status"] for t in env.data["tasks"]] == ["done", "done"]
    assert [t["n_ok"] for t in env.data["tasks"]] == [32, 18]
    assert env.data["failures"] == []
    assert _status(raman_paths, planned) is RunStatus.COMPLETE
    assert "complete" in (env.message or "")
    stored = json.loads((raman_paths.runs_root / planned / "submission.json").read_text("utf-8"))
    assert [r["attempt"] for r in stored] == [1]


def test_cli_submit_state_refusals_exit_codes(
    planned: str, dataset_id: str, raman_paths: PathsSettings
) -> None:
    code, env = _cli("submit", planned, "--resubmit-failed")
    assert code == ExitCode.VALIDATION
    assert "submit it without --resubmit-failed" in env.errors[0]
    assert _cli("submit", "0190b3d4-0000-7000-8000-000000000000")[0] == ExitCode.VALIDATION

    _move(raman_paths, planned, RunStatus.SUBMITTED)
    code, env = _cli("submit", planned)
    assert code == ExitCode.NOTHING_TO_DO
    assert "already submitted" in (env.message or "")
    assert _cli("submit", planned, "--resubmit-failed")[0] == ExitCode.VALIDATION

    _move(raman_paths, planned, RunStatus.RUNNING, RunStatus.FAILED)
    code, env = _cli("submit", planned)
    assert code == ExitCode.VALIDATION
    assert "--resubmit-failed" in env.errors[0]

    _move(raman_paths, planned, RunStatus.CANCELLED)
    for argv in (("submit", planned), ("submit", planned, "--resubmit-failed")):
        code, env = _cli(*argv)
        assert code == ExitCode.VALIDATION
        assert "cancelled" in env.errors[0]

    code, env = _cli("plan", "--dataset", dataset_id, "--priors", str(PRIORS))
    assert env.data is not None
    done = str(env.data["run_id"])
    assert _cli("submit", done)[0] == ExitCode.OK
    for argv in (("submit", done), ("submit", done, "--resubmit-failed")):
        code, env = _cli(*argv)
        assert code == ExitCode.NOTHING_TO_DO
        assert "complete" in (env.message or "")


def test_cli_submit_dry_run_with_confirm_exits_2(planned: str, raman_paths: PathsSettings) -> None:
    code, env = _cli("submit", planned, "--dry-run", "--confirm")
    assert code == ExitCode.VALIDATION
    assert "--dry-run and --confirm" in env.errors[0]
    code, env = _cli("submit", planned, "--dry-run")
    assert code == ExitCode.OK
    assert env.data is not None
    assert (env.data["status"], env.data["record"]["dry_run"]) == ("dry_run", True)
    assert _status(raman_paths, planned) is RunStatus.PLANNED
    assert not (raman_paths.runs_root / planned / "submission.json").exists()


def test_cli_submit_with_a_slurm_profile_exits_2(planned: str, tmp_path: Path) -> None:
    profile = tmp_path / "cluster.yaml"
    profile.write_text(
        "profile: cluster\nbackend: slurm_array\nslurm:\n  partitions: [normal]\n"
        "  cpus_per_task: 1\n  mem_per_cpu: 1G\n  target_task_minutes: 60\n"
        "  max_array_tasks: 10\n  max_concurrent: 10\n  pixi_bin: /usr/bin/true\n",
        encoding="utf-8",
    )
    code, env = _cli("submit", planned, "--profile", str(profile))
    assert code == ExitCode.VALIDATION
    assert "slurm_array" in env.errors[0]


# ---------------------------------------------------------------- status


def test_cli_status_json_lists_task_states(planned: str) -> None:
    code, env = _cli("status", planned)
    assert code == ExitCode.OK
    assert env.data is not None
    assert env.data["run"]["status"] == "planned"
    assert [(t["task_index"], t["status"], t["attempts"]) for t in env.data["tasks"]] == [
        (0, "pending", 0),
        (1, "pending", 0),
    ]
    assert env.data["entered"] == []
    _cli("submit", planned)
    code, env = _cli("status", planned)
    assert env.data is not None
    assert [t["status"] for t in env.data["tasks"]] == ["done", "done"]
    assert env.data["run"]["status"] == "complete"
    assert _cli("status", "0190b3d4-0000-7000-8000-000000000000")[0] == ExitCode.VALIDATION


def _stale_lock(paths: PathsSettings, run_id: str, task: int) -> None:
    folder = shards.task_dir(paths.runs_root / run_id, task)
    folder.mkdir(parents=True)
    lock = folder / task_lock.LOCK_FILE
    lock.write_text(json.dumps(OTHER_HOST), encoding="utf-8")
    stamp = time.time() - 3600
    os.utime(lock, (stamp, stamp))


def test_cli_status_failures_lists_failed_tasks(planned: str, raman_paths: PathsSettings) -> None:
    _move(raman_paths, planned, RunStatus.SUBMITTED, RunStatus.RUNNING)
    _stale_lock(raman_paths, planned, 1)
    code, env = _cli("status", planned, "--failures")
    assert code == ExitCode.OK
    assert env.data is not None
    assert [f["task_index"] for f in env.data["failures"]] == [1]
    assert "elsewhere.example" in env.data["failures"][0]["error"]
    result = runner.invoke(app, ["status", planned, "--failures"])
    assert "task 1: failed" in result.stdout
    assert "elsewhere.example" in result.stdout
    plain = runner.invoke(app, ["status", planned])
    assert "elsewhere.example" not in plain.stdout


def test_cli_status_watch_returns_on_terminal_run(
    planned: str, raman_paths: PathsSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_sleep(seconds: float) -> None:
        pytest.fail(f"watch slept {seconds} s on a finished run")

    monkeypatch.setattr(time, "sleep", no_sleep)
    _move(raman_paths, planned, RunStatus.CANCELLED)
    code, env = _cli("status", planned, "--watch")
    assert code == ExitCode.OK
    assert env.data is not None
    assert env.data["run"]["status"] == "cancelled"


def test_cli_status_watch_refreshes_until_terminal(
    planned: str, raman_paths: PathsSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _move(raman_paths, planned, RunStatus.SUBMITTED, RunStatus.RUNNING)
    naps: list[float] = []

    def nap(seconds: float) -> None:
        naps.append(seconds)
        _stale_lock(raman_paths, planned, len(naps) - 1)  # one more task stops per refresh

    monkeypatch.setattr(time, "sleep", nap)
    code, env = _cli("status", planned, "--watch")
    assert code == ExitCode.OK
    assert env.data is not None
    assert env.data["run"]["status"] == "failed"
    assert len(naps) == 2


# ---------------------------------------------------------------- cancel


def test_cli_cancel_requires_confirm(planned: str, raman_paths: PathsSettings) -> None:
    result = runner.invoke(app, ["cancel", planned, "--json"])
    assert result.exit_code == ExitCode.VALIDATION
    env = CliEnvelope.model_validate_json(result.stdout)
    assert "--confirm" in env.errors[0]
    assert _status(raman_paths, planned) is RunStatus.PLANNED


def test_cli_cancel_planned_run(planned: str, raman_paths: PathsSettings) -> None:
    code, env = _cli("cancel", planned, "--confirm")
    assert code == ExitCode.OK
    assert env.data is not None
    assert env.data["run"]["status"] == "cancelled"
    assert _status(raman_paths, planned) is RunStatus.CANCELLED
    code, env = _cli("cancel", planned, "--confirm")
    assert code == ExitCode.NOTHING_TO_DO
    assert "already cancelled" in (env.message or "")
    assert _cli("cancel", "0190b3d4-0000-7000-8000-000000000000", "--confirm")[0] == 2


# ---------------------------------------------------------------- run


def test_cli_run_plans_submits_and_waits(dataset_id: str, raman_paths: PathsSettings) -> None:
    code, env = _cli("run", "--dataset", dataset_id, "--priors", str(PRIORS))
    assert code == ExitCode.OK, env.errors
    assert env.data is not None
    run_id = env.data["run_id"]
    assert env.data["run_status"] == "complete"
    assert env.data["n_tasks"] == 1
    assert _status(raman_paths, run_id) is RunStatus.COMPLETE
    assert shards.read_done(shards.task_dir(raman_paths.runs_root / run_id, 0)) is not None


def test_cli_run_refuses_non_local_backend(
    dataset_id: str, raman_paths: PathsSettings, tmp_path: Path
) -> None:
    profile = tmp_path / "cluster.yaml"
    profile.write_text(
        "profile: cluster\nbackend: slurm_array\nslurm:\n  partitions: [normal]\n"
        "  cpus_per_task: 1\n  mem_per_cpu: 1G\n  target_task_minutes: 60\n"
        "  max_array_tasks: 10\n  max_concurrent: 10\n  pixi_bin: /usr/bin/true\n",
        encoding="utf-8",
    )
    argv: list[Any] = ["run", "--dataset", dataset_id, "--priors", str(PRIORS)]
    code, env = _cli(*argv, "--profile", str(profile))
    assert code == ExitCode.VALIDATION
    assert "local backend" in env.errors[0]
    assert Catalog(raman_paths.catalog).list_runs() == []


def test_cli_run_with_invalid_priors_exits_2(dataset_id: str, tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("schema_version: '9'\n", encoding="utf-8")
    code, env = _cli("run", "--dataset", dataset_id, "--priors", str(bad))
    assert code == ExitCode.VALIDATION
    assert env.errors


def test_cli_submit_while_another_process_runs_it_exits_3(
    planned: str, raman_paths: PathsSettings
) -> None:
    folder = raman_paths.runs_root / planned / "local"
    folder.mkdir()
    (folder / task_lock.LOCK_FILE).write_text(json.dumps(OTHER_HOST), encoding="utf-8")
    code, env = _cli("submit", planned)
    assert code == ExitCode.NOTHING_TO_DO
    assert "elsewhere.example" in env.errors[0]


def test_cli_status_lists_20_tasks_then_counts_the_rest(dataset_id: str) -> None:
    argv = ("plan", "--dataset", dataset_id, "--priors", str(PRIORS), "--spectra-per-task", "1")
    run_id = str(_cli(*argv)[1].data["run_id"])  # type: ignore[index]
    result = runner.invoke(app, ["status", run_id])
    assert "  task 19: pending (0 attempts)" in result.stdout
    assert "task 20:" not in result.stdout
    assert "... and 5 more tasks" in result.stdout


def test_cli_resubmit_with_nothing_stopped_exits_3(
    planned: str, raman_paths: PathsSettings
) -> None:
    _move(raman_paths, planned, RunStatus.SUBMITTED, RunStatus.RUNNING)
    for task in (0, 1):
        folder = shards.task_dir(raman_paths.runs_root / planned, task)
        folder.mkdir(parents=True)
        (folder / task_lock.LOCK_FILE).write_text(json.dumps(OTHER_HOST), encoding="utf-8")
    code, env = _cli("submit", planned, "--resubmit-failed")
    assert code == ExitCode.NOTHING_TO_DO
    assert "no stopped tasks to resubmit" in (env.message or "")


def test_cli_submit_with_an_unreadable_plan_exits_4(
    planned: str, raman_paths: PathsSettings
) -> None:
    (raman_paths.runs_root / planned / "run_plan.json").write_text("{}", encoding="utf-8")
    code, env = _cli("submit", planned)
    assert code == ExitCode.BACKEND
    assert "run_plan.json" in env.errors[0]
    assert _status(raman_paths, planned) is RunStatus.PLANNED
