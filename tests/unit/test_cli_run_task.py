"""``raman run-task``: exit codes 0, 2, 3 and 4 and the envelope (DESIGN.md §7.2, §10)."""

import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest
from typer.testing import CliRunner

from raman_bayes.cli.main import app
from raman_bayes.cli.output import CliEnvelope, ExitCode
from raman_bayes.config.settings import PathsSettings, load_settings
from raman_bayes.storage import shards, task_lock

WritePlan = Callable[..., Path]
runner = CliRunner()
OTHER_HOST = {"host": "elsewhere.example", "pid": 4242, "started_at": "2026-11-03T11:00:00Z"}
SLURM_SECTION = """\
slurm:
  partitions: [normal]
  cpus_per_task: 1
  mem_per_cpu: 1G
  target_task_minutes: 60
  max_array_tasks: 10
  max_concurrent: 10
  pixi_bin: /usr/bin/true
  lock_stale_minutes: 5
"""


def _invoke(run_dir: Path, task: int = 0, *extra: str) -> tuple[int, CliEnvelope]:
    argv = ["run-task", "--run", run_dir.name, "--task", str(task), *extra, "--json"]
    result = runner.invoke(app, argv)
    (line,) = result.stdout.splitlines()
    return result.exit_code, CliEnvelope.model_validate_json(line)


def _lock_held_elsewhere(run_dir: Path, minutes_old: float) -> None:
    shard = shards.task_dir(run_dir, 0)
    shard.mkdir(parents=True)
    lock = shard / task_lock.LOCK_FILE
    lock.write_text(json.dumps(OTHER_HOST), encoding="utf-8")
    stamp = lock.stat().st_mtime - minutes_old * 60
    os.utime(lock, (stamp, stamp))


def _assert_stopped_after_locking(run_dir: Path, cause: str) -> None:
    # The lock is taken first and left behind, so the task reads as stopped; nothing else is
    # written to its folder, and the task log says why.
    assert sorted(p.name for p in shards.task_dir(run_dir, 0).iterdir()) == [task_lock.LOCK_FILE]
    assert cause in (run_dir / "logs" / "task-0.log").read_text(encoding="utf-8")


def _snapshot(root: Path) -> dict[str, tuple[int, bytes]]:
    return {
        str(p.relative_to(root)): (p.stat().st_mtime_ns, p.read_bytes())
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_cli_run_task_exit_codes(mini_store: Path, write_plan: WritePlan) -> None:
    run_dir = write_plan(mini_store)
    code, env = _invoke(run_dir, 1)
    assert code == ExitCode.OK
    assert env.ok
    assert env.command == "run-task"
    assert env.data is not None
    assert env.data["status"] == "done"
    assert (env.data["run_id"], env.data["task_index"]) == (run_dir.name, 1)
    assert (env.data["spec_start"], env.data["spec_stop"]) == (16, 25)
    assert (env.data["n_parts_written"], env.data["n_parts_kept"]) == (1, 0)
    assert env.data["done"]["n_fits"] == 18
    assert env.data["lock"] is None
    assert "task 1 of run" in (env.message or "")

    code, env = _invoke(run_dir, 1)
    assert code == ExitCode.NOTHING_TO_DO
    assert env.data is not None
    assert env.data["status"] == "already_done"

    shutil.rmtree(mini_store)
    code, env = _invoke(run_dir, 0)
    assert code == ExitCode.BACKEND
    assert env.errors


def test_cli_run_task_text_report(mini_store: Path, write_plan: WritePlan) -> None:
    run_dir = write_plan(mini_store)
    result = runner.invoke(app, ["run-task", "--run", run_dir.name, "--task", "0"])
    assert result.exit_code == 0, result.output
    assert f"task 0 of run {run_dir.name}: done" in result.stdout
    assert "32 fits (32 ok, 0 failed, 0 skipped_qc)" in result.stdout


def test_fresh_lock_exits_3(mini_store: Path, write_plan: WritePlan) -> None:
    run_dir = write_plan(mini_store)
    _lock_held_elsewhere(run_dir, minutes_old=1)
    code, env = _invoke(run_dir)
    assert code == ExitCode.NOTHING_TO_DO
    assert env.data is not None
    assert env.data["status"] == "already_running"
    assert env.data["lock"]["host"] == "elsewhere.example"
    assert "already running" in (env.message or "")
    assert "elsewhere.example" in (env.message or "")


def test_done_task_exits_3_untouched(mini_store: Path, write_plan: WritePlan) -> None:
    run_dir = write_plan(mini_store)
    assert _invoke(run_dir)[0] == ExitCode.OK
    before = _snapshot(run_dir)
    code, _ = _invoke(run_dir)
    assert code == ExitCode.NOTHING_TO_DO
    assert _snapshot(run_dir) == before


def test_lock_stale_minutes_default_30_and_profile_override(
    mini_store: Path, write_plan: WritePlan, tmp_path: Path
) -> None:
    assert load_settings().settings.lock_stale_minutes == 30
    profile = tmp_path / "quick.yaml"
    profile.write_text(f"profile: quick\nbackend: local\n{SLURM_SECTION}", encoding="utf-8")
    assert load_settings(str(profile)).settings.lock_stale_minutes == 5

    run_dir = write_plan(mini_store)
    _lock_held_elsewhere(run_dir, minutes_old=10)
    assert _invoke(run_dir)[0] == ExitCode.NOTHING_TO_DO  # 10 minutes is fresh under 30
    assert _invoke(run_dir, 0, "--profile", str(profile))[0] == ExitCode.OK


def test_missing_store_exits_4(mini_store: Path, write_plan: WritePlan) -> None:
    run_dir = write_plan(mini_store)
    shutil.rmtree(mini_store)
    code, env = _invoke(run_dir)
    assert code == ExitCode.BACKEND
    assert str(mini_store) in env.errors[0]
    _assert_stopped_after_locking(run_dir, str(mini_store))


def test_store_that_changed_since_planning_exits_4(
    mini_store: Path, write_plan: WritePlan, synthetic_store: Callable[..., Path]
) -> None:
    run_dir = write_plan(mini_store)
    shutil.rmtree(mini_store)
    shutil.copytree(synthetic_store(25), mini_store)
    code, env = _invoke(run_dir)
    assert code == ExitCode.BACKEND
    assert "not the plan's" in env.errors[0]


@pytest.mark.parametrize(
    ("engine", "version", "says"),
    [("mock", "9.9.9", "mock 0.1.0 is installed"), ("nonesuch", "1.0", "no engine named")],
)
def test_engine_version_mismatch_exits_4(
    mini_store: Path, write_plan: WritePlan, engine: str, version: str, says: str
) -> None:
    run_dir = write_plan(mini_store, engine=engine, engine_version=version)
    code, env = _invoke(run_dir)
    assert code == ExitCode.BACKEND
    assert f"planned with engine {engine} {version}" in env.errors[0]
    assert says in env.errors[0]
    _assert_stopped_after_locking(run_dir, says)


def test_unknown_run_exits_2(raman_paths: PathsSettings) -> None:
    result = runner.invoke(app, ["run-task", "--run", "no-such-run", "--task", "0", "--json"])
    assert result.exit_code == ExitCode.VALIDATION
    env = CliEnvelope.model_validate_json(result.stdout)
    assert "no run plan" in env.errors[0]
    assert not raman_paths.runs_root.exists()


def test_task_index_out_of_range_exits_2(mini_store: Path, write_plan: WritePlan) -> None:
    run_dir = write_plan(mini_store)
    code, env = _invoke(run_dir, 2)
    assert code == ExitCode.VALIDATION
    assert "has 2 tasks (0 to 1); there is no task 2" in env.errors[0]
    assert not (run_dir / "shards").exists()


def test_unreadable_plan_exits_4(mini_store: Path, write_plan: WritePlan) -> None:
    run_dir = write_plan(mini_store)
    (run_dir / "run_plan.json").write_text("{}", encoding="utf-8")
    code, env = _invoke(run_dir)
    assert code == ExitCode.BACKEND
    assert "run_plan.json" in env.errors[0]


def test_plan_copied_into_another_run_folder_exits_4(
    mini_store: Path, write_plan: WritePlan
) -> None:
    run_dir = write_plan(mini_store)
    other = write_plan(mini_store)
    shutil.copy(run_dir / "run_plan.json", other / "run_plan.json")
    code, env = _invoke(other)
    assert code == ExitCode.BACKEND
    assert f"it is the plan of run {run_dir.name}" in env.errors[0]
