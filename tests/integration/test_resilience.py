"""Interruptions, duplicates and incomplete runs (DESIGN.md §11 I3, I4, I5, I6).

The real ``raman`` console script runs tasks in subprocesses here, so they can be killed and
raced; the profile's paths reach them through the ``RAMAN_PATHS__*`` variables.
"""

import os
import shutil
import signal
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from raman_bayes.cli.main import app
from raman_bayes.cli.output import CliEnvelope, ExitCode
from raman_bayes.config.settings import PathsSettings
from raman_bayes.domain.dataset import QCFlag
from raman_bayes.storage import shards, task_lock
from raman_bayes.storage.catalog import Catalog
from raman_bayes.storage.zarr_dataset import open_dataset

PRIORS = Path(__file__).parents[1] / "fixtures" / "priors" / "design_example.yaml"
N_SPECTRA = 1250  # the full synthetic grid: three sub-batches, so a task can be killed between
runner = CliRunner()


def _cli(*argv: str, code: int = ExitCode.OK) -> dict[str, Any]:
    result = runner.invoke(app, [*argv, "--json"])
    (line,) = result.stdout.splitlines()
    envelope = CliEnvelope.model_validate_json(line)
    assert result.exit_code == code, (envelope.message, envelope.errors)
    assert envelope.data is not None
    return envelope.data


def _registered(paths: PathsSettings, store: Path) -> str:
    manifest = open_dataset(store).manifest
    Catalog(paths.catalog).add_dataset(manifest, store)
    return manifest.dataset_id


def _plan(dataset: str, size: int, seed: int = 9) -> str:
    options = ("--spectra-per-task", str(size), "--seed", str(seed))
    return str(_cli("plan", "--dataset", dataset, "--priors", str(PRIORS), *options)["run_id"])


def _raman(*argv: str) -> subprocess.Popen[str]:
    exe = shutil.which("raman")
    assert exe is not None, "raman not on PATH; run via pixi"
    return subprocess.Popen([exe, *argv], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _peaks(paths: PathsSettings, run_id: str) -> pa.Table:
    table = pq.read_table(paths.runs_root / run_id / "results" / "peaks.parquet")
    return table.drop_columns(["run_id"]).replace_schema_metadata(None).combine_chunks()


def _reference(paths: PathsSettings, dataset: str) -> pa.Table:
    run_id = _plan(dataset, N_SPECTRA)
    _cli("run-task", "--run", run_id, "--task", "0")
    _cli("collect", run_id, "--partial")
    return _peaks(paths, run_id)


@pytest.fixture
def big_dataset(synthetic_store: Callable[..., Path], raman_paths: PathsSettings) -> str:
    return _registered(raman_paths, synthetic_store(N_SPECTRA))


def _wait_for(path: Path, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 120
    while not path.exists():
        assert process.poll() is None, process.communicate()
        assert time.monotonic() < deadline, f"{path} never appeared"
        time.sleep(0.02)


def test_kill_and_resume_identical_no_duplicates(
    big_dataset: str, raman_paths: PathsSettings
) -> None:
    run_id = _plan(big_dataset, N_SPECTRA)
    shard = shards.task_dir(raman_paths.runs_root / run_id, 0)
    with _raman("run-task", "--run", run_id, "--task", "0") as task:
        _wait_for(shards.part_paths(shard, 0).peaks, task)
        task.send_signal(signal.SIGTERM)
        assert task.wait(timeout=60) == -signal.SIGTERM
    assert shards.read_done(shard) is None
    assert (shard / task_lock.LOCK_FILE).is_file()  # left by the killed process

    resumed = _cli("run-task", "--run", run_id, "--task", "0")  # takes over the dead lock
    assert resumed["n_parts_kept"] >= 1
    assert resumed["n_parts_written"] + resumed["n_parts_kept"] == 3
    _cli("collect", run_id, "--partial")  # merge refuses duplicate rows
    peaks = _peaks(raman_paths, run_id)
    assert peaks.num_rows == N_SPECTRA * 4
    assert peaks.equals(_reference(raman_paths, big_dataset))


def test_concurrent_duplicate_one_exits_3_results_intact(
    big_dataset: str, raman_paths: PathsSettings
) -> None:
    run_id = _plan(big_dataset, N_SPECTRA)
    racers = [_raman("run-task", "--run", run_id, "--task", "0", "--json") for _ in range(2)]
    codes = sorted(racer.wait(timeout=300) for racer in racers)
    assert codes == [ExitCode.OK, ExitCode.NOTHING_TO_DO]
    for racer in racers:
        out, _ = racer.communicate()
        envelope = CliEnvelope.model_validate_json(out)
        assert envelope.data is not None
        assert envelope.data["status"] in {"done", "already_running", "already_done"}
    shard = shards.task_dir(raman_paths.runs_root / run_id, 0)
    assert shards.read_done(shard) is not None
    assert not (shard / task_lock.LOCK_FILE).exists()
    _cli("collect", run_id, "--partial")
    assert _peaks(raman_paths, run_id).equals(_reference(raman_paths, big_dataset))


def test_merge_with_all_null_shard(
    synthetic_store: Callable[..., Path], raman_paths: PathsSettings
) -> None:
    flags = dict.fromkeys(range(4, 8), int(QCFlag.SATURATED))  # task 1 is skipped entirely
    dataset = _registered(raman_paths, synthetic_store(8, flags=flags))
    run_id = _plan(dataset, 4)
    _cli("submit", run_id)
    assert _cli("collect", run_id)["run_status"] == "collected"
    peaks = pq.read_table(raman_paths.runs_root / run_id / "results" / "peaks.parquet")
    null_task = peaks.slice(16, 16)
    assert set(null_task.column("spectrum_id").to_pylist()) == {4, 5, 6, 7}
    for name in ("presence_prob", "center_mean", "area_q975", "fwhm_at_bound_frac"):
        assert null_task.column(name).null_count == 16
        assert peaks.schema.field(name).type == pa.float32()


def test_collect_incomplete_refuses_partial_succeeds(
    mini_scan: Path, raman_paths: PathsSettings
) -> None:
    (scan,) = _cli("ingest", str(mini_scan), "--allow-partial-grid")["scans"]
    run_id = _plan(str(scan["dataset_id"]), 16)
    _cli("run-task", "--run", run_id, "--task", "0")
    refused = runner.invoke(app, ["collect", run_id, "--json"])
    assert refused.exit_code == ExitCode.VALIDATION
    assert "--partial" in CliEnvelope.model_validate_json(refused.stdout).errors[0]
    assert not (raman_paths.runs_root / run_id / "results").exists()
    partial = _cli("collect", run_id, "--partial")
    assert (partial["status"], partial["run_status"]) == ("partial", "planned")
    assert partial["manifest"]["partial"] is True
    assert list(partial["manifest"]["shard_hashes"]) == ["0"]
    assert pq.read_metadata(Path(partial["files"][1])).num_rows == 16 * 4
    assert os.path.basename(partial["files"][1]) == "peaks.parquet"
