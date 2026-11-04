"""``run_manifest.json`` (DESIGN.md §8.3): the plan's identity, provenance from the done markers."""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from raman_bayes.domain.errors import MixedCodeError
from raman_bayes.domain.markers import DoneMarker
from raman_bayes.domain.run_manifest import RunManifest, build_manifest
from raman_bayes.storage.merge import merge_tasks
from raman_bayes.storage.run_files import read_plan

FinishedTask = Callable[..., DoneMarker]
# §8.3, copied by hand.
SECTION_8_3 = {
    *("run_id", "dataset_id", "dataset_hash", "priors_name", "priors_version", "priors_hash"),
    *("engine", "engine_version", "engine_config", "preprocessing", "run_seed", "backend"),
    *("profile", "package_version", "git_sha", "git_dirty", "pixi_lock_hash", "host"),
    *("slurm_array_job_id", "started_at", "completed_at", "n_spectra", "n_tasks", "n_failed"),
    *("n_skipped_qc", "partial", "shard_hashes"),
}
T0 = datetime(2026, 11, 3, 12, 0, tzinfo=UTC)


@pytest.fixture
def run_dir(mini_store: Path, write_plan: Callable[..., Path]) -> Path:
    return write_plan(mini_store, spectra_per_task=10)  # tasks 0-9, 10-19, 20-24


def _finish(
    run_dir: Path, finished_task: FinishedTask, changes: dict[int, dict[str, Any]] | None = None
) -> dict[int, DoneMarker]:
    plan = read_plan(run_dir)
    markers = {}
    for task in plan.tasks:
        fields = dict((changes or {}).get(task.task_index, {}))
        status = fields.pop("status", "ok")
        ids = range(task.spec_start, task.spec_stop)
        markers[task.task_index] = finished_task(
            run_dir, task.task_index, ids, status=status, **fields
        )
    return markers


def test_manifest_has_all_section_8_3_keys(run_dir: Path, finished_task: FinishedTask) -> None:
    plan = read_plan(run_dir)
    markers = _finish(run_dir, finished_task)
    manifest = build_manifest(plan, markers, merge_tasks(run_dir, markers).fits)
    assert set(manifest.model_dump()) == SECTION_8_3
    assert (manifest.run_id, manifest.dataset_id, manifest.dataset_hash) == (
        plan.run_id,
        plan.dataset_id,
        plan.dataset_hash,
    )
    assert (manifest.priors_name, manifest.priors_version, manifest.priors_hash) == (
        "nmc_commercial",
        "1.0.0",
        plan.priors_hash,
    )
    assert (manifest.engine, manifest.engine_version) == ("mock", "0.1.0")
    assert manifest.engine_config == plan.engine_config
    assert manifest.preprocessing == plan.priors.preprocessing
    assert (manifest.run_seed, manifest.backend, manifest.profile) == (plan.seed, "local", "local")
    assert (manifest.n_spectra, manifest.n_tasks, manifest.partial) == (25, 3, False)
    assert manifest.shard_hashes == {i: m.shard_hash for i, m in markers.items()}
    assert RunManifest.model_validate_json(manifest.model_dump_json()) == manifest


def test_manifest_counts_failed_and_skipped_from_fits(
    run_dir: Path, finished_task: FinishedTask
) -> None:
    # The markers' own counts are deliberately wrong: the manifest counts the merged rows.
    lying = {"n_failed": 999, "n_skipped_qc": 999}
    markers = _finish(
        run_dir,
        finished_task,
        {1: {"status": "failed", **lying}, 2: {"status": "skipped_qc", **lying}},
    )
    manifest = build_manifest(read_plan(run_dir), markers, merge_tasks(run_dir, markers).fits)
    assert (manifest.n_failed, manifest.n_skipped_qc) == (10 * 2, 5 * 2)


def test_partial_manifest_lists_only_finished_tasks(
    run_dir: Path, finished_task: FinishedTask
) -> None:
    markers = _finish(run_dir, finished_task)
    del markers[1]
    manifest = build_manifest(read_plan(run_dir), markers, merge_tasks(run_dir, markers).fits)
    assert manifest.partial is True
    assert set(manifest.shard_hashes) == {0, 2}
    assert manifest.n_tasks == 3


def test_manifest_provenance_from_done_markers(run_dir: Path, finished_task: FinishedTask) -> None:
    markers = _finish(
        run_dir,
        finished_task,
        {
            0: {"host": "node-b", "slurm_array_job_id": "78", "started_at": T0},
            1: {
                "host": "node-a",
                "slurm_array_job_id": "77",
                "completed_at": T0 + timedelta(hours=2),
            },
            2: {"host": "node-b", "started_at": T0 - timedelta(hours=1)},
        },
    )
    manifest = build_manifest(read_plan(run_dir), markers, merge_tasks(run_dir, markers).fits)
    assert (manifest.package_version, manifest.git_sha, manifest.git_dirty) == (
        "0.1.0",
        "abc1234",
        False,
    )
    assert manifest.pixi_lock_hash == "f" * 64
    assert manifest.host == "node-a, node-b"
    assert manifest.slurm_array_job_id == "77, 78"
    assert manifest.started_at == T0 - timedelta(hours=1)
    assert manifest.completed_at == T0 + timedelta(hours=2)


def test_manifest_without_slurm_has_no_array_id(run_dir: Path, finished_task: FinishedTask) -> None:
    markers = _finish(run_dir, finished_task)
    manifest = build_manifest(read_plan(run_dir), markers, merge_tasks(run_dir, markers).fits)
    assert (manifest.host, manifest.slurm_array_job_id) == ("node-a", None)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("package_version", "0.2.0"),
        ("git_sha", "def5678"),
        ("git_dirty", True),
        ("pixi_lock_hash", "e" * 64),
        ("engine_version", "0.1.1"),
    ],
)
def test_mixed_code_versions_refused(
    run_dir: Path, finished_task: FinishedTask, field: str, value: object
) -> None:
    markers = _finish(run_dir, finished_task, {2: {field: value}})
    with pytest.raises(MixedCodeError, match=rf"tasks ran different code.*{field}"):
        build_manifest(read_plan(run_dir), markers, merge_tasks(run_dir, markers).fits)
