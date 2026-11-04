"""The whole pipeline through the CLI on the fixtures (DESIGN.md §11 I1, I2).

ingest → validate-priors → plan → submit (local backend, spawned workers) → collect → export.
Seed invariance: the same seed gives the same peaks for any task size and worker count.
"""

import io
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pyarrow import ipc
from typer.testing import CliRunner

from raman_bayes.app import run_task as run_task_module
from raman_bayes.cli.main import app
from raman_bayes.cli.output import CliEnvelope, ExitCode
from raman_bayes.config.settings import PathsSettings
from raman_bayes.domain.dataset import QCFlag
from raman_bayes.domain.qc import qc_flags
from raman_bayes.domain.results import (
    PEAKS_SCHEMA,
    SUMMARY_QUANTITIES,
    SUMMARY_STATS,
    is_fits_schema,
)
from raman_bayes.storage.catalog import Catalog
from raman_bayes.storage.zarr_dataset import open_dataset

PRIORS = Path(__file__).parents[1] / "fixtures" / "priors" / "design_example.yaml"
SUMMARIES = [f"{q}_{s}" for q in SUMMARY_QUANTITIES for s in SUMMARY_STATS]
runner = CliRunner()


def _cli(*argv: str, code: int = ExitCode.OK) -> dict[str, Any]:
    result = runner.invoke(app, [*argv, "--json"])
    (line,) = result.stdout.splitlines()
    envelope = CliEnvelope.model_validate_json(line)
    assert result.exit_code == code, envelope.errors
    assert envelope.data is not None
    return envelope.data


@pytest.fixture
def dataset_id(mini_scan: Path, raman_paths: PathsSettings) -> str:
    data = _cli("ingest", str(mini_scan), "--allow-partial-grid")
    (scan,) = data["scans"]
    assert Catalog(raman_paths.catalog).get_dataset(scan["dataset_id"]) is not None
    return str(scan["dataset_id"])


def _run(dataset: str, *plan_options: str) -> str:
    """Plan, submit and collect a run of ``dataset``; return its id."""
    run_id = str(
        _cli("plan", "--dataset", dataset, "--priors", str(PRIORS), *plan_options)["run_id"]
    )
    assert _cli("submit", run_id)["run_status"] == "complete"
    assert _cli("collect", run_id)["run_status"] == "collected"
    return run_id


def _results(paths: PathsSettings, run_id: str, name: str) -> pa.Table:
    return pq.read_table(paths.runs_root / run_id / "results" / name)


def _comparable(table: pa.Table) -> bytes:
    # The run id differs between runs and the metadata carries each run's manifest; every other
    # value must match. Dictionaries are decoded because their order follows the parts.
    plain = table.drop_columns(["run_id"]).replace_schema_metadata(None)
    columns = [
        c.cast(c.type.value_type) if pa.types.is_dictionary(c.type) else c for c in plain.columns
    ]
    decoded = pa.table(columns, names=plain.column_names).combine_chunks()
    sink = io.BytesIO()
    with ipc.new_stream(sink, decoded.schema) as writer:
        writer.write_table(decoded)
    return sink.getvalue()


def test_e2e_fixture_pipeline(dataset_id: str, raman_paths: PathsSettings) -> None:
    valid = _cli("validate-priors", str(PRIORS))
    planned = _cli(
        "plan", "--dataset", dataset_id, "--priors", str(PRIORS), "--spectra-per-task", "16"
    )
    run_id = str(planned["run_id"])
    assert planned["n_tasks"] == 2
    submitted = _cli("submit", run_id)
    assert (submitted["run_status"], [t["status"] for t in submitted["tasks"]]) == (
        "complete",
        ["done", "done"],
    )
    collected = _cli("collect", run_id)
    peaks = _results(raman_paths, run_id, "peaks.parquet")
    fits = _results(raman_paths, run_id, "fits.parquet")
    assert peaks.num_rows == 25 * 2 * 2
    assert fits.num_rows == 25 * 2
    assert peaks.schema.remove_metadata().equals(PEAKS_SCHEMA)
    assert is_fits_schema(fits.schema.remove_metadata())
    assert peaks.column("center_mean").null_count > 0  # absent peaks are null
    manifest = collected["manifest"]
    store = open_dataset(raman_paths.data_root / f"{dataset_id}.zarr")
    assert manifest["dataset_hash"] == store.manifest.content_hash
    assert manifest["priors_hash"] == valid["priors_hash"]
    assert sorted(manifest["shard_hashes"]) == ["0", "1"]
    assert all(len(h) == 64 for h in manifest["shard_hashes"].values())
    assert manifest["partial"] is False

    legacy = _cli("export", run_id, "--format", "legacy-csv")["files"]
    assert [Path(f).name for f in legacy] == ["legacy-lithiation.csv", "legacy-carbon.csv"]
    for path in legacy:
        assert len(Path(path).read_text(encoding="utf-8").splitlines()) == 1 + 25
    (flat,) = _cli("export", run_id, "--format", "csv")["files"]
    assert len(Path(flat).read_text(encoding="utf-8").splitlines()) == 1 + 100


def test_e2e_absent_peaks_null_not_zero(dataset_id: str, raman_paths: PathsSettings) -> None:
    run_id = _run(dataset_id, "--spectra-per-task", "16")
    rows = _results(raman_paths, run_id, "peaks.parquet").to_pylist()
    assert {r["fit_status"] for r in rows} == {"ok"}
    absent = [r for r in rows if r["presence_prob"] < 0.5]
    present = [r for r in rows if r["presence_prob"] >= 0.5]
    assert absent
    assert present
    for row in absent:
        assert [row[c] for c in SUMMARIES] == [None] * len(SUMMARIES)
    for row in present:
        assert None not in [row[c] for c in SUMMARIES]


def test_e2e_synthetic_flat_spectra_give_null_rows(
    synthetic_store: Callable[..., Path], raman_paths: PathsSettings
) -> None:
    n = 6
    counts = np.full((n, 1024), 200.0)
    flags = qc_flags(counts, 5, np.zeros(n, dtype=bool))  # as ingest flags them
    assert set(flags.tolist()) == {QCFlag.SATURATED}  # a flat spectrum is all flat top
    store = synthetic_store(n, counts=counts, flags=dict(enumerate(flags.tolist())))
    Catalog(raman_paths.catalog).add_dataset(open_dataset(store).manifest, store)
    run_id = _run(open_dataset(store).manifest.dataset_id, "--spectra-per-task", "3")
    peaks = _results(raman_paths, run_id, "peaks.parquet")
    assert peaks.num_rows == n * 4
    assert set(peaks.column("fit_status").to_pylist()) == {"skipped_qc"}
    for name in ("presence_prob", *SUMMARIES):
        assert peaks.column(name).null_count == peaks.num_rows, name
        assert peaks.schema.field(name).type == pa.float32()


def test_seed_invariance_1024_vs_8(dataset_id: str, raman_paths: PathsSettings) -> None:
    one_task = _run(dataset_id, "--spectra-per-task", "1024", "--seed", "42")
    many = _run(dataset_id, "--spectra-per-task", "8", "--seed", "42")
    assert len(_cli("status", many)["tasks"]) == 4
    first = _results(raman_paths, one_task, "peaks.parquet")
    second = _results(raman_paths, many, "peaks.parquet")
    assert _comparable(first) == _comparable(second)
    other_seed = _run(dataset_id, "--spectra-per-task", "8", "--seed", "43")
    assert _comparable(_results(raman_paths, other_seed, "peaks.parquet")) != _comparable(first)


def test_seed_invariance_workers_1_vs_4(
    dataset_id: str, raman_paths: PathsSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    peaks = []
    for workers in ("1", "4"):  # local backend: tasks in parallel
        monkeypatch.setenv("RAMAN_LOCAL__WORKERS", workers)
        run_id = _run(dataset_id, "--spectra-per-task", "8", "--seed", "42")
        peaks.append(_results(raman_paths, run_id, "peaks.parquet"))
    monkeypatch.setattr(run_task_module, "SUB_BATCH_SPECTRA", 8)  # four sub-batches per task
    for workers in ("1", "4"):  # run-task --workers: sub-batches in parallel
        options = ("--spectra-per-task", "25", "--seed", "42")
        run_id = str(
            _cli("plan", "--dataset", dataset_id, "--priors", str(PRIORS), *options)["run_id"]
        )
        done = _cli("run-task", "--run", run_id, "--task", "0", "--workers", workers)
        assert done["n_parts_written"] == 4
        assert _cli("collect", run_id, "--partial")["manifest"]["partial"] is False
        peaks.append(_results(raman_paths, run_id, "peaks.parquet"))
    assert len({_comparable(p) for p in peaks}) == 1
