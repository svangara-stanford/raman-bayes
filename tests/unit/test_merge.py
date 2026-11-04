"""Merging shard parts into ``results/`` (DESIGN.md §8.2, §8.3; §11 I3, I5)."""

import json
from collections.abc import Callable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from raman_bayes.app.run_task import TaskRequest, run_task
from raman_bayes.domain.dataset import QCFlag
from raman_bayes.domain.errors import DuplicateRowsError, PartSchemaError, ShardMismatchError
from raman_bayes.domain.markers import DoneMarker
from raman_bayes.domain.results import FITS_SCHEMA, PEAKS_SCHEMA
from raman_bayes.storage import merge, shards
from raman_bayes.storage.merge import MergedTables, merge_tasks, write_results

FinishedTask = Callable[..., DoneMarker]
DICT_STR = pa.dictionary(pa.int32(), pa.string())


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    path = tmp_path / "runs" / "0190b3d4-0000-7000-8000-000000000000"
    path.mkdir(parents=True)
    return path


def _markers(*markers: DoneMarker) -> dict[int, DoneMarker]:
    return {m.task_index: m for m in markers}


def _write(results: Path, tables: MergedTables) -> None:
    write_results(results, tables, json.dumps({"run_id": "r"}))


def test_merge_rejects_schema_mismatch(run_dir: Path, finished_task: FinishedTask) -> None:
    first = finished_task(run_dir, 0, [0, 1])
    second = finished_task(run_dir, 1, [2, 3])
    path = shards.part_paths(shards.task_dir(run_dir, 1), 0).peaks
    table = pq.read_table(path)
    widened = table.set_column(
        table.schema.get_field_index("center_mean"),
        pa.field("center_mean", pa.float64()),
        table.column("center_mean").cast(pa.float64()),
    )
    pq.write_table(widened, path)
    rehashed = second.model_copy(
        update={"shard_hash": shards.summarize_parts(shards.task_dir(run_dir, 1)).shard_hash}
    )
    with pytest.raises(PartSchemaError, match=r"peaks-000\.parquet: not written with the peaks"):
        merge_tasks(run_dir, _markers(first, rehashed))


def test_merge_rejects_a_fits_part_with_a_float64_diagnostic(
    run_dir: Path, finished_task: FinishedTask
) -> None:
    marker = finished_task(run_dir, 0, [0])
    folder = shards.task_dir(run_dir, 0)
    path = shards.part_paths(folder, 0).fits
    table = pq.read_table(path)
    pq.write_table(table.append_column("diag_rhat", pa.array([1.0] * table.num_rows)), path)
    rehashed = marker.model_copy(update={"shard_hash": shards.summarize_parts(folder).shard_hash})
    with pytest.raises(PartSchemaError, match=r"fits-000\.parquet: not written with the fits"):
        merge_tasks(run_dir, _markers(rehashed))


def test_merge_all_null_shard_keeps_float32(
    synthetic_store: Callable[..., Path], write_plan: Callable[..., Path]
) -> None:
    flags = {i: QCFlag.SATURATED for i in range(4, 8)}  # task 1 is skipped entirely
    run_dir = write_plan(synthetic_store(8, flags=flags), spectra_per_task=4)
    markers = {}
    for task in (0, 1):
        outcome = run_task(TaskRequest(run_dir, task))
        assert outcome.marker is not None
        markers[task] = outcome.marker
    tables = merge_tasks(run_dir, markers)
    _write(run_dir / "results", tables)
    peaks = pq.read_table(run_dir / "results" / "peaks.parquet")
    assert peaks.schema.remove_metadata().equals(PEAKS_SCHEMA)
    skipped = peaks.filter(pa.compute.greater_equal(peaks.column("spectrum_id"), 4))
    assert skipped.num_rows == 16
    assert skipped.column("center_mean").null_count == 16
    assert skipped.column("center_mean").type == pa.float32()


def test_peaks_sorted_with_64k_row_groups(run_dir: Path, finished_task: FinishedTask) -> None:
    later = finished_task(run_dir, 1, [5, 6])
    earlier = finished_task(run_dir, 0, [0, 1])
    tables = merge_tasks(run_dir, {1: later, 0: earlier})
    assert tables.peaks.column("spectrum_id").to_pylist() == [0] * 4 + [1] * 4 + [5] * 4 + [6] * 4
    assert tables.peaks.column("peak_id").to_pylist()[:4] == ["u1", "u2", "d", "g"]
    big = MergedTables(
        fits=tables.fits,
        peaks=pa.concat_tables([tables.peaks] * 4100),  # 65,600 rows
    )
    _write(run_dir / "results", big)
    metadata = pq.read_metadata(run_dir / "results" / "peaks.parquet")
    assert merge.ROW_GROUP_ROWS == 65536
    assert [metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)] == [
        65536,
        64,
    ]


def test_fits_diag_columns_unioned_as_float32(run_dir: Path, finished_task: FinishedTask) -> None:
    first = finished_task(run_dir, 0, [0], diagnostics={"rhat": 1.01})
    second = finished_task(run_dir, 1, [1], diagnostics={"ess": 400.0})
    third = finished_task(run_dir, 2, [2])
    fits = merge_tasks(run_dir, _markers(first, second, third)).fits
    assert fits.schema.names == [*FITS_SCHEMA.names, "diag_ess", "diag_rhat"]
    for name in ("diag_ess", "diag_rhat"):
        assert fits.schema.field(name).type == pa.float32()
    assert fits.column("diag_rhat").to_pylist() == pytest.approx(
        [1.01, 1.01, None, None, None, None]
    )
    assert fits.column("diag_ess").to_pylist() == [None, None, 400.0, 400.0, None, None]


def test_kv_metadata_has_manifest_and_schema_version(
    run_dir: Path, finished_task: FinishedTask
) -> None:
    tables = merge_tasks(run_dir, _markers(finished_task(run_dir, 0, [0, 1])))
    manifest = json.dumps({"run_id": run_dir.name, "partial": False})
    written = write_results(run_dir / "results", tables, manifest)
    assert sorted(p.name for p in written) == ["fits.parquet", "peaks.parquet", "run_manifest.json"]
    for name in ("peaks.parquet", "fits.parquet"):
        metadata = pq.read_schema(run_dir / "results" / name).metadata
        assert metadata[b"run_manifest.json"].decode() == manifest
        assert metadata[b"schema_version"] == b"1.0"
    assert (run_dir / "results" / "run_manifest.json").read_text("utf-8") == manifest
    assert list((run_dir / "results").glob("*.tmp")) == []


def test_dictionary_columns_preserved(run_dir: Path, finished_task: FinishedTask) -> None:
    tables = merge_tasks(
        run_dir, _markers(finished_task(run_dir, 0, [0]), finished_task(run_dir, 1, [1]))
    )
    _write(run_dir / "results", tables)
    peaks = pq.read_table(run_dir / "results" / "peaks.parquet")
    fits = pq.read_table(run_dir / "results" / "fits.parquet")
    for table, names in (
        (peaks, ("run_id", "dataset_id", "window_id", "end_member_id", "peak_id", "fit_status")),
        (fits, ("run_id", "window_id", "status", "engine", "engine_version")),
    ):
        for name in names:
            assert table.schema.field(name).type == DICT_STR, name
    assert peaks.schema.remove_metadata().equals(PEAKS_SCHEMA)


def test_merge_rejects_duplicate_keys(run_dir: Path, finished_task: FinishedTask) -> None:
    first = finished_task(run_dir, 0, [0, 1])
    overlapping = finished_task(run_dir, 1, [1, 2])
    with pytest.raises(DuplicateRowsError, match="spectrum 1"):
        merge_tasks(run_dir, _markers(first, overlapping))


@pytest.mark.parametrize("damage", ["rewritten_part", "extra_part", "missing_part"])
def test_shard_hashes_checked_against_done_markers(
    run_dir: Path, finished_task: FinishedTask, damage: str
) -> None:
    marker = finished_task(run_dir, 0, [0, 1])
    folder = shards.task_dir(run_dir, 0)
    if damage == "rewritten_part":
        finished_task(run_dir, 0, [0, 1], status="failed")  # new parts and marker ...
        shards.write_done(folder, marker)  # ... then the old marker back
    elif damage == "extra_part":
        (folder / "fits-001.parquet").write_bytes((folder / "fits-000.parquet").read_bytes())
    else:
        (folder / "peaks-000.parquet").unlink()
    with pytest.raises(ShardMismatchError, match="task=00000"):
        merge_tasks(run_dir, _markers(marker))
