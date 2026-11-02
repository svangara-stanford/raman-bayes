"""``app.ingest``: scan folders -> dataset.zarr + catalog row (DESIGN.md §3, §4, §8.5, §10)."""

import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from raman_bayes.app import ingest as ingest_module
from raman_bayes.app.ingest import IngestRequest, ScanOutcome, ScanStatus, ingest
from raman_bayes.cli.main import app
from raman_bayes.cli.output import CliEnvelope, ExitCode
from raman_bayes.config.settings import PathsSettings
from raman_bayes.domain.dataset import DatasetManifest, QCFlag
from raman_bayes.storage.catalog import Catalog
from raman_bayes.storage.zarr_dataset import DatasetArrays, open_dataset, write_dataset

NOW = datetime(2026, 11, 2, 12, 0, tzinfo=UTC)
ReplaceOnce = Callable[[Path, bytes, bytes], None]
runner = CliRunner()


def _one(request: IngestRequest, paths: PathsSettings) -> ScanOutcome:
    (outcome,) = ingest(request, paths, now=NOW)
    return outcome


def _partial(
    path: Path, *, dry_run: bool = False, allow_axis_mismatch: bool = False
) -> IngestRequest:
    return IngestRequest(
        path=path,
        allow_partial_grid=True,
        allow_axis_mismatch=allow_axis_mismatch,
        dry_run=dry_run,
    )


def _cli(*argv: str) -> tuple[int, CliEnvelope]:
    result = runner.invoke(app, ["ingest", *argv, "--json"])
    (line,) = result.stdout.splitlines()
    return result.exit_code, CliEnvelope.model_validate_json(line)


def _copy_as(scan: Path, new: str) -> Path:
    # A second scan beside the first: same spectra, every name changed to ``new``.
    target = Path(shutil.copytree(scan, scan.with_name(new)))
    for path in target.iterdir():
        path.rename(target / path.name.replace(f"{scan.name}-", f"{new}-", 1))
    return target


def _shift_axis(spectrum: Path, replace_once: ReplaceOnce, delta: float) -> None:
    first = spectrum.read_bytes().split(b"\r\n", 1)[0]
    shift, count = first.split(b"\t")
    moved = f"{float(shift) + delta:.5f}".encode() + b"\t" + count
    replace_once(spectrum, first + b"\r\n", moved + b"\r\n")


def test_ingest_fixture_allow_partial(mini_scan: Path, raman_paths: PathsSettings) -> None:
    outcome = _one(IngestRequest(path=mini_scan, sample="S1", allow_partial_grid=True), raman_paths)
    assert outcome.status is ScanStatus.INGESTED, outcome.errors
    manifest = outcome.manifest
    assert manifest is not None
    assert (manifest.scan, manifest.sample, manifest.n_spectra) == ("NMC811_v0", "S1", 25)
    assert manifest.grid.allow_partial is True
    assert (manifest.grid.nx, manifest.grid.ny, manifest.grid.nz) == (50, 25, 1)
    assert manifest.created_at == NOW
    assert outcome.store_uri == raman_paths.data_root / f"{manifest.dataset_id}.zarr"
    stored = open_dataset(outcome.store_uri)
    assert stored.manifest == manifest
    arrays = stored.read()
    assert arrays.intensity.shape == (25, 1024)
    assert arrays.wavenumber[0] == -41.55539
    assert arrays.grid_mask.shape == (1, 25, 50)
    assert arrays.grid_mask.sum() == 25
    assert arrays.grid_mask[0, :5, :5].all()


def test_ingest_fixture_without_allow_partial_fails_count_check(
    mini_scan: Path, raman_paths: PathsSettings
) -> None:
    outcome = _one(IngestRequest(path=mini_scan), raman_paths)
    assert outcome.status is ScanStatus.FAILED
    (error,) = outcome.errors
    assert "25 of 1250" in error
    assert not raman_paths.data_root.exists()
    assert not raman_paths.catalog.exists()


def test_complete_grid_needs_no_allow_partial(
    corner_scan: Path, raman_paths: PathsSettings
) -> None:
    outcome = _one(IngestRequest(path=corner_scan), raman_paths)
    assert outcome.status is ScanStatus.INGESTED, outcome.errors
    assert outcome.manifest is not None
    assert outcome.manifest.grid.allow_partial is False


def test_second_ingest_is_noop_exit_3(mini_scan: Path, raman_paths: PathsSettings) -> None:
    code, first = _cli(str(mini_scan), "--allow-partial-grid")
    assert code == ExitCode.OK, first
    assert first.data is not None
    (scan,) = first.data["scans"]
    store = Path(scan["store_uri"])
    stamp = (store / "zarr.json").stat().st_mtime_ns
    code, second = _cli(str(mini_scan), "--allow-partial-grid")
    assert code == ExitCode.NOTHING_TO_DO
    assert second.data is not None
    assert second.data["scans"][0]["status"] == "exists"
    assert second.data["scans"][0]["dataset_id"] == scan["dataset_id"]
    assert (store / "zarr.json").stat().st_mtime_ns == stamp
    assert len(Catalog(raman_paths.catalog).list_datasets()) == 1


def test_ingest_registers_dataset_in_catalog(mini_scan: Path, raman_paths: PathsSettings) -> None:
    outcome = _one(_partial(mini_scan), raman_paths)
    assert outcome.manifest is not None
    row = Catalog(raman_paths.catalog).get_dataset(outcome.manifest.dataset_id)
    assert row is not None
    assert row.store_uri == str(outcome.store_uri)
    assert row.manifest == outcome.manifest
    assert (row.scan, row.n_spectra, row.n_wn) == ("NMC811_v0", 25, 1024)
    assert row.content_hash == outcome.manifest.content_hash


def test_store_without_catalog_row_is_registered(
    mini_scan: Path, raman_paths: PathsSettings
) -> None:
    # An ingest interrupted between renaming the store into place and the catalog write.
    first = _one(_partial(mini_scan), raman_paths)
    assert first.manifest is not None
    raman_paths.catalog.unlink()
    again = _one(_partial(mini_scan), raman_paths)
    assert again.status is ScanStatus.INGESTED
    row = Catalog(raman_paths.catalog).get_dataset(first.manifest.dataset_id)
    assert row is not None
    assert row.manifest == first.manifest  # the store's manifest, not a new created_at


def test_ingest_parent_folder_ingests_each_scan(
    mini_scan: Path, raman_paths: PathsSettings
) -> None:
    other = _copy_as(mini_scan, "NMC811_v9")
    outcomes = ingest(_partial(mini_scan.parent), raman_paths, now=NOW)
    assert [o.folder for o in outcomes] == [mini_scan, other]
    assert [o.status for o in outcomes] == [ScanStatus.INGESTED] * 2
    ids = [o.manifest.dataset_id for o in outcomes if o.manifest is not None]
    assert ids[0].startswith("NMC811_v0-")
    assert ids[1].startswith("NMC811_v9-")
    assert len(Catalog(raman_paths.catalog).list_datasets()) == 2


def test_one_failed_scan_does_not_stop_the_others(
    mini_scan: Path, raman_paths: PathsSettings, replace_once: ReplaceOnce
) -> None:
    broken = _copy_as(mini_scan, "NMC811_v9")
    replace_once(
        broken / "NMC811_v9-Info.txt",
        b"Spectral series number (maximum) : 1",
        b"Spectral series number (maximum) : 2",
    )
    first, second = ingest(_partial(mini_scan.parent), raman_paths, now=NOW)
    assert first.status is ScanStatus.INGESTED
    assert second.status is ScanStatus.FAILED


def test_dry_run_reports_and_writes_nothing(mini_scan: Path, raman_paths: PathsSettings) -> None:
    outcome = _one(_partial(mini_scan, dry_run=True), raman_paths)
    assert outcome.status is ScanStatus.DRY_RUN, outcome.errors
    preview = outcome.preview
    assert preview is not None
    assert preview.info.scan == "NMC811_v0"
    assert preview.info.grid.allow_partial is True
    assert (preview.n_files, preview.n_positions) == (25, 1250)
    assert preview.first_files == tuple(f"NMC811_v0-z0y{i // 5}x{i % 5}.txt" for i in range(10))
    assert outcome.manifest is None
    assert not raman_paths.data_root.exists()
    assert not raman_paths.catalog.exists()


def test_dry_run_reports_grid_errors(mini_scan: Path, raman_paths: PathsSettings) -> None:
    outcome = _one(IngestRequest(path=mini_scan, dry_run=True), raman_paths)
    assert outcome.status is ScanStatus.FAILED
    assert "25 of 1250" in outcome.errors[0]


def test_axis_mismatch_fails_unless_allowed(
    mini_scan: Path, raman_paths: PathsSettings, replace_once: ReplaceOnce
) -> None:
    _shift_axis(mini_scan / "NMC811_v0-z0y2x3.txt", replace_once, 0.1)
    refused = _one(_partial(mini_scan), raman_paths)
    assert refused.status is ScanStatus.FAILED
    (error,) = refused.errors
    assert "NMC811_v0-z0y2x3.txt" in error
    assert "--allow-axis-mismatch" in error
    assert not raman_paths.data_root.exists()

    allowed = _one(_partial(mini_scan, allow_axis_mismatch=True), raman_paths)
    assert allowed.status is ScanStatus.INGESTED
    assert allowed.store_uri is not None
    assert any("NMC811_v0-z0y2x3.txt" in w for w in allowed.warnings)
    arrays = open_dataset(allowed.store_uri).read()
    flagged = [
        name
        for name, flags in zip(arrays.source_file, arrays.qc_flags, strict=True)
        if flags & QCFlag.AXIS_MISMATCH
    ]
    assert flagged == ["NMC811_v0-z0y2x3.txt"]


def test_axis_within_tolerance_is_not_flagged(
    mini_scan: Path, raman_paths: PathsSettings, replace_once: ReplaceOnce
) -> None:
    _shift_axis(mini_scan / "NMC811_v0-z0y2x3.txt", replace_once, 0.04)
    outcome = _one(_partial(mini_scan), raman_paths)
    assert outcome.status is ScanStatus.INGESTED
    assert outcome.store_uri is not None
    assert not np.any(open_dataset(outcome.store_uri).read().qc_flags & QCFlag.AXIS_MISMATCH)


def test_failed_ingest_leaves_no_store(
    mini_scan: Path, raman_paths: PathsSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def write_then_fail(path: Path, manifest: DatasetManifest, arrays: DatasetArrays) -> None:
        write_dataset(path, manifest, arrays)
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(ingest_module, "write_dataset", write_then_fail)
    with pytest.raises(OSError, match="No space left"):
        ingest(_partial(mini_scan), raman_paths, now=NOW)
    assert list(raman_paths.data_root.iterdir()) == []
    assert Catalog(raman_paths.catalog).list_datasets() == []


def test_n_series_2_exits_2_with_time_series_message(
    mini_scan: Path, raman_paths: PathsSettings, replace_once: ReplaceOnce
) -> None:
    replace_once(
        mini_scan / "NMC811_v0-Info.txt",
        b"Spectral series number (maximum) : 1",
        b"Spectral series number (maximum) : 2",
    )
    code, envelope = _cli(str(mini_scan), "--allow-partial-grid")
    assert code == ExitCode.VALIDATION
    (error,) = envelope.errors
    assert "time series not supported yet" in error
    assert not raman_paths.data_root.exists()


def test_folder_name_must_equal_scan(mini_scan: Path, raman_paths: PathsSettings) -> None:
    renamed = mini_scan.rename(mini_scan.with_name("renamed"))
    outcome = _one(_partial(renamed), raman_paths)
    assert outcome.status is ScanStatus.FAILED
    (error,) = outcome.errors
    assert "NMC811_v0-Info.txt" in error
    assert "'renamed'" in error


def test_path_without_scan_folder_fails(tmp_path: Path, raman_paths: PathsSettings) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    for path in (empty, tmp_path / "absent"):
        outcome = _one(_partial(path), raman_paths)
        assert outcome.status is ScanStatus.FAILED
        assert str(path) in outcome.errors[0]


def test_unrecognised_files_are_warnings(mini_scan: Path, raman_paths: PathsSettings) -> None:
    (mini_scan / "notes.txt").write_text("not a spectrum")
    (mini_scan / ".DS_Store").write_bytes(b"\0")
    outcome = _one(_partial(mini_scan), raman_paths)
    assert outcome.status is ScanStatus.INGESTED
    assert outcome.manifest is not None
    assert outcome.manifest.n_spectra == 25
    assert any("notes.txt" in w for w in outcome.warnings)
