"""Ingest Nanofinder scan folders into ``dataset.zarr`` stores and the catalog (DESIGN.md §3, §4).

Each scan is read and checked completely, and its content hash computed in memory, before
anything is written. A dataset already in the catalog is skipped. A new store is written to a
staging folder beside its final path and renamed into place, so a failed ingest leaves no
store; the catalog row is written last. A store found at the final path without a catalog row
is what an interrupted ingest leaves, and is registered as it is. Scans are independent: one
that fails does not stop the others.
"""

import logging
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

import numpy as np
from pydantic import JsonValue

from raman_bayes.config.settings import PathsSettings
from raman_bayes.domain.dataset import DatasetManifest, ScanInfo
from raman_bayes.domain.errors import AxisMismatchError, InputError, ScanFolderError
from raman_bayes.domain.qc import qc_flags
from raman_bayes.io.nanofinder import AXIS_TOLERANCE_CM1, axis_matches, parse_info, read_spectrum
from raman_bayes.io.scan_folder import (
    GridLayout,
    ScanListing,
    find_scan_folders,
    layout_grid,
    list_scan_folder,
)
from raman_bayes.storage.catalog import Catalog
from raman_bayes.storage.zarr_dataset import (
    DatasetArrays,
    content_hash,
    counts_as_float32,
    make_dataset_id,
    open_dataset,
    write_dataset,
)

__all__ = ["PREVIEW_FILES", "IngestPreview", "IngestRequest", "ScanOutcome", "ScanStatus", "ingest"]

PREVIEW_FILES = 10
_log = logging.getLogger(__name__)


class ScanStatus(StrEnum):
    """What ingest did with one scan folder."""

    INGESTED = "ingested"
    EXISTS = "exists"
    DRY_RUN = "dry_run"
    FAILED = "failed"


@dataclass(frozen=True)
class IngestRequest:
    """``raman ingest`` options: a scan folder or a folder of them, and how to treat it."""

    path: Path
    sample: str | None = None
    allow_partial_grid: bool = False
    allow_axis_mismatch: bool = False
    dry_run: bool = False


@dataclass(frozen=True)
class IngestPreview:
    """What a dry run found: the parsed Info.txt, the file count and the first files."""

    info: ScanInfo
    n_files: int
    n_positions: int
    first_files: tuple[str, ...]


@dataclass(frozen=True)
class ScanOutcome:
    """The result for one scan folder.

    ``manifest`` and ``store_uri`` are set when ingested or already there, ``preview`` on a dry
    run, and ``errors`` exactly when the status is failed.
    """

    folder: Path
    status: ScanStatus
    manifest: DatasetManifest | None = None
    store_uri: Path | None = None
    preview: IngestPreview | None = None
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def as_json(self) -> dict[str, JsonValue]:
        """Return the outcome as JSON values, the manifest and preview nested."""
        manifest = self.manifest
        preview = self.preview
        return {
            "folder": str(self.folder),
            "status": self.status.value,
            "dataset_id": manifest.dataset_id if manifest else None,
            "store_uri": str(self.store_uri) if self.store_uri else None,
            "n_spectra": manifest.n_spectra if manifest else None,
            "manifest": manifest.model_dump(mode="json") if manifest else None,
            "preview": None
            if preview is None
            else {
                "info": preview.info.model_dump(mode="json"),
                "n_files": preview.n_files,
                "n_positions": preview.n_positions,
                "first_files": list(preview.first_files),
            },
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class _Scan:
    """A scan read and checked in memory, ready to store."""

    info: ScanInfo
    arrays: DatasetArrays
    warnings: tuple[str, ...]


def ingest(
    request: IngestRequest, paths: PathsSettings, *, now: datetime | None = None
) -> tuple[ScanOutcome, ...]:
    """Ingest every scan folder at ``request.path``; one outcome per folder, in name order.

    ``now`` (default: the current time) is the ``created_at`` of new datasets. Problems with
    the input become failed outcomes; errors writing the store or catalog propagate.
    """
    try:
        folders = find_scan_folders(request.path)
    except ScanFolderError as exc:
        return (ScanOutcome(request.path, ScanStatus.FAILED, errors=(str(exc),)),)
    created_at = now or datetime.now(UTC)
    return tuple(_ingest_folder(folder, request, paths, created_at) for folder in folders)


def _ingest_folder(
    folder: Path, request: IngestRequest, paths: PathsSettings, created_at: datetime
) -> ScanOutcome:
    try:
        listing = list_scan_folder(folder)
        info = _with_partial(parse_info(listing.info_path), request.allow_partial_grid)
        layout = layout_grid(info.scan, listing.spectra, info.grid)
        if request.dry_run:
            preview = IngestPreview(
                info, len(layout.files), info.grid.size, layout.files[:PREVIEW_FILES]
            )
            return ScanOutcome(
                folder, ScanStatus.DRY_RUN, preview=preview, warnings=_ignored(listing)
            )
        scan = _read(listing, info, layout, request.allow_axis_mismatch)
    except (InputError, OSError, UnicodeDecodeError) as exc:
        return ScanOutcome(folder, ScanStatus.FAILED, errors=(str(exc),))
    return _store(folder, scan, request.sample, paths, created_at)


def _with_partial(info: ScanInfo, allow_partial: bool) -> ScanInfo:
    grid = info.grid.model_copy(update={"allow_partial": allow_partial})
    return info.model_copy(update={"grid": grid})


def _ignored(listing: ScanListing) -> tuple[str, ...]:
    if not listing.ignored:
        return ()
    return (f"{listing.folder}: ignored files that are not spectra: {', '.join(listing.ignored)}",)


def _read(
    listing: ScanListing, info: ScanInfo, layout: GridLayout, allow_axis_mismatch: bool
) -> _Scan:
    spectra = [read_spectrum(listing.folder / name, info.n_wn) for name in layout.files]
    # The dataset axis is the first spectrum's in raster order; every file is checked against it.
    axis = spectra[0].wavenumber
    mismatch = np.array([not axis_matches(s.wavenumber, axis) for s in spectra])
    mismatched = [name for name, bad in zip(layout.files, mismatch, strict=True) if bad]
    if mismatched and not allow_axis_mismatch:
        raise AxisMismatchError(listing.folder, mismatched, AXIS_TOLERANCE_CM1)
    raw = np.vstack([s.intensity for s in spectra])
    iz, iy, ix = (column.copy() for column in layout.indices.T)
    grid = info.grid
    arrays = DatasetArrays(
        intensity=counts_as_float32(raw, layout.files),
        wavenumber=axis,
        spectrum_id=np.arange(len(spectra), dtype=np.uint32),
        ix=ix,
        iy=iy,
        iz=iz,
        x_um=grid.x.start_um + ix * grid.x.step_um,
        y_um=grid.y.start_um + iy * grid.y.step_um,
        z_um=grid.z.start_um + iz * grid.z.step_um,
        source_file=layout.files,
        qc_flags=qc_flags(raw, info.acq.n_accum, mismatch),
        grid_mask=layout.mask,
    )
    warnings = _ignored(listing)
    if mismatched:
        flagged = f"{listing.folder}: axis_mismatch flagged on {', '.join(mismatched)}"
        warnings = (*warnings, flagged)
    return _Scan(info=info, arrays=arrays, warnings=warnings)


def _store(
    folder: Path, scan: _Scan, sample: str | None, paths: PathsSettings, created_at: datetime
) -> ScanOutcome:
    arrays = scan.arrays
    digest = content_hash(arrays.intensity, arrays.wavenumber, arrays.source_file)
    dataset_id = make_dataset_id(scan.info.scan, digest)
    catalog = Catalog(paths.catalog)
    existing = catalog.get_dataset(dataset_id)
    if existing is not None:
        _log.info("%s: already ingested as %s; skipped", folder, dataset_id)
        return ScanOutcome(
            folder,
            ScanStatus.EXISTS,
            manifest=existing.manifest,
            store_uri=Path(existing.store_uri),
            warnings=scan.warnings,
        )
    store = paths.data_root / f"{dataset_id}.zarr"
    if store.exists():
        manifest = open_dataset(store).manifest
    else:
        manifest = DatasetManifest.model_validate(
            {
                **scan.info.model_dump(),
                "dataset_id": dataset_id,
                "sample": sample,
                "n_spectra": len(arrays.source_file),
                "content_hash": digest,
                "created_at": created_at,
            }
        )
        _write_atomically(store, manifest, arrays)
    catalog.add_dataset(manifest, store)
    _log.info("%s: ingested as %s", folder, dataset_id)
    return ScanOutcome(
        folder, ScanStatus.INGESTED, manifest=manifest, store_uri=store, warnings=scan.warnings
    )


def _write_atomically(store: Path, manifest: DatasetManifest, arrays: DatasetArrays) -> None:
    store.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{store.name}.", dir=store.parent))
    try:
        write_dataset(staging / store.name, manifest, arrays)
        (staging / store.name).rename(store)
    finally:
        shutil.rmtree(staging)
