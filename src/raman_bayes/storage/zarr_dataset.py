"""The ``dataset.zarr`` store (DESIGN.md §4): Zarr v3, one row per spectrum, plus identity.

``content_hash`` is BLAKE3 over the arrays as written (float32 intensity, float64 wavenumber,
sorted source-file basenames), so a dataset's id depends neither on where the raw files were
nor on how their numbers were formatted. Ingest computes it in memory before writing, so an
existing dataset is recognised without writing anything. The root attrs repeat the manifest's
``content_hash`` and ``created_at``, and the reader refuses a store whose copies disagree.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import blake3
import numpy as np
import numpy.typing as npt
import zarr
from pydantic import ValidationError
from zarr.codecs import BloscCodec

from raman_bayes.domain.dataset import DatasetManifest
from raman_bayes.domain.errors import CountTooLargeError, DatasetStoreError

__all__ = [
    "MAX_EXACT_COUNT",
    "STORE_SCHEMA_VERSION",
    "DatasetArrays",
    "StoredDataset",
    "content_hash",
    "counts_as_float32",
    "make_dataset_id",
    "open_dataset",
    "write_dataset",
]

STORE_SCHEMA_VERSION = "1.0"
MAX_EXACT_COUNT = 2**24  # float32 holds every integer of smaller magnitude exactly
_DATASET_ID_HEX = 10
_ROWS_PER_CHUNK = 1024
_ROWS_PER_SHARD = 65536
_PER_SPECTRUM = ("spectrum_id", "ix", "iy", "iz", "x_um", "y_um", "z_um", "qc_flags")


@dataclass(frozen=True)
class DatasetArrays:
    """The arrays of a dataset, or of a slice of its spectra (§4 layout).

    Per-spectrum arrays have ``n`` rows; ``intensity`` is ``[n, n_wn]``. ``wavenumber`` and
    ``grid_mask`` (``[nz, ny, nx]``) always describe the whole dataset.
    """

    intensity: npt.NDArray[np.float32]
    wavenumber: npt.NDArray[np.float64]
    spectrum_id: npt.NDArray[np.uint32]
    ix: npt.NDArray[np.int32]
    iy: npt.NDArray[np.int32]
    iz: npt.NDArray[np.int32]
    x_um: npt.NDArray[np.float64]
    y_um: npt.NDArray[np.float64]
    z_um: npt.NDArray[np.float64]
    source_file: tuple[str, ...]
    qc_flags: npt.NDArray[np.uint16]
    grid_mask: npt.NDArray[np.bool_]


def counts_as_float32(
    intensity: npt.NDArray[np.float64], source_files: tuple[str, ...]
) -> npt.NDArray[np.float32]:
    """Return ``intensity`` as float32, which stores every finite count exactly.

    Raises CountTooLargeError naming the first spectrum (row order, matching ``source_files``)
    with a finite count of magnitude ``MAX_EXACT_COUNT`` or more. NaN and infinities pass; QC
    flags them.
    """
    too_large = np.isfinite(intensity) & (np.abs(intensity) >= MAX_EXACT_COUNT)
    rows = np.flatnonzero(too_large.any(axis=1))
    if rows.size:
        row = int(rows[0])
        value = float(intensity[row][too_large[row]][0])
        raise CountTooLargeError(source_files[row], value, MAX_EXACT_COUNT)
    return intensity.astype(np.float32)


def content_hash(
    intensity: npt.NDArray[np.float32],
    wavenumber: npt.NDArray[np.float64],
    source_files: tuple[str, ...],
) -> str:
    """Return the BLAKE3 hex digest of the §4 byte sequence.

    That is ``intensity`` as little-endian float32 in C order, then ``wavenumber`` as
    little-endian float64, then the sorted ``source_files`` joined by newlines in UTF-8.
    """
    digest = blake3.blake3()
    digest.update(np.ascontiguousarray(intensity, dtype="<f4").tobytes())
    digest.update(np.ascontiguousarray(wavenumber, dtype="<f8").tobytes())
    digest.update("\n".join(sorted(source_files)).encode("utf-8"))
    return digest.hexdigest()


def make_dataset_id(scan: str, digest: str) -> str:
    """Return ``<scan>-<first 10 hex digits of digest>``."""
    return f"{scan}-{digest[:_DATASET_ID_HEX]}"


def write_dataset(path: Path, manifest: DatasetManifest, arrays: DatasetArrays) -> None:
    """Create the store at ``path`` from ``arrays``, with ``manifest`` in its attrs.

    ``path`` must not exist yet: callers write to a fresh path and rename it into place.
    """
    group = zarr.open_group(path, mode="w-", zarr_format=3)
    n_wn = arrays.wavenumber.shape[0]
    group.create_array(
        "intensity",
        data=arrays.intensity,
        chunks=(_ROWS_PER_CHUNK, n_wn),
        shards=(_ROWS_PER_SHARD, n_wn),
        compressors=[BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")],
    )
    group.create_array("wavenumber", data=arrays.wavenumber)
    for name in _PER_SPECTRUM:
        group.create_array(name, data=getattr(arrays, name), chunks=(_ROWS_PER_CHUNK,))
    group.create_array(
        "source_file",
        data=np.array(arrays.source_file, dtype=np.dtypes.StringDType()),
        chunks=(_ROWS_PER_CHUNK,),
    )
    group.create_array("grid_mask", data=arrays.grid_mask)
    group.attrs.update(
        {
            "schema_version": STORE_SCHEMA_VERSION,
            "manifest": manifest.model_dump(mode="json"),
            "content_hash": manifest.content_hash,
            "created_at": manifest.created_at.isoformat(),
        }
    )


@dataclass(frozen=True)
class StoredDataset:
    """An opened, checked store; ``read`` loads arrays on demand."""

    path: Path
    manifest: DatasetManifest
    _group: zarr.Group = field(repr=False, compare=False)

    def wavenumber(self) -> npt.NDArray[np.float64]:
        """Return the dataset axis alone, without reading any spectra."""
        axis: npt.NDArray[np.float64] = self._array("wavenumber")[:]
        return axis

    def read(self, start: int = 0, stop: int | None = None) -> DatasetArrays:
        """Return spectra ``start`` to ``stop`` (exclusive; default all) and the shared arrays."""
        rows = slice(start, stop)
        per_spectrum = {name: self._array(name)[rows] for name in _PER_SPECTRUM}
        return DatasetArrays(
            intensity=self._array("intensity")[rows],
            wavenumber=self.wavenumber(),
            source_file=tuple(str(name) for name in self._array("source_file")[rows]),
            grid_mask=self._array("grid_mask")[...],
            **per_spectrum,
        )

    def _array(self, name: str) -> Any:
        array = self._group.get(name)
        if not isinstance(array, zarr.Array):
            raise DatasetStoreError(self.path, f"no {name} array")
        return array


def open_dataset(path: Path) -> StoredDataset:
    """Open the store at ``path`` and check its attrs against its manifest.

    Raises DatasetStoreError when there is no store, the schema version is unknown, the manifest
    is invalid, or the ``content_hash`` or ``created_at`` attrs disagree with it.
    """
    if not (path / "zarr.json").is_file():
        raise DatasetStoreError(path, "no dataset store here")
    group = zarr.open_group(path, mode="r")
    attrs = group.attrs.asdict()
    if attrs.get("schema_version") != STORE_SCHEMA_VERSION:
        version = attrs.get("schema_version")
        raise DatasetStoreError(path, f"schema_version {version!r} is not {STORE_SCHEMA_VERSION}")
    try:
        manifest = DatasetManifest.model_validate(attrs.get("manifest"))
    except ValidationError as exc:
        raise DatasetStoreError(path, f"invalid manifest: {exc}") from exc
    if attrs.get("content_hash") != manifest.content_hash:
        raise DatasetStoreError(path, "content_hash attr disagrees with the manifest")
    if _instant(attrs.get("created_at")) != manifest.created_at:
        raise DatasetStoreError(path, "created_at attr disagrees with the manifest")
    return StoredDataset(path=path, manifest=manifest, _group=group)


def _instant(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None
