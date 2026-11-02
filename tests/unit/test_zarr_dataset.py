"""The ``dataset.zarr`` store (DESIGN.md §4): layout, round trip, attrs, content hash and id."""

import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import blake3
import numpy as np
import pytest
import zarr
from zarr.codecs import BloscCodec

from raman_bayes.app.ingest import IngestRequest, ScanStatus, ingest
from raman_bayes.config.settings import PathsSettings
from raman_bayes.domain.dataset import DatasetManifest
from raman_bayes.domain.errors import CountTooLargeError, DatasetStoreError
from raman_bayes.io.nanofinder import parse_info
from raman_bayes.storage.zarr_dataset import (
    DatasetArrays,
    content_hash,
    counts_as_float32,
    make_dataset_id,
    open_dataset,
    write_dataset,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
INFO = parse_info(FIXTURES / "info_txt" / "NMC811_v0-Info.txt")
NOW = datetime(2026, 11, 2, 12, 0, tzinfo=UTC)
N_WN = 32
# Copied by hand from DESIGN.md §4: name -> (dtype, shape given n_spectra and n_wn).
LAYOUT = {
    "intensity": ("float32", ("n", "wn")),
    "wavenumber": ("float64", ("wn",)),
    "spectrum_id": ("uint32", ("n",)),
    "ix": ("int32", ("n",)),
    "iy": ("int32", ("n",)),
    "iz": ("int32", ("n",)),
    "x_um": ("float64", ("n",)),
    "y_um": ("float64", ("n",)),
    "z_um": ("float64", ("n",)),
    "source_file": ("str", ("n",)),
    "qc_flags": ("uint16", ("n",)),
    "grid_mask": ("bool", ("nz", "ny", "nx")),
}


def _arrays(n: int = 6) -> DatasetArrays:
    rng = np.random.default_rng(0)
    intensity = rng.integers(0, 2**20, size=(n, N_WN)).astype(np.float32)
    intensity[1, 3] = np.nan  # bit-identical must hold for NaN too
    mask = np.zeros((1, 25, 50), dtype=bool)
    mask[0, 0, :n] = True
    ix = np.arange(n, dtype=np.int32)
    zeros = np.zeros(n, dtype=np.int32)
    return DatasetArrays(
        intensity=intensity,
        wavenumber=np.linspace(-41.55539, 1945.8536, N_WN),
        spectrum_id=np.arange(n, dtype=np.uint32),
        ix=ix,
        iy=zeros,
        iz=zeros.copy(),
        x_um=(28.5139 + ix * 1.0005).astype(np.float64),
        y_um=np.full(n, 26.9586),
        z_um=np.full(n, 42.6788),
        source_file=tuple(f"NMC811_v0-z0y0x{i}.txt" for i in range(n)),
        qc_flags=np.array([0, 1, 0, 2, 8, 0][:n], dtype=np.uint16),
        grid_mask=mask,
    )


def _manifest(arrays: DatasetArrays) -> DatasetManifest:
    digest = content_hash(arrays.intensity, arrays.wavenumber, arrays.source_file)
    fields = INFO.model_dump()
    fields["grid"]["allow_partial"] = True
    fields |= {
        "n_wn": N_WN,
        "dataset_id": make_dataset_id(INFO.scan, digest),
        "sample": "cathode A",
        "n_spectra": len(arrays.source_file),
        "content_hash": digest,
        "created_at": NOW,
    }
    return DatasetManifest.model_validate(fields)


@pytest.fixture
def store(tmp_path: Path) -> Path:
    arrays = _arrays()
    path = tmp_path / "d.zarr"
    write_dataset(path, _manifest(arrays), arrays)
    return path


def _ingested(scan: Path, paths: PathsSettings) -> Path:
    (outcome,) = ingest(IngestRequest(path=scan, allow_partial_grid=True), paths, now=NOW)
    assert outcome.status is ScanStatus.INGESTED, outcome.errors
    assert outcome.store_uri is not None
    return outcome.store_uri


# ---------------------------------------------------------------- layout and round trip


def test_round_trip_bit_identical(store: Path) -> None:
    arrays = _arrays()
    stored = open_dataset(store)
    assert stored.manifest == _manifest(arrays)
    back = stored.read()
    assert back.source_file == arrays.source_file
    for name in LAYOUT:
        if name == "source_file":
            continue
        original, read = getattr(arrays, name), getattr(back, name)
        assert read.dtype == original.dtype, name
        assert read.shape == original.shape, name
        assert read.tobytes() == original.tobytes(), name


def test_read_slices_per_spectrum_arrays(store: Path) -> None:
    arrays = _arrays()
    part = open_dataset(store).read(2, 5)
    np.testing.assert_array_equal(part.spectrum_id, [2, 3, 4])
    np.testing.assert_array_equal(part.intensity, arrays.intensity[2:5])
    assert part.source_file == arrays.source_file[2:5]
    np.testing.assert_array_equal(part.wavenumber, arrays.wavenumber)
    np.testing.assert_array_equal(part.grid_mask, arrays.grid_mask)


def test_arrays_dtypes_shapes_chunks_shards_codecs(store: Path) -> None:
    group = zarr.open_group(store, mode="r")
    assert group.metadata.zarr_format == 3
    assert set(group.array_keys()) == set(LAYOUT)
    sizes = {"n": 6, "wn": N_WN, "nz": 1, "ny": 25, "nx": 50}
    for name, (dtype, dims) in LAYOUT.items():
        array = group[name]
        assert isinstance(array, zarr.Array)
        assert array.shape == tuple(sizes[d] for d in dims), name
        if dtype == "str":
            assert isinstance(array.dtype, np.dtypes.StringDType), name
        else:
            assert array.dtype == np.dtype(dtype), name
    intensity = group["intensity"]
    assert isinstance(intensity, zarr.Array)
    assert intensity.chunks == (1024, N_WN)
    assert intensity.shards == (65536, N_WN)
    (codec,) = intensity.compressors
    assert isinstance(codec, BloscCodec)
    assert (codec.cname, codec.clevel, codec.shuffle) == ("zstd", 3, "shuffle")


def test_spectrum_id_is_raster_order(mini_scan: Path, raman_paths: PathsSettings) -> None:
    stored = open_dataset(_ingested(mini_scan, raman_paths)).read()
    np.testing.assert_array_equal(stored.spectrum_id, np.arange(25, dtype=np.uint32))
    order = np.lexsort((stored.ix, stored.iy, stored.iz))
    np.testing.assert_array_equal(order, np.arange(25))
    expected = [f"NMC811_v0-z0y{iy}x{ix}.txt" for iy in range(5) for ix in range(5)]
    assert list(stored.source_file) == expected


def test_coordinates_from_frame_dimentions(mini_scan: Path, raman_paths: PathsSettings) -> None:
    stored = open_dataset(_ingested(mini_scan, raman_paths)).read()
    np.testing.assert_allclose(stored.x_um, 28.5139 + stored.ix * 1.0005, rtol=0, atol=1e-12)
    np.testing.assert_allclose(stored.y_um, 26.9586 + stored.iy * 0.9996, rtol=0, atol=1e-12)
    # Z has one point: Frame Dimentions' position, not Frame Stage's 75.06 um.
    np.testing.assert_array_equal(stored.z_um, np.full(25, 42.6788))


def test_attrs_repeat_manifest_values(store: Path) -> None:
    attrs = dict(zarr.open_group(store, mode="r").attrs)
    assert set(attrs) == {"schema_version", "manifest", "content_hash", "created_at"}
    assert attrs["schema_version"] == "1.0"
    manifest = DatasetManifest.model_validate(attrs["manifest"])
    assert manifest == _manifest(_arrays())
    assert attrs["content_hash"] == manifest.content_hash
    assert isinstance(attrs["created_at"], str)
    assert datetime.fromisoformat(attrs["created_at"]) == manifest.created_at


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("content_hash", "0" * 64),
        ("created_at", "2027-01-01T00:00:00+00:00"),
        ("schema_version", "9.9"),
    ],
)
def test_reader_rejects_attrs_disagreeing_with_manifest(store: Path, key: str, value: str) -> None:
    zarr.open_group(store, mode="r+").attrs.update({key: value})
    with pytest.raises(DatasetStoreError, match=key):
        open_dataset(store)


def test_open_missing_store_is_a_store_error(tmp_path: Path) -> None:
    with pytest.raises(DatasetStoreError, match="no dataset store"):
        open_dataset(tmp_path / "absent.zarr")


# ---------------------------------------------------------------- counts and identity


def test_count_at_or_above_2_pow_24_refused() -> None:
    below = np.array([[0.0, 2.0**24 - 1, -(2.0**24 - 1), np.inf, np.nan]])
    packed = counts_as_float32(below, ("a.txt",))
    assert packed.dtype == np.float32
    np.testing.assert_array_equal(packed[0, :3], below[0, :3])
    for bad in (2.0**24, -(2.0**24), 2.0**30):
        values = np.array([[1.0, 2.0], [3.0, bad]])
        with pytest.raises(CountTooLargeError) as exc:
            counts_as_float32(values, ("a.txt", "b.txt"))
        assert "b.txt" in str(exc.value)
        assert "a.txt" not in str(exc.value)


def test_content_hash_matches_definition() -> None:
    arrays = _arrays()
    names = ("b.txt", "a.txt", "c.txt", "d.txt", "e.txt", "f.txt")
    expected = blake3.blake3()
    expected.update(arrays.intensity.astype("<f4").tobytes(order="C"))
    expected.update(arrays.wavenumber.astype("<f8").tobytes(order="C"))
    expected.update("\n".join(sorted(names)).encode("utf-8"))
    digest = content_hash(arrays.intensity, arrays.wavenumber, names)
    assert digest == expected.hexdigest()
    assert len(digest) == 64
    assert make_dataset_id("NMC811_v0", digest) == f"NMC811_v0-{digest[:10]}"


def test_dataset_id_deterministic_and_location_independent(
    mini_scan: Path, raman_paths: PathsSettings, tmp_path: Path
) -> None:
    elsewhere = Path(shutil.copytree(mini_scan, tmp_path / "moved" / "deeper" / mini_scan.name))
    other_state = PathsSettings(
        data_root=tmp_path / "other" / "datasets",
        runs_root=tmp_path / "other" / "runs",
        catalog=tmp_path / "other" / "catalog.sqlite",
    )
    first = open_dataset(_ingested(mini_scan, raman_paths)).manifest
    second = open_dataset(_ingested(elsewhere, other_state)).manifest
    assert first.dataset_id == second.dataset_id
    assert first.content_hash == second.content_hash
    assert first.dataset_id == f"NMC811_v0-{first.content_hash[:10]}"


def test_dataset_id_changes_with_one_count(
    mini_scan: Path,
    raman_paths: PathsSettings,
    tmp_path: Path,
    replace_once: Callable[[Path, bytes, bytes], None],
) -> None:
    changed = Path(shutil.copytree(mini_scan, tmp_path / "changed" / mini_scan.name))
    spectrum = changed / "NMC811_v0-z0y3x2.txt"
    first_line = spectrum.read_bytes().split(b"\r\n", 1)[0]
    shift, count = first_line.split(b"\t")
    bumped = shift + b"\t" + f"{float(count) + 1:.2f}".encode()
    replace_once(spectrum, first_line + b"\r\n", bumped + b"\r\n")
    original = open_dataset(_ingested(mini_scan, raman_paths)).manifest
    other_state = PathsSettings(
        data_root=tmp_path / "other" / "datasets",
        runs_root=tmp_path / "other" / "runs",
        catalog=tmp_path / "other" / "catalog.sqlite",
    )
    edited = open_dataset(_ingested(changed, other_state)).manifest
    assert edited.dataset_id != original.dataset_id
    assert edited.content_hash != original.content_hash
