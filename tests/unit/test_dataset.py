"""Dataset models (DESIGN.md §3, §4): QC bits, filename grammar, manifest structure."""

import re
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import BaseModel, ValidationError

from raman_bayes.domain import dataset
from raman_bayes.domain._base import DomainModel
from raman_bayes.domain.dataset import (
    FILENAME_RE,
    AcquisitionInfo,
    AxisSpec,
    DatasetManifest,
    FilenamePattern,
    GridSpec,
    InstrumentInfo,
    QCFlag,
    ScanInfo,
)

# Dotted manifest keys: the §3 table, plus grid.allow_partial, filename_pattern.regex and
# instrument.raw_stage (§3 text), n_spectra (§3 partial grids) and the provisional
# identity/provenance fields. Copied by hand; never generated from the code under test.
MANIFEST_KEYS = {
    "scan",
    "acquired_at",
    "duration_s",
    "instrument.system",
    "instrument.detector",
    "instrument.positioner",
    "instrument.raw_stage",
    "n_wn",
    "n_series",
    "grid.nx",
    "grid.ny",
    "grid.nz",
    *(f"grid.{axis}.{key}" for axis in "xyz" for key in ("start_um", "stop_um", "step_um")),
    "grid.allow_partial",
    "acq.exposure_s",
    "acq.n_accum",
    "acq.ccd_temp_c",
    "acq.central_pixel",
    "acq.head_model",
    "schema_version",
    "dataset_id",
    "sample",
    "n_spectra",
    "filename_pattern.regex",
    "content_hash",
    "created_at",
}

SCAN_INFO_KEYS = {
    "scan",
    "acquired_at",
    "duration_s",
    "instrument",
    "acq",
    "n_wn",
    "n_series",
    "grid",
}


def _dotted_keys(model: type[BaseModel], prefix: str = "") -> set[str]:
    keys: set[str] = set()
    for name, field in model.model_fields.items():
        annotation = field.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            keys |= _dotted_keys(annotation, f"{prefix}{name}.")
        else:
            keys.add(f"{prefix}{name}")
    return keys


# ---------------------------------------------------------------- QC flags


def test_qc_flag_bits_match_section3() -> None:
    assert {flag.name: flag.value for flag in QCFlag} == {
        "NAN_OR_INF": 1 << 0,
        "SATURATED": 1 << 1,
        "AXIS_MISMATCH": 1 << 2,
        "NEGATIVE": 1 << 3,
    }


def test_qc_flags_combine_and_fit_uint16() -> None:
    combined = QCFlag.NAN_OR_INF | QCFlag.SATURATED | QCFlag.AXIS_MISMATCH | QCFlag.NEGATIVE
    assert int(np.uint16(combined)) == int(combined) == 0b1111
    assert QCFlag(int(np.uint16(QCFlag.SATURATED | QCFlag.NEGATIVE))) & QCFlag.SATURATED


# ---------------------------------------------------------------- filename grammar


def test_filename_re_is_verbatim_from_section3() -> None:
    assert FILENAME_RE == r"^(?P<scan>.+?)-z(?P<iz>\d+)y(?P<iy>\d+)x(?P<ix>\d+)\.txt$"


@pytest.mark.parametrize(
    ("name", "scan", "iz", "iy", "ix"),
    [
        ("NMC811_v0-z0y24x49.txt", "NMC811_v0", 0, 24, 49),
        ("NMC111_v2-z0y0x0.txt", "NMC111_v2", 0, 0, 0),
        ("sample-a-b-z3y10x7.txt", "sample-a-b", 3, 10, 7),
        ("odd-z9-z1y2x3.txt", "odd-z9", 1, 2, 3),
    ],
)
def test_filename_re_matches(name: str, scan: str, iz: int, iy: int, ix: int) -> None:
    match = re.match(FILENAME_RE, name)
    assert match is not None
    assert match["scan"] == scan
    assert (int(match["iz"]), int(match["iy"]), int(match["ix"])) == (iz, iy, ix)


@pytest.mark.parametrize(
    "name",
    [
        "NMC811_v0-Info.txt",
        "NMC811_v0-z0y0x0.TXT",
        "NMC811_v0-z0y0.txt",
        "NMC811_v0-z0x0y0.txt",
        "NMC811_v0-z0y0x0.txt.bak",
        "-z0y0x0.txt",
        "NMC811_v0-z-1y0x0.txt",
    ],
)
def test_filename_re_rejects(name: str) -> None:
    assert re.match(FILENAME_RE, name) is None


# ---------------------------------------------------------------- model structure


def test_manifest_keys_match_section3() -> None:
    assert _dotted_keys(DatasetManifest) == MANIFEST_KEYS


def test_scan_info_keys() -> None:
    assert set(ScanInfo.model_fields) == SCAN_INFO_KEYS


def test_manifest_extends_scan_info() -> None:
    assert issubclass(DatasetManifest, ScanInfo)


def test_defaults() -> None:
    assert FilenamePattern().regex == FILENAME_RE
    assert GridSpec.model_fields["allow_partial"].default is False
    assert DatasetManifest.model_fields["schema_version"].default == "1.0"
    assert DatasetManifest.model_fields["filename_pattern"].default == FilenamePattern()


def test_every_model_is_frozen_and_forbids_extra() -> None:
    models = {
        obj
        for obj in vars(dataset).values()
        if isinstance(obj, type) and issubclass(obj, DomainModel) and obj is not DomainModel
    }
    assert models == {
        AxisSpec,
        GridSpec,
        InstrumentInfo,
        AcquisitionInfo,
        FilenamePattern,
        ScanInfo,
        DatasetManifest,
    }
    for model in models:
        assert model.model_config.get("frozen") is True, model.__name__
        assert model.model_config.get("extra") == "forbid", model.__name__


def test_scan_info_is_marked_provisional() -> None:
    assert ScanInfo.__doc__ is not None
    assert "provisional" in ScanInfo.__doc__.lower()


def test_raw_stage_is_verbatim_text() -> None:
    block = (
        "Axis : 1\nAxis is present\nAxis scan start : 28.51\n \n"
        "Axis : 2\nAxis is present\nAxis scan start : 26.96\n"
    )
    info = InstrumentInfo(system="s", detector="d", positioner="p", raw_stage=block)
    assert info.raw_stage == block


# ---------------------------------------------------------------- round-trip

finite = st.floats(allow_nan=False, allow_infinity=False)
naive_dt = st.datetimes(min_value=datetime(1990, 1, 1), max_value=datetime(2100, 1, 1))
utc_dt = st.datetimes(
    min_value=datetime(1990, 1, 1), max_value=datetime(2100, 1, 1), timezones=st.just(UTC)
)
counts = st.integers(min_value=1, max_value=4096)
axes = st.builds(AxisSpec, start_um=finite, stop_um=finite, step_um=finite)
grids = st.builds(
    GridSpec, nx=counts, ny=counts, nz=counts, x=axes, y=axes, z=axes, allow_partial=st.booleans()
)
instruments = st.builds(
    InstrumentInfo, system=st.text(), detector=st.text(), positioner=st.text(), raw_stage=st.text()
)
acqs = st.builds(
    AcquisitionInfo,
    exposure_s=finite,
    n_accum=counts,
    ccd_temp_c=finite,
    central_pixel=st.integers(min_value=0, max_value=4096),
    head_model=st.text(),
)
manifests = st.builds(
    DatasetManifest,
    scan=st.text(min_size=1),
    acquired_at=naive_dt,
    duration_s=finite,
    instrument=instruments,
    acq=acqs,
    n_wn=counts,
    n_series=st.just(1),
    grid=grids,
    dataset_id=st.text(min_size=1),
    sample=st.none() | st.text(),
    n_spectra=counts,
    content_hash=st.text(alphabet="0123456789abcdef", min_size=1),
    created_at=utc_dt,
)


@given(manifests)
def test_manifest_json_round_trip(manifest: DatasetManifest) -> None:
    assert DatasetManifest.model_validate_json(manifest.model_dump_json()) == manifest


@given(manifests)
def test_manifest_python_round_trip(manifest: DatasetManifest) -> None:
    assert DatasetManifest.model_validate(manifest.model_dump()) == manifest


def test_manifest_rejects_unknown_nested_key() -> None:
    axis = {"start_um": 0.0, "stop_um": 1.0, "step_um": 1.0}
    grid: dict[str, Any] = {"nx": 2, "ny": 1, "nz": 1, "x": axis, "y": axis, "z": axis}
    GridSpec.model_validate(grid)
    with pytest.raises(ValidationError) as exc:
        GridSpec.model_validate({**grid, "x": {**axis, "points": 2}})
    assert exc.value.errors()[0]["loc"] == ("x", "points")


def _manifest_fields(**overrides: Any) -> dict[str, Any]:
    axis = {"start_um": 0.0, "stop_um": 0.0, "step_um": 0.0}
    fields: dict[str, Any] = {
        "scan": "NMC811_v0",
        "acquired_at": datetime(2026, 5, 28, 15, 47, 23),
        "duration_s": 2090.0,
        "instrument": {"system": "s", "detector": "d", "positioner": "p", "raw_stage": ""},
        "acq": {
            "exposure_s": 0.2,
            "n_accum": 5,
            "ccd_temp_c": -72.0,
            "central_pixel": 505,
            "head_model": "DU920P_BEX2D",
        },
        "n_wn": 1024,
        "n_series": 1,
        "grid": {"nx": 1, "ny": 1, "nz": 1, "x": axis, "y": axis, "z": axis},
        "dataset_id": "NMC811_v0-0123456789",
        "sample": None,
        "n_spectra": 1,
        "content_hash": "0123456789",
        "created_at": datetime(2026, 10, 23, 12, 0, tzinfo=UTC),
    }
    return fields | overrides


def test_created_at_rejects_naive_datetime() -> None:
    with pytest.raises(ValidationError) as exc:
        DatasetManifest.model_validate(_manifest_fields(created_at=datetime(2026, 10, 23, 12, 0)))
    (error,) = exc.value.errors()
    assert error["loc"] == ("created_at",)
    assert error["type"] == "timezone_aware"


def test_created_at_rejects_naive_iso_string() -> None:
    with pytest.raises(ValidationError):
        DatasetManifest.model_validate(_manifest_fields(created_at="2026-10-23T12:00:00"))


def test_created_at_keeps_utc_through_json() -> None:
    manifest = DatasetManifest.model_validate(_manifest_fields())
    assert manifest.model_dump(mode="json")["created_at"] == "2026-10-23T12:00:00Z"
    restored = DatasetManifest.model_validate_json(manifest.model_dump_json())
    assert restored.created_at.utcoffset() == timedelta(0)


def test_created_at_accepts_other_offsets_as_same_instant() -> None:
    pacific = datetime(2026, 10, 23, 5, 0, tzinfo=timezone(timedelta(hours=-7)))
    manifest = DatasetManifest.model_validate(_manifest_fields(created_at=pacific))
    assert manifest.created_at == datetime(2026, 10, 23, 12, 0, tzinfo=UTC)


def test_acquired_at_stays_naive() -> None:
    manifest = DatasetManifest.model_validate(_manifest_fields())
    assert manifest.acquired_at.tzinfo is None
    restored = DatasetManifest.model_validate_json(manifest.model_dump_json())
    assert restored.acquired_at == datetime(2026, 5, 28, 15, 47, 23)
    assert restored.acquired_at.tzinfo is None
