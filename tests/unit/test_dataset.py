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
# instrument.raw_stage (§3 text), n_spectra (§3 partial grids) and the §4 identity and
# provenance fields. Copied by hand; never generated from the code under test.
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


def test_scan_info_is_not_provisional() -> None:
    assert ScanInfo.__doc__ is not None
    assert "provisional" not in ScanInfo.__doc__.lower()


def test_raw_stage_is_verbatim_text() -> None:
    block = (
        "Axis : 1\nAxis is present\nAxis scan start : 28.51\n \n"
        "Axis : 2\nAxis is present\nAxis scan start : 26.96\n"
    )
    info = InstrumentInfo(system="s", detector="d", positioner="p", raw_stage=block)
    assert info.raw_stage == block


# ---------------------------------------------------------------- round-trip

finite = st.floats(allow_nan=False, allow_infinity=False)
coordinate = st.floats(min_value=-1e4, max_value=1e4, allow_nan=False)
step = st.floats(min_value=1e-3, max_value=10.0) | st.floats(min_value=-10.0, max_value=-1e-3)
naive_dt = st.datetimes(min_value=datetime(1990, 1, 1), max_value=datetime(2100, 1, 1))
utc_dt = st.datetimes(
    min_value=datetime(1990, 1, 1), max_value=datetime(2100, 1, 1), timezones=st.just(UTC)
)
counts = st.integers(min_value=1, max_value=4096)


@st.composite
def axes(draw: st.DrawFn, n: int) -> AxisSpec:
    """Draw an axis consistent with ``n`` points (§3): one point, or ends within half a step."""
    start = draw(coordinate)
    if n == 1:
        return AxisSpec(start_um=start, stop_um=start, step_um=0.0)
    step_um = draw(step)
    stop = start + (n - 1) * step_um + draw(st.floats(-0.49, 0.49)) * abs(step_um)
    return AxisSpec(start_um=start, stop_um=stop, step_um=step_um)


@st.composite
def grids(draw: st.DrawFn) -> GridSpec:
    nx, ny, nz = draw(counts), draw(counts), draw(st.integers(min_value=1, max_value=8))
    return GridSpec(
        nx=nx,
        ny=ny,
        nz=nz,
        x=draw(axes(nx)),
        y=draw(axes(ny)),
        z=draw(axes(nz)),
        allow_partial=draw(st.booleans()),
    )


instruments = st.builds(
    InstrumentInfo, system=st.text(), detector=st.text(), positioner=st.text(), raw_stage=st.text()
)
acqs = st.builds(
    AcquisitionInfo,
    exposure_s=st.floats(min_value=1e-6, max_value=1e4),
    n_accum=counts,
    ccd_temp_c=finite,
    central_pixel=st.integers(min_value=0, max_value=4096),
    head_model=st.text(),
)


@st.composite
def manifests(draw: st.DrawFn) -> DatasetManifest:
    grid = draw(grids())
    size = grid.nx * grid.ny * grid.nz
    n_spectra = draw(st.integers(1, size)) if grid.allow_partial else size
    return DatasetManifest(
        scan=draw(st.text(min_size=1)),
        acquired_at=draw(naive_dt),
        duration_s=draw(st.floats(min_value=0, max_value=1e6)),
        instrument=draw(instruments),
        acq=draw(acqs),
        n_wn=draw(counts),
        n_series=1,
        grid=grid,
        dataset_id=draw(st.text(min_size=1)),
        sample=draw(st.none() | st.text()),
        n_spectra=n_spectra,
        content_hash=draw(st.text(alphabet="0123456789abcdef", min_size=1)),
        created_at=draw(utc_dt),
    )


@given(manifests())
def test_manifest_json_round_trip(manifest: DatasetManifest) -> None:
    assert DatasetManifest.model_validate_json(manifest.model_dump_json()) == manifest


@given(manifests())
def test_manifest_python_round_trip(manifest: DatasetManifest) -> None:
    assert DatasetManifest.model_validate(manifest.model_dump()) == manifest


def test_manifest_rejects_unknown_nested_key() -> None:
    axis = {"start_um": 0.0, "stop_um": 1.0, "step_um": 1.0}
    grid: dict[str, Any] = {"nx": 2, "ny": 2, "nz": 2, "x": axis, "y": axis, "z": axis}
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


# ---------------------------------------------------------------- §3 constraints


def _grid_fields(**overrides: Any) -> dict[str, Any]:
    point = {"start_um": 42.6788, "stop_um": 42.6788, "step_um": 0.0}
    x = {"start_um": 28.5139, "stop_um": 77.5378, "step_um": 1.0005}
    y = {"start_um": 26.9586, "stop_um": 50.9488, "step_um": 0.9996}
    return {"nx": 50, "ny": 25, "nz": 1, "x": x, "y": y, "z": point} | overrides


def _error_types(exc: pytest.ExceptionInfo[ValidationError]) -> list[tuple[Any, str]]:
    return [(error["loc"], error["type"]) for error in exc.value.errors()]


@pytest.mark.parametrize("count", ["nx", "ny", "nz"])
@pytest.mark.parametrize("value", [0, -1])
def test_grid_counts_must_be_positive(count: str, value: int) -> None:
    GridSpec.model_validate(_grid_fields())
    with pytest.raises(ValidationError) as exc:
        GridSpec.model_validate(_grid_fields(**{count: value}))
    assert _error_types(exc) == [((count,), "greater_than_equal")]


@pytest.mark.parametrize(
    "axis",
    [
        {"start_um": 1.0, "stop_um": 1.0, "step_um": 0.5},
        {"start_um": 1.0, "stop_um": 2.0, "step_um": 0.0},
        {"start_um": float("nan"), "stop_um": float("nan"), "step_um": 0.0},
    ],
)
def test_single_point_axis_needs_zero_step_and_equal_ends(axis: dict[str, float]) -> None:
    with pytest.raises(ValidationError, match="axis z") as exc:
        GridSpec.model_validate(_grid_fields(z=axis))
    assert _error_types(exc) == [((), "single_point_axis")]


def test_multi_point_axis_stop_within_half_step() -> None:
    def x_axis(stop: float, step_um: float = 1.0) -> dict[str, Any]:
        return _grid_fields(nx=11, x={"start_um": 0.0, "stop_um": stop, "step_um": step_um})

    for stop in (10.0, 9.5, 10.5, 10.0001):
        GridSpec.model_validate(x_axis(stop))
    GridSpec.model_validate(x_axis(-10.0, -1.0))
    for stop, step_um in ((10.51, 1.0), (9.49, 1.0), (10.0, -1.0), (float("nan"), 1.0)):
        with pytest.raises(ValidationError, match="axis x") as exc:
            GridSpec.model_validate(x_axis(stop, step_um))
        assert _error_types(exc) == [((), "axis_stop_mismatch")]
    with pytest.raises(ValidationError, match="axis x") as exc:
        GridSpec.model_validate(x_axis(0.0, 0.0))
    assert _error_types(exc) == [((), "multi_point_axis_zero_step")]


def _acq_fields(**overrides: Any) -> dict[str, Any]:
    fields = {
        "exposure_s": 0.2,
        "n_accum": 5,
        "ccd_temp_c": -72.0,
        "central_pixel": 505,
        "head_model": "DU920P_BEX2D",
    }
    return fields | overrides


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("exposure_s", 0.0, "greater_than"),
        ("exposure_s", -0.2, "greater_than"),
        ("n_accum", 0, "greater_than_equal"),
        ("central_pixel", -1, "greater_than_equal"),
    ],
)
def test_acquisition_constraints(field: str, value: float, error: str) -> None:
    AcquisitionInfo.model_validate(_acq_fields(central_pixel=0, n_accum=1))
    with pytest.raises(ValidationError) as exc:
        AcquisitionInfo.model_validate(_acq_fields(**{field: value}))
    assert _error_types(exc) == [((field,), error)]


@pytest.mark.parametrize("n_series", [0, 2, 10])
def test_n_series_other_than_1_rejected_by_model(n_series: int) -> None:
    fields = {k: v for k, v in _manifest_fields().items() if k in SCAN_INFO_KEYS}
    ScanInfo.model_validate(fields)
    with pytest.raises(ValidationError, match="time series") as exc:
        ScanInfo.model_validate(fields | {"n_series": n_series})
    assert _error_types(exc) == [(("n_series",), "time_series_not_supported")]


def test_duration_must_be_non_negative() -> None:
    DatasetManifest.model_validate(_manifest_fields(duration_s=0.0))
    with pytest.raises(ValidationError) as exc:
        DatasetManifest.model_validate(_manifest_fields(duration_s=-1.0))
    assert _error_types(exc) == [(("duration_s",), "greater_than_equal")]


def test_scan_n_wn_and_n_spectra_lower_bounds() -> None:
    for field, value in (("scan", ""), ("n_wn", 0), ("n_spectra", 0)):
        with pytest.raises(ValidationError) as exc:
            DatasetManifest.model_validate(_manifest_fields(**{field: value}))
        assert [loc for loc, _ in _error_types(exc)] == [(field,)]


def test_n_spectra_must_equal_grid_unless_partial() -> None:
    grid = _grid_fields()
    full = 50 * 25
    DatasetManifest.model_validate(_manifest_fields(grid=grid, n_spectra=full))
    for n_spectra in (full - 1, full + 1):
        with pytest.raises(ValidationError, match="1250") as exc:
            DatasetManifest.model_validate(_manifest_fields(grid=grid, n_spectra=n_spectra))
        assert _error_types(exc) == [((), "n_spectra_grid_mismatch")]
    partial = grid | {"allow_partial": True}
    for n_spectra in (1, 25, full):
        DatasetManifest.model_validate(_manifest_fields(grid=partial, n_spectra=n_spectra))
    with pytest.raises(ValidationError) as exc:
        DatasetManifest.model_validate(_manifest_fields(grid=partial, n_spectra=full + 1))
    assert _error_types(exc) == [((), "n_spectra_exceeds_grid")]
