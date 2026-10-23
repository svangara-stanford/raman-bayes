"""Result dataclasses (DESIGN.md §6) and explicit Parquet schemas (§8.2)."""

import dataclasses
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import TypeAdapter, ValidationError

from raman_bayes.domain.results import (
    FITS_SCHEMA,
    PEAKS_SCHEMA,
    PeakPosterior,
    PosteriorSummary,
    WindowFitResult,
)

DICT_STR = pa.dictionary(pa.int32(), pa.string())
F32 = pa.float32()
QUANTITIES = ("center", "fwhm", "area", "height", "eta")
STATS = ("mean", "median", "sd", "q025", "q975")

# (name, type, nullable) in §8.2 order. Copied by hand; never generated from the code under test.
PEAKS_COLUMNS: list[tuple[str, pa.DataType, bool]] = [
    ("run_id", DICT_STR, False),
    ("dataset_id", DICT_STR, False),
    ("spectrum_id", pa.uint32(), False),
    ("ix", pa.int32(), False),
    ("iy", pa.int32(), False),
    ("iz", pa.int32(), False),
    ("x_um", pa.float64(), False),
    ("y_um", pa.float64(), False),
    ("z_um", pa.float64(), False),
    ("window_id", DICT_STR, False),
    ("end_member_id", DICT_STR, True),
    ("peak_id", DICT_STR, False),
    ("presence_prob", F32, True),
    *((f"{q}_{s}", F32, True) for q in QUANTITIES for s in STATS),
    ("center_at_bound_frac", F32, True),
    ("fwhm_at_bound_frac", F32, True),
    ("fit_status", DICT_STR, False),
]
# Fixed columns only; merge (Phase 1) appends the engine-defined diag_* columns.
FITS_COLUMNS: list[tuple[str, pa.DataType, bool]] = [
    ("run_id", DICT_STR, False),
    ("spectrum_id", pa.uint32(), False),
    ("window_id", DICT_STR, False),
    ("status", DICT_STR, False),
    ("runtime_s", pa.float64(), False),
    ("noise_sd", F32, True),
    ("engine", DICT_STR, False),
    ("engine_version", DICT_STR, False),
    ("error", pa.string(), True),
]

DATACLASS_FIELDS: list[tuple[type, list[str]]] = [
    (PosteriorSummary, ["mean", "median", "sd", "q025", "q975", "at_bound_frac"]),
    (PeakPosterior, ["peak_id", "presence_prob", "center", "fwhm", "area", "height", "eta"]),
    (
        WindowFitResult,
        [
            "spectrum_id",
            "window_id",
            "status",
            "peaks",
            "diagnostics",
            "runtime_s",
            "engine",
            "engine_version",
            "error",
        ],
    ),
]


# ---------------------------------------------------------------- schemas


@pytest.mark.parametrize(
    ("schema", "columns"),
    [(PEAKS_SCHEMA, PEAKS_COLUMNS), (FITS_SCHEMA, FITS_COLUMNS)],
    ids=["peaks", "fits"],
)
def test_schema_columns(schema: pa.Schema, columns: list[tuple[str, pa.DataType, bool]]) -> None:
    assert [(f.name, f.type, f.nullable) for f in schema] == columns


def test_peaks_schema_has_41_columns() -> None:
    assert len(PEAKS_SCHEMA) == 41


@pytest.mark.parametrize("schema", [PEAKS_SCHEMA, FITS_SCHEMA], ids=["peaks", "fits"])
def test_schema_carries_no_metadata(schema: pa.Schema) -> None:
    # run_manifest.json and schema_version are attached per file at write time (§8.2).
    assert schema.metadata is None


def _absent_peak_row(spectrum_id: int) -> dict[str, Any]:
    row: dict[str, Any] = dict.fromkeys(PEAKS_SCHEMA.names)
    row |= {
        "run_id": "0190b3d4-0000-7000-8000-000000000000",
        "dataset_id": "NMC811_v0-0123456789",
        "spectrum_id": spectrum_id,
        "ix": spectrum_id,
        "iy": 0,
        "iz": 0,
        "x_um": 0.5 * spectrum_id,
        "y_um": 0.0,
        "z_um": 42.68,
        "window_id": "lithiation",
        "peak_id": "u1",
        "presence_prob": 0.1,
        "fit_status": "ok",
    }
    return row


def test_all_null_column_keeps_float32_through_parquet(tmp_path: Path) -> None:
    rows = [_absent_peak_row(i) for i in range(3)]
    table = pa.Table.from_pylist(rows, schema=PEAKS_SCHEMA)
    assert table.schema.field("center_mean").type == pa.float32()
    assert table.column("center_mean").null_count == 3

    path = tmp_path / "part-000.parquet"
    pq.write_table(table, path)
    back = pq.read_table(path)
    assert back.schema.field("center_mean").type == pa.float32()
    assert back.schema.remove_metadata() == PEAKS_SCHEMA


def test_without_schema_pyarrow_infers_null_type() -> None:
    # Control: this is the failure the explicit schema prevents (§8.2).
    table = pa.Table.from_pylist([_absent_peak_row(i) for i in range(3)])
    assert table.schema.field("center_mean").type == pa.null()


# ---------------------------------------------------------------- dataclasses


@pytest.mark.parametrize(
    ("cls", "names"), DATACLASS_FIELDS, ids=lambda v: getattr(v, "__name__", "")
)
def test_dataclass_fields_match_design(cls: type, names: list[str]) -> None:
    assert dataclasses.is_dataclass(cls)
    assert [f.name for f in dataclasses.fields(cls)] == names
    assert cls.__dataclass_params__.frozen  # type: ignore[attr-defined]


def _summary(value: float = 465.0) -> PosteriorSummary:
    return PosteriorSummary(mean=value, median=value, sd=1.0, q025=value - 2, q975=value + 2)


def test_defaults() -> None:
    assert _summary().at_bound_frac is None
    result = WindowFitResult(
        spectrum_id=0,
        window_id="lithiation",
        status="ok",
        peaks=(),
        diagnostics={},
        runtime_s=0.0,
        engine="mock",
        engine_version="0.1.0",
    )
    assert result.error is None


def test_dataclasses_are_frozen() -> None:
    summary = _summary()
    with pytest.raises(dataclasses.FrozenInstanceError):
        summary.mean = 0.0  # type: ignore[misc]
    peak = PeakPosterior("u1", 0.2, None, None, None, None, None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        peak.presence_prob = 1.0  # type: ignore[misc]


# ---------------------------------------------------------------- round-trip

finite = st.floats(allow_nan=False, allow_infinity=False)
summaries = st.builds(
    PosteriorSummary,
    mean=finite,
    median=finite,
    sd=finite,
    q025=finite,
    q975=finite,
    at_bound_frac=st.none() | st.floats(min_value=0, max_value=1),
)
maybe_summary = st.none() | summaries
peak_posteriors = st.builds(
    PeakPosterior,
    peak_id=st.text(max_size=32),
    presence_prob=st.floats(min_value=0, max_value=1),
    center=maybe_summary,
    fwhm=maybe_summary,
    area=maybe_summary,
    height=maybe_summary,
    eta=maybe_summary,
)
fit_results = st.builds(
    WindowFitResult,
    spectrum_id=st.integers(min_value=0, max_value=2**32 - 1),
    window_id=st.text(max_size=32),
    status=st.sampled_from(["ok", "failed", "skipped_qc"]),
    peaks=st.lists(peak_posteriors, max_size=3).map(tuple),
    diagnostics=st.dictionaries(st.text(max_size=16), finite, max_size=3),
    runtime_s=st.floats(min_value=0, allow_infinity=False),
    engine=st.text(max_size=16),
    engine_version=st.text(max_size=16),
    error=st.none() | st.text(),
)

FIT_RESULT_ADAPTER = TypeAdapter(WindowFitResult)


@given(fit_results)
def test_window_fit_result_json_round_trip(result: WindowFitResult) -> None:
    json = FIT_RESULT_ADAPTER.dump_json(result)
    assert FIT_RESULT_ADAPTER.validate_json(json) == result


def test_status_rejects_unknown_value() -> None:
    good = FIT_RESULT_ADAPTER.dump_python(
        WindowFitResult(0, "carbon", "ok", (), {}, 0.1, "mock", "0.1.0")
    )
    with pytest.raises(ValidationError):
        FIT_RESULT_ADAPTER.validate_python(good | {"status": "crashed"})
