"""Arrow tables of fit results in the explicit §8.2 schemas (DESIGN.md §8.2, §15 #6)."""

from pathlib import Path

import pyarrow as pa
import pytest
import yaml

from raman_bayes.domain.priors import PriorsSpec
from raman_bayes.domain.results import (
    FITS_SCHEMA,
    PEAKS_SCHEMA,
    PeakPosterior,
    PosteriorSummary,
    WindowFitResult,
    fits_schema,
    is_fits_schema,
)
from raman_bayes.storage.result_tables import FitRecord, SpectrumPlace, fits_table, peaks_table

FIXTURE = Path(__file__).parents[1] / "fixtures" / "priors" / "design_example.yaml"
PRIORS = PriorsSpec.model_validate(yaml.safe_load(FIXTURE.read_text(encoding="utf-8")))
PLACE = SpectrumPlace(ix=3, iy=1, iz=0, x_um=31.5, y_um=27.9, z_um=42.7)
SUMMARY_COLUMNS = [
    *(
        f"{q}_{s}"
        for q in ("center", "fwhm", "area", "height", "eta")
        for s in ("mean", "median", "sd", "q025", "q975")
    ),
    "center_at_bound_frac",
    "fwhm_at_bound_frac",
]


def _summary(value: float) -> PosteriorSummary:
    return PosteriorSummary(value, value, 0.05 * value, 0.9 * value, 1.1 * value, 0.25)


def _result(status: str = "ok", peaks: tuple[PeakPosterior, ...] = ()) -> WindowFitResult:
    return WindowFitResult(
        spectrum_id=9,
        window_id="lithiation",
        status=status,  # type: ignore[arg-type]
        peaks=peaks,
        diagnostics={},
        runtime_s=0.002,
        engine="mock",
        engine_version="0.1.0",
        error="boom" if status == "failed" else None,
    )


def test_fits_table_unions_diagnostics_as_float32_columns() -> None:
    results = [
        WindowFitResult(1, "lithiation", "ok", (), {"rhat": 1.01}, 0.1, "e", "1"),
        WindowFitResult(1, "carbon", "ok", (), {"ess": 410.0}, 0.2, "e", "1"),
    ]
    table = fits_table("run", [FitRecord(results[0], 2.5), FitRecord(results[1], None)])
    assert table.schema.equals(fits_schema(["ess", "rhat"]))
    assert table.schema.names == [*FITS_SCHEMA.names, "diag_ess", "diag_rhat"]
    assert table.column("diag_rhat").to_pylist() == pytest.approx([1.01, None])
    assert table.column("diag_ess").to_pylist() == [None, 410.0]
    assert table.column("noise_sd").to_pylist() == [2.5, None]
    assert table.column("status").type == pa.dictionary(pa.int32(), pa.string())


def test_is_fits_schema() -> None:
    assert is_fits_schema(FITS_SCHEMA)
    assert is_fits_schema(fits_schema(["rhat"]))
    assert not is_fits_schema(PEAKS_SCHEMA)
    assert not is_fits_schema(FITS_SCHEMA.append(pa.field("diag_rhat", pa.float64())))
    assert not is_fits_schema(FITS_SCHEMA.append(pa.field("extra", pa.float32())))
    assert not is_fits_schema(pa.schema(list(FITS_SCHEMA)[:-1]))


def test_present_peak_row_carries_every_summary() -> None:
    u1 = PeakPosterior("u1", 0.97, *(_summary(v) for v in (465.0, 50.0, 900.0, 40.0, 0.5)))
    table = peaks_table("run", "ds", PRIORS, [_result(peaks=(u1,))], {9: PLACE})
    assert table.schema.equals(PEAKS_SCHEMA)
    rows = table.to_pylist()
    assert [r["peak_id"] for r in rows] == ["u1", "u2"]
    row = rows[0]
    assert (row["run_id"], row["dataset_id"], row["spectrum_id"]) == ("run", "ds", 9)
    assert (row["ix"], row["iy"], row["iz"]) == (3, 1, 0)
    assert (row["x_um"], row["y_um"], row["z_um"]) == (31.5, 27.9, 42.7)
    assert (row["window_id"], row["end_member_id"], row["fit_status"]) == (
        "lithiation",
        "lithiated_oxide",
        "ok",
    )
    assert row["presence_prob"] == pytest.approx(0.97)
    assert row["center_median"] == pytest.approx(465.0)
    assert row["area_q975"] == pytest.approx(990.0)
    assert row["center_at_bound_frac"] == pytest.approx(0.25)
    assert row["fwhm_at_bound_frac"] == pytest.approx(0.25)


def test_unreported_and_absent_peaks_are_null_not_zero() -> None:
    absent = PeakPosterior("u1", 0.2, None, None, None, None, None)
    table = peaks_table("run", "ds", PRIORS, [_result(peaks=(absent,))], {9: PLACE})
    u1, u2 = table.to_pylist()
    for row in (u1, u2):
        assert [row[name] for name in SUMMARY_COLUMNS] == [None] * 27
    assert u1["presence_prob"] == pytest.approx(0.2)
    assert u2["presence_prob"] is None


@pytest.mark.parametrize("status", ["failed", "skipped_qc"])
def test_fits_without_posteriors_give_one_null_row_per_recipe_peak(status: str) -> None:
    table = peaks_table("run", "ds", PRIORS, [_result(status)], {9: PLACE})
    assert table.column("peak_id").to_pylist() == ["u1", "u2"]
    assert table.column("fit_status").to_pylist() == [status, status]
    assert table.column("presence_prob").null_count == 2


def test_peak_in_no_end_member_has_null_end_member_id() -> None:
    data = PRIORS.model_dump()
    data["end_members"][1]["peaks"] = ["d"]
    recipe = PriorsSpec.model_validate(data)
    result = WindowFitResult(9, "carbon", "skipped_qc", (), {}, 0.0, "mock", "0.1.0")
    table = peaks_table("run", "ds", recipe, [result], {9: PLACE})
    assert table.column("end_member_id").to_pylist() == ["carbon", None]
