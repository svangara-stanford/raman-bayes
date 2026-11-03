"""Per-spectrum QC flags (DESIGN.md §3): NaN/inf, saturation, axis mismatch, negative counts."""

import numpy as np
import numpy.typing as npt
import pytest

from raman_bayes.domain.dataset import QCFlag
from raman_bayes.domain.qc import QCCounts, qc_counts, qc_flags

N_WN = 64


def _baseline() -> npt.NDArray[np.float64]:
    # A smooth positive spectrum with a unique maximum, so nothing is flagged.
    x = np.arange(N_WN, dtype=np.float64)
    return 1000.0 + 500.0 * np.exp(-0.5 * ((x - 20.0) / 3.0) ** 2)


def _flags(*rows: npt.NDArray[np.float64], n_accum: int = 1) -> list[QCFlag]:
    flags = qc_flags(np.vstack(rows), n_accum, np.zeros(len(rows), dtype=bool))
    assert flags.dtype == np.uint16
    return [QCFlag(int(f)) for f in flags]


def test_clean_spectrum_has_no_flags() -> None:
    assert _flags(_baseline()) == [QCFlag(0)]


@pytest.mark.parametrize("n_accum", [1, 5, 10])
def test_saturated_at_n_accum_times_65535(n_accum: int) -> None:
    y = _baseline()
    y[40] = n_accum * 65535
    assert _flags(y, n_accum=n_accum) == [QCFlag.SATURATED]


def test_just_below_threshold_not_saturated() -> None:
    y = _baseline()
    y[40] = 5 * 65535 - 1
    assert _flags(y, n_accum=5) == [QCFlag(0)]


def test_flat_top_of_three_sets_saturated() -> None:
    y = _baseline()
    y[19:22] = 2000.0
    assert _flags(y) == [QCFlag.SATURATED]


def test_two_equal_samples_at_max_not_flat_top() -> None:
    y = _baseline()
    y[19:21] = 2000.0
    # Three equal samples that are not the maximum are not a flat top either.
    y[40:43] = 1800.0
    assert _flags(y) == [QCFlag(0)]


def test_tall_legitimate_peak_not_flagged() -> None:
    x = np.arange(N_WN, dtype=np.float64)
    y = 1000.0 + 60000.0 * np.exp(-0.5 * ((x - 30.0) / 2.0) ** 2)  # far above 65535 / n_accum
    assert np.max(y) < 5 * 65535
    assert _flags(y, n_accum=5) == [QCFlag(0)]


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nan_or_inf_flag(bad: float) -> None:
    y = _baseline()
    y[10] = bad
    assert _flags(y) == [QCFlag.NAN_OR_INF]


def test_negative_flag() -> None:
    y = _baseline()
    y[5] = -1.0
    assert _flags(y) == [QCFlag.NEGATIVE]


def test_axis_mismatch_flag() -> None:
    rows = np.vstack([_baseline(), _baseline()])
    flags = qc_flags(rows, 1, np.array([False, True]))
    assert [QCFlag(int(f)) for f in flags] == [QCFlag(0), QCFlag.AXIS_MISMATCH]


def test_flags_combine_in_uint16() -> None:
    y = _baseline()
    y[5] = -1.0
    y[10] = np.nan
    y[40] = 65535.0
    (flags,) = qc_flags(y[np.newaxis, :], 1, np.array([True]))
    assert flags == 0b1111
    assert QCFlag(int(flags)) == (
        QCFlag.NAN_OR_INF | QCFlag.SATURATED | QCFlag.AXIS_MISMATCH | QCFlag.NEGATIVE
    )


def test_rows_are_flagged_independently() -> None:
    saturated = _baseline()
    saturated[40] = 65535.0
    assert _flags(_baseline(), saturated, _baseline()) == [
        QCFlag(0),
        QCFlag.SATURATED,
        QCFlag(0),
    ]


def test_qc_counts_per_flag_and_flagged_spectra() -> None:
    flags = np.array(
        [0, QCFlag.SATURATED, QCFlag.SATURATED | QCFlag.NEGATIVE, QCFlag.NAN_OR_INF, 0],
        dtype=np.uint16,
    )
    counts = qc_counts(flags)
    assert counts == QCCounts(
        n_spectra=5,
        n_flagged=3,  # the per-flag counts sum to 4: one spectrum carries two flags
        flags={"nan_or_inf": 1, "saturated": 2, "axis_mismatch": 0, "negative": 1},
    )
    assert list(counts.flags) == ["nan_or_inf", "saturated", "axis_mismatch", "negative"]


def test_qc_counts_of_a_clean_dataset_lists_every_flag() -> None:
    counts = qc_counts(np.zeros(25, dtype=np.uint16))
    assert counts.model_dump(mode="json") == {
        "n_spectra": 25,
        "n_flagged": 0,
        "flags": {"nan_or_inf": 0, "saturated": 0, "axis_mismatch": 0, "negative": 0},
    }
