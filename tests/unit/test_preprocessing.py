"""Per-window preprocessing (DESIGN.md §7.2): clip, despike, baseline, smooth, clamp, trim."""

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from scipy.stats import norm

from raman_bayes.domain.priors import (
    BaselineSpec,
    DespikeSpec,
    PreprocessingSpec,
    SmoothingSpec,
    Window,
)
from raman_bayes.io.nanofinder import read_spectrum
from raman_bayes.ports.engine import PreparedSpectrum
from raman_bayes.preprocessing import steps
from raman_bayes.preprocessing.prepare import RawSpectrum, prepare, segment
from raman_bayes.preprocessing.steps import als_baseline, despike, noise_sd

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
REAL = read_spectrum(FIXTURES / "nanofinder_mini" / "NMC811_v0" / "NMC811_v0-z0y0x0.txt", 1024)
AXIS = REAL.wavenumber  # the real, non-uniform axis: 1.72 to 2.19 cm^-1 per channel
LO, HI, MARGIN = 350.0, 700.0, 25.0
WINDOW = Window.model_validate(
    {
        "id": "lithiation",
        "range_cm1": [LO, HI],
        "peaks": [
            {
                "id": "u1",
                "center": {"mean": 465, "bounds": [445, 485]},
                "fwhm": {"bounds": [35, 70]},
            }
        ],
    }
)
DESPIKE = DespikeSpec(method="median_mad", kernel=3, thresh=3.0)
BASELINE = BaselineSpec(method="als", lam=1.0e5, p=0.01)
SAVGOL = SmoothingSpec(method="savgol", window=11, polyorder=3)
NORMAL_MAD_SCALE = 1 / norm.ppf(0.75)  # 1.4826...


def _pre(**overrides: Any) -> PreprocessingSpec:
    fields: dict[str, Any] = {
        "margin_cm1": MARGIN,
        "despike": None,
        "baseline": None,
        "smoothing": None,
        "clamp_nonnegative": False,
    }
    return PreprocessingSpec(**(fields | overrides))


def _raw(intensity: npt.NDArray[np.float64], spectrum_id: int = 7, flags: int = 0) -> RawSpectrum:
    return RawSpectrum(spectrum_id=spectrum_id, intensity=intensity, qc_flags=flags)


def _in(lo: float, hi: float, axis: npt.NDArray[np.float64] = AXIS) -> npt.NDArray[np.bool_]:
    return (axis >= lo) & (axis <= hi)


def _noisy(seed: int = 0, sigma: float = 5.0) -> npt.NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    return 1000.0 + rng.normal(0.0, sigma, AXIS.size)


def _gaussian(center: float, height: float, sd_cm1: float) -> npt.NDArray[np.float64]:
    return height * np.exp(-0.5 * ((AXIS - center) / sd_cm1) ** 2)


# ---------------------------------------------------------------- clip and trim


def test_clip_includes_margin() -> None:
    seg = segment(AXIS, (LO, HI), MARGIN)
    np.testing.assert_array_equal(AXIS[seg.clip], AXIS[_in(LO - MARGIN, HI + MARGIN)])
    np.testing.assert_array_equal(AXIS[seg.clip][seg.window], AXIS[_in(LO, HI)])
    assert seg.window.start > 0
    assert seg.window.stop < seg.clip.stop - seg.clip.start


def test_clip_with_zero_margin() -> None:
    seg = segment(AXIS, (LO, HI), 0.0)
    np.testing.assert_array_equal(AXIS[seg.clip], AXIS[_in(LO, HI)])
    assert (seg.window.start, seg.window.stop) == (0, seg.clip.stop - seg.clip.start)


def test_bounds_on_a_channel_are_inclusive() -> None:
    lo, hi = float(AXIS[200]), float(AXIS[300])
    seg = segment(AXIS, (lo, hi), 0.0)
    assert (seg.clip.start, seg.clip.stop) == (200, 301)


def test_trim_returns_exactly_window_channels() -> None:
    prepared = prepare(_raw(REAL.intensity), AXIS, WINDOW, _pre())
    np.testing.assert_array_equal(prepared.wavenumber, AXIS[_in(LO, HI)])
    np.testing.assert_array_equal(prepared.intensity, REAL.intensity[_in(LO, HI)])
    assert prepared.wavenumber.size == 171


def test_trim_on_non_uniform_axis() -> None:
    # Spacing grows from 0.5 to about 4.5 cm^-1, so the two margins hold different channel counts.
    axis = 300.0 + np.cumsum(np.linspace(0.5, 4.5, 200))
    intensity = np.arange(axis.size, dtype=np.float64)
    seg = segment(axis, (LO, HI), MARGIN)
    left = seg.window.start
    right = (seg.clip.stop - seg.clip.start) - seg.window.stop
    assert left > right > 0
    prepared = prepare(_raw(intensity), axis, WINDOW, _pre())
    np.testing.assert_array_equal(prepared.wavenumber, axis[_in(LO, HI, axis)])
    np.testing.assert_array_equal(prepared.intensity, intensity[_in(LO, HI, axis)])


# ---------------------------------------------------------------- despike


def test_single_channel_spike_removed() -> None:
    y = _noisy()
    k = int(np.flatnonzero(_in(LO, HI))[40])
    neighbours = (y[k - 1] + y[k + 1]) / 2
    y[k] += 5000.0
    prepared = prepare(_raw(y), AXIS, WINDOW, _pre(despike=DESPIKE))
    spike = int(np.flatnonzero(prepared.wavenumber == AXIS[k])[0])
    assert abs(prepared.intensity[spike] - neighbours) < 30.0


def test_adjacent_real_peak_preserved() -> None:
    peak = _gaussian(520.0, 800.0, 12.0)  # FWHM ~28 cm^-1, narrower than any prior allows
    k = int(np.argmin(np.abs(AXIS - 545.0)))  # a spike on the peak's flank
    y = _noisy() + peak
    y[k] += 5000.0
    clean = prepare(_raw(_noisy() + peak), AXIS, WINDOW, _pre())
    despiked = prepare(_raw(y), AXIS, WINDOW, _pre(despike=DESPIKE))
    spike = int(np.flatnonzero(despiked.wavenumber == AXIS[k])[0])
    assert abs(despiked.intensity[spike] - clean.intensity[spike]) < 0.05 * 800.0
    # The peak top is left alone. Over 200 noise draws it is still trimmed in about 2 %: the
    # curvature at the top of a peak this narrow sometimes takes the residual past 3 sd.
    assert np.max(despiked.intensity) == np.max(clean.intensity)
    assert int(np.argmax(despiked.intensity)) == int(np.argmax(clean.intensity))


def test_despike_none_is_identity() -> None:
    y = _noisy()
    y[np.flatnonzero(_in(LO, HI))[40]] += 5000.0
    prepared = prepare(_raw(y), AXIS, WINDOW, _pre(despike=None))
    np.testing.assert_array_equal(prepared.intensity, y[_in(LO, HI)])


def test_despike_threshold_is_in_first_difference_sigmas() -> None:
    y = _noisy(seed=3)
    running = median_filter(y, size=3, mode="nearest")
    sigma = NORMAL_MAD_SCALE * np.median(np.abs(np.diff(y) - np.median(np.diff(y)))) / np.sqrt(2)
    expected = np.where(np.abs(y - running) > 3.0 * sigma, running, y)
    np.testing.assert_allclose(despike(y, DESPIKE), expected)


def test_despike_leaves_pure_noise_almost_untouched() -> None:
    # A 3-point median residual is exactly 0 for a third of the samples and shrunk for the
    # rest, so its own MAD is about 0.46 sigma and would flag some 13 % of pure noise.
    changed = [
        np.mean(despike(_noisy(seed, 5.0), DESPIKE) != _noisy(seed, 5.0)) for seed in range(20)
    ]
    assert np.mean(changed) < 0.01


def test_despike_kernel_and_threshold() -> None:
    y = np.zeros(50)
    y[10] = 100.0
    y[30:32] = 100.0  # two channels wide: a 3-point median keeps it
    out = despike(y, DESPIKE)
    assert out[10] == 0.0
    np.testing.assert_array_equal(out[30:32], [100.0, 100.0])
    wide = despike(y, DespikeSpec(method="median_mad", kernel=5, thresh=3.0))
    np.testing.assert_array_equal(wide[30:32], [0.0, 0.0])


# ---------------------------------------------------------------- baseline


def test_als_flattens_sloped_baseline_within_tolerance() -> None:
    sigma = 5.0
    slope = 2000.0 + 3.0 * (AXIS - AXIS[0])
    y = slope + np.random.default_rng(1).normal(0.0, sigma, AXIS.size)
    prepared = prepare(_raw(y), AXIS, WINDOW, _pre(baseline=BASELINE))
    # The asymmetric fit sits under the noise, so the flattened residual is offset by under 3
    # sigma and keeps no trend.
    assert abs(prepared.intensity.mean()) < 3 * sigma
    fit = np.polyfit(prepared.wavenumber, prepared.intensity, 1)
    assert abs(fit[0]) * (HI - LO) < 2 * sigma


def test_als_keeps_peak_height_within_tolerance() -> None:
    height = 600.0
    y = 2000.0 + 3.0 * (AXIS - AXIS[0]) + _gaussian(520.0, height, 15.0)
    y += np.random.default_rng(2).normal(0.0, 5.0, AXIS.size)
    prepared = prepare(_raw(y), AXIS, WINDOW, _pre(baseline=BASELINE))
    assert abs(np.max(prepared.intensity) - height) / height < 0.10


def test_als_baseline_of_a_line_is_the_line() -> None:
    line = 10.0 + 0.5 * np.arange(120.0)
    np.testing.assert_allclose(als_baseline(line, BASELINE), line, rtol=0, atol=1e-6)


# ---------------------------------------------------------------- smoothing and clamp


def test_savgol_uses_channel_window_and_interp_mode() -> None:
    y = _noisy(sigma=20.0)
    zero_margin = _pre(margin_cm1=0.0, smoothing=SAVGOL)
    prepared = prepare(_raw(y), AXIS, WINDOW, zero_margin)
    window = y[_in(LO, HI)]
    expected = savgol_filter(window, 11, 3, deriv=0, mode="interp")
    np.testing.assert_array_equal(prepared.intensity, expected)
    # With no margin the edges are kept, where interp differs from the other modes.
    assert not np.array_equal(prepared.intensity, savgol_filter(window, 11, 3, mode="mirror"))


def test_smoothing_none_is_identity() -> None:
    y = _noisy(sigma=20.0)
    prepared = prepare(_raw(y), AXIS, WINDOW, _pre(smoothing=None))
    np.testing.assert_array_equal(prepared.intensity, y[_in(LO, HI)])


def test_clamp_nonnegative() -> None:
    y = _noisy(sigma=20.0) - 1000.0
    prepared = prepare(_raw(y), AXIS, WINDOW, _pre(clamp_nonnegative=True))
    window = y[_in(LO, HI)]
    assert (window < 0).any()
    np.testing.assert_array_equal(prepared.intensity, np.maximum(window, 0.0))


def test_clamp_false_keeps_negatives() -> None:
    y = _noisy(sigma=20.0) - 1000.0
    prepared = prepare(_raw(y), AXIS, WINDOW, _pre(clamp_nonnegative=False))
    np.testing.assert_array_equal(prepared.intensity, y[_in(LO, HI)])
    assert (prepared.intensity < 0).any()


# ---------------------------------------------------------------- noise


def test_noise_sd_is_scaled_mad_of_median9_residual() -> None:
    y = _noisy(sigma=8.0) + _gaussian(520.0, 300.0, 15.0)
    prepared = prepare(_raw(y), AXIS, WINDOW, _pre())
    trimmed = y[_in(LO, HI)]
    residual = trimmed - median_filter(trimmed, size=9, mode="nearest")
    expected = NORMAL_MAD_SCALE * np.median(np.abs(residual - np.median(residual)))
    assert prepared.noise_sd == pytest.approx(expected, rel=1e-12)
    assert noise_sd(trimmed) == pytest.approx(expected, rel=1e-12)


def test_noise_sd_of_white_noise_near_sigma() -> None:
    sigma = 7.0
    y = np.random.default_rng(3).normal(100.0, sigma, 4000)
    # The residual of a 9-point median is exactly 0 wherever a sample is its own window's
    # median (1 in 9), which shrinks the MAD: on white noise the estimate is 0.864 sigma.
    assert noise_sd(y) == pytest.approx(0.864 * sigma, rel=0.05)


def test_noise_sd_of_a_constant_is_zero() -> None:
    assert noise_sd(np.full(50, 3.0)) == 0.0


# ---------------------------------------------------------------- the pipeline


def test_steps_run_in_section_7_2_order(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int]] = []

    def recorder(name: str, real: Any) -> Any:
        def record(y: npt.NDArray[np.float64], *args: Any) -> Any:
            calls.append((name, y.size))
            return real(y, *args)

        return record

    for name in ("despike", "als_baseline", "smooth", "clamp_nonnegative", "noise_sd"):
        monkeypatch.setattr(steps, name, recorder(name, getattr(steps, name)))
    full = _pre(despike=DESPIKE, baseline=BASELINE, smoothing=SAVGOL, clamp_nonnegative=True)
    prepared = prepare(_raw(REAL.intensity), AXIS, WINDOW, full)
    clipped = int(_in(LO - MARGIN, HI + MARGIN).sum())
    trimmed = int(_in(LO, HI).sum())
    assert calls == [
        ("despike", clipped),
        ("als_baseline", clipped),
        ("smooth", clipped),
        ("clamp_nonnegative", clipped),
        ("noise_sd", trimmed),
    ]
    assert prepared.wavenumber.size == trimmed


def test_prepare_returns_float64_prepared_spectrum() -> None:
    full = _pre(despike=DESPIKE, baseline=BASELINE, smoothing=SAVGOL, clamp_nonnegative=True)
    raw = _raw(REAL.intensity.astype(np.float32), spectrum_id=11, flags=8)
    prepared = prepare(raw, AXIS, WINDOW, full)
    assert isinstance(prepared, PreparedSpectrum)
    assert prepared.wavenumber.dtype == prepared.intensity.dtype == np.float64
    assert prepared.wavenumber.shape == prepared.intensity.shape == (171,)
    assert (prepared.spectrum_id, prepared.qc_flags) == (11, 8)
    assert type(prepared.noise_sd) is float
    assert prepared.noise_sd > 0
    assert (prepared.intensity >= 0).all()


def test_preprocessing_is_deterministic() -> None:
    full = _pre(despike=DESPIKE, baseline=BASELINE, smoothing=SAVGOL, clamp_nonnegative=True)
    first = prepare(_raw(REAL.intensity.copy()), AXIS, WINDOW, full)
    second = prepare(_raw(REAL.intensity.copy()), AXIS, WINDOW, full)
    assert first.intensity.tobytes() == second.intensity.tobytes()
    assert first.noise_sd == second.noise_sd


def test_prepare_does_not_modify_its_input() -> None:
    y = REAL.intensity.copy()
    full = _pre(despike=DESPIKE, baseline=BASELINE, smoothing=SAVGOL, clamp_nonnegative=True)
    prepare(_raw(y), AXIS, WINDOW, full)
    np.testing.assert_array_equal(y, REAL.intensity)
