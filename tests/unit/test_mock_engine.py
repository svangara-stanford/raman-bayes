"""MockEngine (DESIGN.md §6): deterministic stand-in with the PeakEngine interface."""

import dataclasses

import numpy as np
import numpy.typing as npt
import pytest
from scipy.special import expit

from raman_bayes.domain.priors import Window
from raman_bayes.domain.results import PeakPosterior, PosteriorSummary, WindowFitResult
from raman_bayes.engine.mock import MockEngine
from raman_bayes.ports.engine import EngineConfig, PreparedSpectrum

WINDOW = Window.model_validate(
    {
        "id": "lithiation",
        "range_cm1": [350, 700],
        "peaks": [
            {
                "id": "u1",
                "center": {"mean": 465, "bounds": [445, 485]},
                "fwhm": {"bounds": [35, 70]},
            },
            {
                "id": "u2",
                "center": {"mean": 568, "bounds": [550, 585]},
                "fwhm": {"bounds": [30, 60]},
            },
        ],
    }
)
AXIS = np.linspace(350.0, 700.0, 176)  # 2 cm^-1 channels
CFG = EngineConfig()
SEED = np.random.SeedSequence(7, spawn_key=(0, 3))
ENGINE = MockEngine()


def _gauss(axis: npt.NDArray[np.float64], center: float, height: float) -> npt.NDArray[np.float64]:
    return height * np.exp(-0.5 * ((axis - center) / 10.0) ** 2)


def _spectrum(
    intensity: npt.NDArray[np.float64] | None = None,
    noise_sd: float = 10.0,
    axis: npt.NDArray[np.float64] = AXIS,
) -> PreparedSpectrum:
    if intensity is None:
        intensity = _gauss(axis, 466.0, 400.0) + _gauss(axis, 570.0, 5.0)
    return PreparedSpectrum(
        spectrum_id=3, wavenumber=axis, intensity=intensity, noise_sd=noise_sd, qc_flags=0
    )


def _fit(s: PreparedSpectrum, seed: np.random.SeedSequence = SEED) -> WindowFitResult:
    return ENGINE.fit(s, WINDOW, CFG, np.random.default_rng(seed))


def _peak(result: WindowFitResult, peak_id: str) -> PeakPosterior:
    (peak,) = [p for p in result.peaks if p.peak_id == peak_id]
    return peak


def _value(summary: PosteriorSummary | None) -> float:
    assert summary is not None
    assert summary.mean == summary.median
    return summary.mean


def _expected_presence(
    heights: list[float], noise_sd: float, seed: np.random.SeedSequence
) -> list[float]:
    rng = np.random.default_rng(seed)
    out = []
    for height in heights:  # one jitter draw per peak, in YAML order
        jitter = rng.normal(0.0, 0.02)
        out.append(float(np.clip(expit((height / noise_sd - 3.0) / 2.0) + jitter, 0.0, 1.0)))
    return out


def test_name_and_version() -> None:
    assert (MockEngine.name, MockEngine.version) == ("mock", "0.1.0")
    result = _fit(_spectrum())
    assert (result.engine, result.engine_version) == ("mock", "0.1.0")
    assert (result.spectrum_id, result.window_id, result.status) == (3, "lithiation", "ok")
    assert [p.peak_id for p in result.peaks] == ["u1", "u2"]
    assert result.error is None
    assert result.runtime_s >= 0


def test_validate_accepts_any_window_and_refuses_options() -> None:
    assert ENGINE.validate(WINDOW, CFG) == []
    (message,) = ENGINE.validate(WINDOW, EngineConfig(options={"chains": 4}))
    assert "chains" in message


def test_center_is_argmax_within_bounds() -> None:
    # The global maximum (at 400 cm^-1) is outside both peaks' bounds.
    y = _gauss(AXIS, 400.0, 2000.0) + _gauss(AXIS, 476.0, 300.0) + _gauss(AXIS, 556.0, 300.0)
    result = _fit(_spectrum(y))
    inside = (AXIS >= 445) & (AXIS <= 485)
    expected = AXIS[inside][np.argmax(y[inside])]
    assert _value(_peak(result, "u1").center) == expected
    assert _value(_peak(result, "u2").center) == pytest.approx(556.0, abs=2.0)


def test_fwhm_is_bounds_midpoint() -> None:
    result = _fit(_spectrum(_gauss(AXIS, 466.0, 400.0) + _gauss(AXIS, 570.0, 400.0)))
    assert _value(_peak(result, "u1").fwhm) == 52.5
    assert _value(_peak(result, "u2").fwhm) == 45.0


def test_area_is_sum_times_channel_width() -> None:
    axis = 350.0 + np.cumsum(np.linspace(1.5, 2.5, 176))  # uneven channels
    y = _gauss(axis, 466.0, 400.0)
    result = _fit(_spectrum(y, axis=axis))
    inside = (axis >= 445) & (axis <= 485)
    expected = float(np.sum(y[inside] * np.abs(np.gradient(axis))[inside]))
    assert _value(_peak(result, "u1").area) == pytest.approx(expected, rel=1e-12)


def test_height_is_max_within_bounds() -> None:
    y = _gauss(AXIS, 400.0, 2000.0) + _gauss(AXIS, 476.0, 300.0)
    inside = (AXIS >= 445) & (AXIS <= 485)
    assert _value(_peak(_fit(_spectrum(y)), "u1").height) == float(np.max(y[inside]))


def test_eta_is_one_half() -> None:
    assert _value(_peak(_fit(_spectrum()), "u1").eta) == 0.5


def test_presence_formula_with_seeded_jitter() -> None:
    s = _spectrum()
    heights = [
        float(np.max(s.intensity[(lo <= AXIS) & (hi >= AXIS)]))
        for lo, hi in ((445, 485), (550, 585))
    ]
    result = _fit(s)
    expected = _expected_presence(heights, s.noise_sd, SEED)
    assert [p.presence_prob for p in result.peaks] == pytest.approx(expected, abs=1e-15)


def test_presence_clipped_to_unit_interval() -> None:
    strong = _spectrum(_gauss(AXIS, 466.0, 1e6) + _gauss(AXIS, 570.0, 1e6))
    weak = _spectrum(-_gauss(AXIS, 466.0, 1e6) - _gauss(AXIS, 570.0, 1e6))
    values = [
        p.presence_prob
        for k in range(40)
        for s in (strong, weak)
        for p in _fit(s, np.random.SeedSequence(k)).peaks
    ]
    assert min(values) == 0.0
    assert max(values) == 1.0
    assert all(0.0 <= v <= 1.0 for v in values)


def test_absent_peaks_have_none_summaries() -> None:
    result = _fit(_spectrum())  # u1 is 40 sd tall, u2 half a sd
    present, absent = _peak(result, "u1"), _peak(result, "u2")
    assert present.presence_prob >= 0.5
    assert absent.presence_prob < 0.5
    summaries = ("center", "fwhm", "area", "height", "eta")
    assert all(getattr(present, name) is not None for name in summaries)
    assert all(getattr(absent, name) is None for name in summaries)


def test_sd_five_percent_and_quantiles_1_96_sd() -> None:
    peak = _peak(_fit(_spectrum()), "u1")
    for name in ("center", "fwhm", "area", "height", "eta"):
        summary = getattr(peak, name)
        assert isinstance(summary, PosteriorSummary)
        value = summary.mean
        assert summary.sd == pytest.approx(0.05 * abs(value), rel=1e-12)
        assert summary.q025 == pytest.approx(value - 1.96 * summary.sd, rel=1e-12)
        assert summary.q975 == pytest.approx(value + 1.96 * summary.sd, rel=1e-12)
        assert summary.at_bound_frac is None


def test_presence_is_scale_free() -> None:
    # NMC111 counts are about 3x lower than NMC811; presence depends only on height / noise_sd.
    base = _spectrum()
    tripled = _spectrum(base.intensity * 3.0, noise_sd=base.noise_sd * 3.0)
    one, three = _fit(base), _fit(tripled)
    assert [p.presence_prob for p in one.peaks] == pytest.approx(
        [p.presence_prob for p in three.peaks], abs=1e-12
    )
    assert _value(_peak(three, "u1").height) == pytest.approx(3 * _value(_peak(one, "u1").height))
    assert _value(_peak(three, "u1").center) == _value(_peak(one, "u1").center)


def test_zero_noise_sd_does_not_divide_by_zero() -> None:
    y = _gauss(AXIS, 466.0, 400.0)  # u1 positive, u2 essentially 0 but positive
    y[(AXIS >= 550) & (AXIS <= 585)] = 0.0
    result = _fit(_spectrum(y, noise_sd=0.0))
    rng = np.random.default_rng(SEED)
    jitters = [rng.normal(0.0, 0.02) for _ in range(2)]
    expected = [float(np.clip(1.0 + jitters[0], 0, 1)), float(np.clip(0.0 + jitters[1], 0, 1))]
    assert [p.presence_prob for p in result.peaks] == pytest.approx(expected, abs=1e-15)


def test_deterministic_for_a_seed() -> None:
    first = dataclasses.replace(_fit(_spectrum()), runtime_s=0.0)
    second = dataclasses.replace(_fit(_spectrum()), runtime_s=0.0)
    assert first == second
    other = _fit(_spectrum(), np.random.SeedSequence(8))
    assert [p.presence_prob for p in other.peaks] != [p.presence_prob for p in first.peaks]


def test_never_calls_np_random_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_: object) -> None:
        pytest.fail("np.random.seed called")

    # The legacy global state is what np.random.seed would change, so it is read on purpose.
    before = np.random.get_state(legacy=False)  # noqa: NPY002
    monkeypatch.setattr(np.random, "seed", forbidden)
    ENGINE.fit_many([_spectrum()] * 3, WINDOW, CFG, [SEED] * 3)
    after = np.random.get_state(legacy=False)  # noqa: NPY002
    np.testing.assert_array_equal(before["state"]["key"], after["state"]["key"])
    assert before["state"]["pos"] == after["state"]["pos"]
    assert (before["has_gauss"], before["gauss"]) == (after["has_gauss"], after["gauss"])


def test_fit_many_equals_fit_loop() -> None:
    spectra = [
        dataclasses.replace(
            _spectrum(_gauss(AXIS, 466.0, h) + _gauss(AXIS, 570.0, h / 2)), spectrum_id=i
        )
        for i, h in enumerate((20.0, 60.0, 400.0))
    ]
    seeds = [np.random.SeedSequence(7, spawn_key=(0, i)) for i in range(3)]
    many = ENGINE.fit_many(spectra, WINDOW, CFG, seeds)
    loop = [_fit(s, seed) for s, seed in zip(spectra, seeds, strict=True)]
    assert [dataclasses.replace(r, runtime_s=0.0) for r in many] == [
        dataclasses.replace(r, runtime_s=0.0) for r in loop
    ]


def test_no_channel_inside_center_bounds_is_a_failed_fit() -> None:
    sparse_axis = np.array([350.0, 440.0, 490.0, 560.0, 700.0])  # nothing in [445, 485]
    result = _fit(_spectrum(np.ones(5), axis=sparse_axis))
    assert result.status == "failed"
    assert result.peaks == ()
    assert result.error is not None
    assert "u1" in result.error
