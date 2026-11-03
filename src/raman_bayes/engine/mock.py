"""MockEngine (DESIGN.md §6): a deterministic, microsecond stand-in for the Bayesian engine.

Each peak is read straight off the preprocessed spectrum inside its center bounds: center at
the maximum, height the maximum, area the counts times channel width, FWHM the middle of its
bounds, eta 0.5. Presence is ``sigmoid((height / noise_sd - 3) / 2)`` plus seeded jitter, so it
depends on signal-to-noise and not on absolute counts. Summaries are the value with a 5 % sd.
"""

import time

import numpy as np
from scipy.special import expit

from raman_bayes.domain.priors import Bounds, Peak, Window
from raman_bayes.domain.results import PeakPosterior, PosteriorSummary, WindowFitResult
from raman_bayes.ports.engine import EngineConfig, PeakEngine, PreparedSpectrum

__all__ = ["MockEngine"]

_JITTER_SD = 0.02
_PRESENT = 0.5  # below this presence the summaries are None (absent)
_SNR_OFFSET = 3.0
_SNR_SCALE = 2.0
_RELATIVE_SD = 0.05
_Z_975 = 1.96
_ETA = 0.5


class MockEngine(PeakEngine):
    """The ``mock`` engine; its ``fit_many`` is the PeakEngine default loop over ``fit``."""

    name = "mock"
    version = "0.1.0"

    def validate(self, window: Window, cfg: EngineConfig) -> list[str]:
        """Accept every window; refuse engine options, since the mock has none."""
        if cfg.options:
            return [f"the mock engine takes no options; got {', '.join(sorted(cfg.options))}"]
        return []

    def fit(
        self, s: PreparedSpectrum, window: Window, cfg: EngineConfig, rng: np.random.Generator
    ) -> WindowFitResult:
        """Fit every peak of ``window``; ``rng`` gives one jitter draw per peak, in recipe order.

        A peak whose center bounds hold no channel makes the whole fit ``failed``.
        """
        started = time.perf_counter()
        inside = [_within(s, peak.center.bounds) for peak in window.peaks]
        empty = [peak.id for peak, mask in zip(window.peaks, inside, strict=True) if not mask.any()]
        if empty:
            error = f"no channel inside center.bounds of peak(s) {', '.join(empty)}"
            return self._result(s, window, (), started, error)
        widths = np.abs(np.gradient(s.wavenumber))
        peaks = tuple(
            _peak(peak, s, mask, widths, float(rng.normal(0.0, _JITTER_SD)))
            for peak, mask in zip(window.peaks, inside, strict=True)
        )
        return self._result(s, window, peaks, started, None)

    def _result(
        self,
        s: PreparedSpectrum,
        window: Window,
        peaks: tuple[PeakPosterior, ...],
        started: float,
        error: str | None,
    ) -> WindowFitResult:
        return WindowFitResult(
            spectrum_id=s.spectrum_id,
            window_id=window.id,
            status="failed" if error else "ok",
            peaks=peaks,
            diagnostics={},
            runtime_s=time.perf_counter() - started,
            engine=self.name,
            engine_version=self.version,
            error=error,
        )


def _within(s: PreparedSpectrum, bounds: Bounds) -> np.ndarray:
    lo, hi = bounds
    return (s.wavenumber >= lo) & (s.wavenumber <= hi)


def _peak(
    peak: Peak, s: PreparedSpectrum, inside: np.ndarray, widths: np.ndarray, jitter: float
) -> PeakPosterior:
    y = s.intensity[inside]
    top = int(np.argmax(y))
    height = float(y[top])
    presence = float(np.clip(_presence(height, s.noise_sd) + jitter, 0.0, 1.0))
    if presence < _PRESENT:
        return PeakPosterior(peak.id, presence, None, None, None, None, None)
    lo, hi = peak.fwhm.bounds
    return PeakPosterior(
        peak_id=peak.id,
        presence_prob=presence,
        center=_summary(float(s.wavenumber[inside][top])),
        fwhm=_summary((lo + hi) / 2),
        area=_summary(float(np.sum(y * widths[inside]))),
        height=_summary(height),
        eta=_summary(_ETA),
    )


def _presence(height: float, noise_sd: float) -> float:
    # With no noise any positive height is certain; the ratio would divide by zero.
    if noise_sd == 0:
        return 1.0 if height > 0 else 0.0
    return float(expit((height / noise_sd - _SNR_OFFSET) / _SNR_SCALE))


def _summary(value: float) -> PosteriorSummary:
    sd = _RELATIVE_SD * abs(value)
    return PosteriorSummary(
        mean=value, median=value, sd=sd, q025=value - _Z_975 * sd, q975=value + _Z_975 * sd
    )
