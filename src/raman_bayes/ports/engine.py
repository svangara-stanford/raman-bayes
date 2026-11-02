"""PeakEngine Protocol and its inputs (DESIGN.md §6).

Engines do no I/O, read no global config, never call ``np.random.seed`` and spawn no threads.
The orchestrator calls ``fit_many``; an engine that subclasses PeakEngine explicitly inherits
the default, which loops over ``fit`` with one Generator per SeedSequence.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from raman_bayes.domain.priors import Window
from raman_bayes.domain.results import WindowFitResult


@dataclass(frozen=True)
class PreparedSpectrum:
    """One spectrum, preprocessed and trimmed to a window (margin removed)."""

    spectrum_id: int
    wavenumber: npt.NDArray[np.float64]
    intensity: npt.NDArray[np.float64]
    noise_sd: float  # 1.4826 x MAD of the residual of a 9-point median filter (§7.2)
    qc_flags: int


@dataclass(frozen=True)
class EngineConfig:
    """Sampler settings; ``options`` holds engine-specific keys."""

    n_iter: int = 6000
    burn_in: int = 3000
    thin: int = 5
    options: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class PeakEngine(Protocol):
    """Fits the peaks of one window to preprocessed spectra."""

    name: str
    version: str

    def validate(self, window: Window, cfg: EngineConfig) -> list[str]:
        """Return the reasons this engine cannot fit ``window`` with ``cfg``; empty if it can."""

    def fit(
        self, s: PreparedSpectrum, window: Window, cfg: EngineConfig, rng: np.random.Generator
    ) -> WindowFitResult:
        """Fit ``window`` to ``s``, drawing randomness only from ``rng``."""

    def fit_many(
        self,
        ss: Sequence[PreparedSpectrum],
        window: Window,
        cfg: EngineConfig,
        seeds: Sequence[np.random.SeedSequence],
    ) -> list[WindowFitResult]:
        """Fit each spectrum with its own seed; results are in the order of ``ss``.

        Raises ValueError when ``ss`` and ``seeds`` differ in length.
        """
        return [
            self.fit(s, window, cfg, np.random.default_rng(seed))
            for s, seed in zip(ss, seeds, strict=True)
        ]
