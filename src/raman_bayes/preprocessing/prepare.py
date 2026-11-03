"""Turn one raw spectrum into the PreparedSpectrum an engine fits for one window (DESIGN.md §7.2).

Order: clip to the window's range plus ``margin_cm1`` on each side, despike, subtract the
baseline, smooth, clamp, trim the margin, then estimate ``noise_sd`` on what is left. Steps that
look at neighbours (median filters, the baseline's smoothness penalty, Savitzky-Golay) therefore
have their edge effects in the margin, which trim removes. Channel ranges are inclusive and
come from the dataset axis, which is ascending but not evenly spaced in cm^-1.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from raman_bayes.domain.priors import Bounds, PreprocessingSpec, Window
from raman_bayes.ports.engine import PreparedSpectrum
from raman_bayes.preprocessing import steps

__all__ = ["RawSpectrum", "Segment", "prepare", "segment"]


@dataclass(frozen=True)
class Segment:
    """The channels of one window on an axis.

    ``clip`` selects ``range ± margin`` from the full axis; ``window`` selects ``range`` from
    the clipped segment.
    """

    clip: slice
    window: slice


@dataclass(frozen=True)
class RawSpectrum:
    """One stored spectrum: its id, counts on the full dataset axis, and QC flags."""

    spectrum_id: int
    intensity: npt.NDArray[np.floating]
    qc_flags: int


def segment(wavenumber: npt.NDArray[np.float64], range_cm1: Bounds, margin_cm1: float) -> Segment:
    """Return the channels of ``range_cm1`` and of ``range_cm1`` widened by ``margin_cm1``.

    ``wavenumber`` must be ascending. A channel exactly on a bound is included.
    """
    lo, hi = range_cm1
    start = int(np.searchsorted(wavenumber, lo - margin_cm1, side="left"))
    stop = int(np.searchsorted(wavenumber, hi + margin_cm1, side="right"))
    first = int(np.searchsorted(wavenumber, lo, side="left"))
    last = int(np.searchsorted(wavenumber, hi, side="right"))
    return Segment(clip=slice(start, stop), window=slice(first - start, last - start))


def prepare(
    raw: RawSpectrum,
    wavenumber: npt.NDArray[np.float64],
    window: Window,
    preprocessing: PreprocessingSpec,
) -> PreparedSpectrum:
    """Preprocess ``raw`` for ``window`` in the §7.2 order; float64 out, ``raw`` unchanged.

    ``wavenumber`` is the full dataset axis that ``raw.intensity`` is sampled on. The window
    plus margin must cover at least three channels when a baseline is fitted and
    ``smoothing.window`` channels when smoothing; plan time checks both (§5 rule 8).
    """
    seg = segment(wavenumber, window.range_cm1, preprocessing.margin_cm1)
    y = np.array(raw.intensity[seg.clip], dtype=np.float64)
    if preprocessing.despike is not None:
        y = steps.despike(y, preprocessing.despike)
    if preprocessing.baseline is not None:
        y = y - steps.als_baseline(y, preprocessing.baseline)
    if preprocessing.smoothing is not None:
        y = steps.smooth(y, preprocessing.smoothing)
    if preprocessing.clamp_nonnegative:
        y = steps.clamp_nonnegative(y)
    y = y[seg.window].copy()
    return PreparedSpectrum(
        spectrum_id=int(raw.spectrum_id),
        wavenumber=np.array(wavenumber[seg.clip][seg.window], dtype=np.float64),
        intensity=y,
        noise_sd=steps.noise_sd(y),
        qc_flags=int(raw.qc_flags),
    )
