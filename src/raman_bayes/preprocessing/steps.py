"""The preprocessing steps of DESIGN.md §7.2, each on one clipped segment of one spectrum.

Every step returns a new array and leaves its input alone. Spreads are the median absolute
deviation scaled to a normal sd (x 1.4826), so a threshold in "MADs" reads as one in sigmas.
"""

import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.linalg import solveh_banded
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from scipy.stats import median_abs_deviation

from raman_bayes.domain.priors import BaselineSpec, DespikeSpec, SmoothingSpec

__all__ = [
    "ALS_MAX_ITERATIONS",
    "NOISE_FILTER_CHANNELS",
    "als_baseline",
    "clamp_nonnegative",
    "despike",
    "noise_sd",
    "smooth",
]

ALS_MAX_ITERATIONS = 10
NOISE_FILTER_CHANNELS = 9

Signal = npt.NDArray[np.float64]


def despike(y: Signal, spec: DespikeSpec) -> Signal:
    """Replace spikes by the running median.

    A spike is a sample further than ``spec.thresh`` scaled MADs of the residual from the
    ``spec.kernel``-channel running median. A one-channel cosmic ray leaves a large residual;
    a peak several channels wide follows the median and is kept.
    """
    baseline = median_filter(y, size=spec.kernel, mode="nearest")
    residual = y - baseline
    spread = median_abs_deviation(residual, scale="normal")
    return np.where(np.abs(residual) > spec.thresh * spread, baseline, y)


def als_baseline(y: Signal, spec: BaselineSpec) -> Signal:
    """Return the asymmetric least-squares baseline of ``y`` (Eilers and Boelens, 2005).

    Minimises ``sum(w * (y - z)**2) + lam * sum(diff(z, 2)**2)``, where samples above the
    baseline get weight ``p`` and those below ``1 - p``, reweighting until the weights settle or
    ``ALS_MAX_ITERATIONS`` passes. ``y`` needs at least three samples.
    """
    n = y.size
    second_difference = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n))
    penalty = spec.lam * (second_difference.T @ second_difference)
    # The system is symmetric and pentadiagonal; solveh_banded takes its upper bands.
    bands = np.zeros((3, n))
    bands[0, 2:] = penalty.diagonal(2)
    bands[1, 1:] = penalty.diagonal(1)
    bands[2] = penalty.diagonal(0)
    weights = np.ones(n)
    for _ in range(ALS_MAX_ITERATIONS):
        system = bands.copy()
        system[2] += weights
        z: Signal = solveh_banded(system, weights * y)
        updated = np.where(y > z, spec.p, 1.0 - spec.p)
        if np.array_equal(updated, weights):
            break
        weights = updated
    return z


def smooth(y: Signal, spec: SmoothingSpec) -> Signal:
    """Savitzky-Golay smoothing over ``spec.window`` channels, ``interp`` mode at the ends."""
    smoothed: Signal = savgol_filter(y, spec.window, spec.polyorder, deriv=0, mode="interp")
    return smoothed


def clamp_nonnegative(y: Signal) -> Signal:
    """Return ``y`` with negative samples set to 0."""
    return np.maximum(y, 0.0)


def noise_sd(y: Signal) -> float:
    """Return the robust noise sd of ``y``: 1.4826 x MAD of its residual from a 9-point median.

    The running median follows peaks and baseline, so the residual is mostly noise. Ends use
    ``mode="nearest"``. A constant ``y`` gives 0.

    On white noise of sd sigma this reads about 0.864 sigma, not sigma: the median of 9 is one
    of the 9, so about 1 residual in 9 is exactly 0, which pulls the MAD down. Kept as decided
    for Phase 1 (B3); to be revisited when the real engine defines the noise it needs.
    """
    residual = y - median_filter(y, size=NOISE_FILTER_CHANNELS, mode="nearest")
    return float(median_abs_deviation(residual, scale="normal"))
