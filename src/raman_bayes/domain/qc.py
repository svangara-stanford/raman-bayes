"""Per-spectrum QC flags (DESIGN.md §3), computed once at ingest from the raw counts.

Saturation has two signatures: a count at the detector's full scale times the number of
accumulations (the Andor DU920P is 16-bit and counts are summed), or a flat top, where clipping
leaves at least three consecutive samples equal to the spectrum maximum. ``qc_counts``
summarises a dataset's flags for ``ingest`` and ``datasets show``.
"""

import numpy as np
import numpy.typing as npt
from scipy.ndimage import minimum_filter1d

from raman_bayes.domain._base import DomainModel
from raman_bayes.domain.dataset import QCFlag

__all__ = ["FLAT_TOP_SAMPLES", "FULL_SCALE_COUNTS", "QCCounts", "qc_counts", "qc_flags"]

FULL_SCALE_COUNTS = 65535
FLAT_TOP_SAMPLES = 3


def qc_flags(
    intensity: npt.NDArray[np.float64], n_accum: int, axis_mismatch: npt.NDArray[np.bool_]
) -> npt.NDArray[np.uint16]:
    """Return the QCFlag bits of each spectrum, as ``uint16``.

    ``intensity`` is ``[n_spectra, n_wn]`` raw counts and ``axis_mismatch[i]`` whether spectrum
    ``i`` failed the axis check. NaN and infinities set ``nan_or_inf`` only; saturation and
    ``negative`` look at the finite samples.
    """
    finite = np.isfinite(intensity)
    values = np.where(finite, intensity, -np.inf)
    peak = values.max(axis=1, keepdims=True)
    at_peak = (finite & (values == peak)).astype(np.uint8)
    # A sample's 3-wide minimum is 1 exactly when it and both neighbours sit at the maximum.
    flat_top = minimum_filter1d(at_peak, FLAT_TOP_SAMPLES, axis=1, mode="constant").any(axis=1)
    saturated = (peak[:, 0] >= n_accum * FULL_SCALE_COUNTS) | flat_top
    negative = (finite & (intensity < 0)).any(axis=1)
    bits = (
        np.where(~finite.all(axis=1), QCFlag.NAN_OR_INF, 0)
        | np.where(saturated, QCFlag.SATURATED, 0)
        | np.where(axis_mismatch, QCFlag.AXIS_MISMATCH, 0)
        | np.where(negative, QCFlag.NEGATIVE, 0)
    )
    return bits.astype(np.uint16)


class QCCounts(DomainModel):
    """How many spectra carry each QC flag.

    ``flags`` has every QCFlag, lower-case, in bit order, zeros included. ``n_flagged`` counts
    spectra with any flag; a spectrum can carry several, so ``flags`` may sum to more.
    """

    n_spectra: int
    n_flagged: int
    flags: dict[str, int]


def qc_counts(flags: npt.NDArray[np.unsignedinteger]) -> QCCounts:
    """Count the spectra carrying each flag in ``flags``, the per-spectrum QCFlag bits."""
    return QCCounts(
        n_spectra=int(flags.size),
        n_flagged=int(np.count_nonzero(flags)),
        flags={
            str(flag.name).lower(): int(np.count_nonzero(flags & flag.value)) for flag in QCFlag
        },
    )
