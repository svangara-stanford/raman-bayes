"""The legacy per-window priors CSV (DESIGN.md §5): import to a Window, export from one.

One row per peak: a name, then center, center bounds and FWHM bounds. Files may carry a UTF-8
BOM and LF or CRLF line endings; exports are LF, no BOM, numbers as Python float reprs. Ids
must be lowercase (§5 rule 1), so a name is lowercased into the id and kept as the label when
that changes it; two names equal up to case are one peak twice.
"""

import csv
from pathlib import Path

from raman_bayes.domain.errors import LegacyCsvHeaderError, LegacyCsvRowError
from raman_bayes.domain.priors import Bounds, CenterPrior, FwhmPrior, Peak, Window

__all__ = ["HEADER", "format_legacy_csv", "read_legacy_csv"]

HEADER = (
    "",
    "x (center)",
    "x (lower limit)",
    "x (upper limit)",
    "FWHM (lower limit)",
    "FWHM (upper limit)",
)


def read_legacy_csv(path: Path, window_id: str, range_cm1: Bounds) -> Window:
    """Read the peaks of ``path`` into a Window with the given id and range.

    The window is not checked against the §5 rules. Raises OSError when ``path`` cannot be
    read, LegacyCsvHeaderError when the first row is not ``HEADER``, and LegacyCsvRowError,
    naming the line, for a malformed or repeated row.
    """
    rows = csv.reader(path.read_bytes().decode("utf-8-sig").splitlines())
    if tuple(next(rows, ())) != HEADER:
        raise LegacyCsvHeaderError(path, HEADER)
    peaks: dict[str, Peak] = {}
    for line_number, row in enumerate(rows, start=2):
        if not row:
            continue
        peak = _peak(path, line_number, row)
        if peak.id in peaks:
            raise LegacyCsvRowError(
                path, line_number, f"peak {row[0]!r} repeats an earlier row (ids ignore case)"
            )
        peaks[peak.id] = peak
    return Window(id=window_id, range_cm1=range_cm1, peaks=tuple(peaks.values()))


def _peak(path: Path, line_number: int, row: list[str]) -> Peak:
    name, *fields = row
    try:
        mean, c_lo, c_hi, f_lo, f_hi = (float(field) for field in fields)
    except ValueError:
        raise LegacyCsvRowError(
            path, line_number, "expected a peak name followed by five numbers"
        ) from None
    if not name:
        raise LegacyCsvRowError(path, line_number, "the peak name is empty")
    peak_id = name.lower()
    return Peak(
        id=peak_id,
        label=name if name != peak_id else None,
        center=CenterPrior(mean=mean, bounds=(c_lo, c_hi)),
        fwhm=FwhmPrior(bounds=(f_lo, f_hi)),
    )


def format_legacy_csv(window: Window) -> str:
    """Return ``window``'s peaks as legacy CSV text: ``HEADER``, one row per peak, LF endings."""
    lines = [",".join(HEADER)]
    for peak in window.peaks:
        values = (peak.center.mean, *peak.center.bounds, *peak.fwhm.bounds)
        lines.append(",".join([peak.id, *(repr(v) for v in values)]))
    return "\n".join(lines) + "\n"
