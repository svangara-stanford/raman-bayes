"""Spectrum filename parsing: ``<scan>-z<iz>y<iy>x<ix>.txt`` by default (DESIGN.md §3).

Another instrument can supply its own pattern; it must define the named groups ``scan``,
``iz``, ``iy`` and ``ix``. Patterns are matched against the whole name, anchored or not.
"""

import re
from dataclasses import dataclass

from raman_bayes.domain.dataset import FILENAME_RE
from raman_bayes.domain.errors import FilenamePatternGroupsError, FilenamePatternSyntaxError

__all__ = [
    "DEFAULT_PATTERN",
    "REQUIRED_GROUPS",
    "SpectrumName",
    "compile_pattern",
    "parse_filename",
]

REQUIRED_GROUPS = ("scan", "iz", "iy", "ix")


@dataclass(frozen=True)
class SpectrumName:
    """The scan name and grid indices encoded in one spectrum filename."""

    scan: str
    iz: int
    iy: int
    ix: int


def compile_pattern(regex: str) -> re.Pattern[str]:
    """Compile a filename pattern.

    Raises FilenamePatternSyntaxError when ``regex`` does not compile and
    FilenamePatternGroupsError when it lacks one of ``REQUIRED_GROUPS``.
    """
    try:
        pattern = re.compile(regex)
    except re.error as exc:
        raise FilenamePatternSyntaxError(regex, exc) from exc
    missing = [group for group in REQUIRED_GROUPS if group not in pattern.groupindex]
    if missing:
        raise FilenamePatternGroupsError(regex, missing)
    return pattern


DEFAULT_PATTERN = compile_pattern(FILENAME_RE)


def parse_filename(name: str, pattern: re.Pattern[str] = DEFAULT_PATTERN) -> SpectrumName | None:
    """Return the scan and indices in ``name``, or None when it is not a spectrum filename.

    ``pattern`` must come from ``compile_pattern``. Indices are integers, so ``z01`` and ``z1``
    give the same index.
    """
    # fullmatch, not match: FILENAME_RE ends in "$", which also matches before a final "\n".
    match = pattern.fullmatch(name)
    if match is None:
        return None
    return SpectrumName(
        scan=match["scan"], iz=int(match["iz"]), iy=int(match["iy"]), ix=int(match["ix"])
    )
