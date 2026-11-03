"""The priors rules (DESIGN.md §5): file-level rules 1-7, rule 8 on a dataset axis, rule 9 hash.

Rule 8 needs a dataset axis and runs at plan time (``check_axis``). Every rule reports every
violation it finds,
so a user fixes a file in one pass. ``check_window`` holds the rules that need only one window,
so an imported window fragment can be checked before it joins a file. Ids are unique within
their kind (windows, peaks across all windows, end members): the §5 example names both a
window and an end member ``carbon``.
"""

import hashlib
import itertools
import json
import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from raman_bayes.domain.priors import Bounds, Peak, PriorsSpec, Window

__all__ = [
    "ID_PATTERN",
    "KNOWN_SCHEMA_VERSIONS",
    "MIN_SMOOTHING_WINDOW",
    "MIN_WINDOW_CHANNELS",
    "Finding",
    "RuleReport",
    "Severity",
    "canonical_json",
    "check_axis",
    "check_rules",
    "check_window",
    "priors_hash",
]

KNOWN_SCHEMA_VERSIONS = frozenset({"1.0"})
MIN_SMOOTHING_WINDOW = 3
MIN_WINDOW_CHANNELS = 32
ID_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,31}")  # always fullmatch: "$" would accept "id\n"

Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class Finding:
    """One rule violation: ``rule`` is its §5 number; errors block a run, warnings do not."""

    rule: int
    severity: Severity
    message: str


@dataclass(frozen=True)
class RuleReport:
    """Every finding from ``check_rules``, in rule order."""

    findings: tuple[Finding, ...]

    @property
    def errors(self) -> tuple[str, ...]:
        """Messages of the error findings."""
        return tuple(f.message for f in self.findings if f.severity == "error")

    @property
    def warnings(self) -> tuple[str, ...]:
        """Messages of the warning findings."""
        return tuple(f.message for f in self.findings if f.severity == "warning")


def check_rules(spec: PriorsSpec) -> RuleReport:
    """Apply rules 1-7 to ``spec`` and return every finding, by rule; never raises."""
    findings = itertools.chain(
        _schema_and_unique_ids(spec),
        _bad_ids("end member", [m.id for m in spec.end_members]),
        *(_window_findings(window) for window in spec.windows),
        _rule4(spec),
        _rule6(spec),
        _rule7(spec),
    )
    return RuleReport(tuple(sorted(findings, key=lambda f: f.rule)))


def check_window(window: Window) -> RuleReport:
    """Apply the single-window rules (1 id pattern, 2, 3, 5) to ``window``; never raises."""
    return RuleReport(tuple(sorted(_window_findings(window), key=lambda f: f.rule)))


def _window_findings(window: Window) -> Iterator[Finding]:
    yield from _bad_ids("window", [window.id])
    yield from _bad_ids("peak", [p.id for p in window.peaks])
    yield from _rule2(window)
    yield from _rule3(window)
    yield from _rule5(window)


def check_axis(spec: PriorsSpec, wavenumber: npt.NDArray[np.float64]) -> RuleReport:
    """Apply rule 8 to every window of ``spec`` on an ascending dataset axis; never raises.

    Each window must cover at least ``MIN_WINDOW_CHANNELS`` channels, lie within the axis
    once widened by ``margin_cm1``, and, when smoothing is set, its widened segment must hold
    at least ``smoothing.window`` channels. Channel ranges include their bounds.
    """
    return RuleReport(tuple(f for window in spec.windows for f in _rule8(spec, window, wavenumber)))


def _rule8(spec: PriorsSpec, window: Window, axis: npt.NDArray[np.float64]) -> Iterator[Finding]:
    lo, hi = window.range_cm1
    margin = spec.preprocessing.margin_cm1
    smoothing = spec.preprocessing.smoothing
    channels = int(np.count_nonzero((axis >= lo) & (axis <= hi)))
    if channels < MIN_WINDOW_CHANNELS:
        yield Finding(
            8,
            "error",
            f"window {window.id}: range_cm1 {list(window.range_cm1)} covers {channels} channels "
            f"of the dataset axis; at least {MIN_WINDOW_CHANNELS} are needed",
        )
    if lo - margin < axis[0] or hi + margin > axis[-1]:
        yield Finding(
            8,
            "error",
            f"window {window.id}: range_cm1 widened by margin_cm1 ({margin:g}) spans "
            f"[{lo - margin:g}, {hi + margin:g}], beyond the dataset axis "
            f"[{axis[0]:g}, {axis[-1]:g}]",
        )
    widened = int(np.count_nonzero((axis >= lo - margin) & (axis <= hi + margin)))
    if smoothing is not None and widened < smoothing.window:
        yield Finding(
            8,
            "error",
            f"window {window.id}: range_cm1 widened by margin_cm1 covers {widened} channels, "
            f"fewer than smoothing.window ({smoothing.window})",
        )


def canonical_json(spec: PriorsSpec) -> bytes:
    """Return ``spec`` as JSON with sorted keys, no whitespace, unescaped UTF-8.

    Defaults are included, so a key spelled out with its default value hashes the same as an
    omitted one; numbers are the parsed floats, so ``465`` and ``465.0`` agree.
    """
    text = json.dumps(
        spec.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return text.encode("utf-8")


def priors_hash(spec: PriorsSpec) -> str:
    """Return the SHA-256 hex digest of ``canonical_json(spec)`` (rule 9)."""
    return hashlib.sha256(canonical_json(spec)).hexdigest()


def _overlap(a: Bounds, b: Bounds) -> bool:
    # Open intervals: windows or bounds that only touch do not overlap.
    return a[0] < b[1] and b[0] < a[1]


def _schema_and_unique_ids(spec: PriorsSpec) -> Iterator[Finding]:
    if spec.schema_version not in KNOWN_SCHEMA_VERSIONS:
        known = ", ".join(sorted(KNOWN_SCHEMA_VERSIONS))
        yield Finding(1, "error", f"schema_version {spec.schema_version!r} is not one of: {known}")
    kinds = {
        "window": [w.id for w in spec.windows],
        "peak": [p.id for w in spec.windows for p in w.peaks],
        "end member": [m.id for m in spec.end_members],
    }
    for kind, ids in kinds.items():
        for id_, count in Counter(ids).items():
            if count > 1:
                yield Finding(1, "error", f"{kind} id {id_!r} is used {count} times")


def _bad_ids(kind: str, ids: list[str]) -> Iterator[Finding]:
    for id_ in ids:
        if ID_PATTERN.fullmatch(id_) is None:
            yield Finding(
                1,
                "error",
                f"{kind} id {id_!r} must be a lowercase letter followed by up to 31 "
                "lowercase letters, digits or underscores",
            )


def _rule2(window: Window) -> Iterator[Finding]:
    lo, hi = window.range_cm1
    for peak in window.peaks:
        c_lo, c_hi = peak.center.bounds
        where = f"peak {peak.id} in window {window.id}"
        if not (lo <= c_lo and c_hi <= hi):
            yield Finding(
                2,
                "error",
                f"{where}: center.bounds {list(peak.center.bounds)} are not inside "
                f"range_cm1 {list(window.range_cm1)}",
            )
        elif min(c_lo - lo, hi - c_hi) < peak.fwhm.bounds[0]:
            yield Finding(
                2,
                "warning",
                f"{where}: center.bounds {list(peak.center.bounds)} come within the minimum "
                f"FWHM ({peak.fwhm.bounds[0]}) of the window edge",
            )


def _rule3(window: Window) -> Iterator[Finding]:
    if not window.range_cm1[0] < window.range_cm1[1]:
        yield Finding(3, "error", f"window {window.id}: range_cm1 lower must be below upper")
    for peak in window.peaks:
        yield from _peak_bounds(window, peak)


def _peak_bounds(window: Window, peak: Peak) -> Iterator[Finding]:
    where = f"peak {peak.id} in window {window.id}"
    center, fwhm = peak.center, peak.fwhm.bounds
    width = window.range_cm1[1] - window.range_cm1[0]
    for name, (lower, upper) in (("center.bounds", center.bounds), ("fwhm.bounds", fwhm)):
        if not lower < upper:
            yield Finding(3, "error", f"{where}: {name} lower {lower} must be below upper {upper}")
    if not center.bounds[0] <= center.mean <= center.bounds[1]:
        yield Finding(
            3, "error", f"{where}: center.mean {center.mean} is outside {list(center.bounds)}"
        )
    if not (fwhm[0] > 0 and fwhm[1] > 0):
        yield Finding(3, "error", f"{where}: fwhm.bounds {list(fwhm)} must be positive")
    if fwhm[1] > width:
        yield Finding(
            3, "error", f"{where}: fwhm.bounds upper {fwhm[1]} exceeds the window width {width}"
        )


def _rule4(spec: PriorsSpec) -> Iterator[Finding]:
    for a, b in itertools.combinations(spec.windows, 2):
        if _overlap(a.range_cm1, b.range_cm1) and not (a.allow_overlap and b.allow_overlap):
            yield Finding(
                4,
                "error",
                f"windows {a.id} and {b.id} overlap; set allow_overlap: true on both to allow it",
            )


def _rule5(window: Window) -> Iterator[Finding]:
    if window.ordered:
        return
    for a, b in itertools.combinations(window.peaks, 2):
        if _overlap(a.center.bounds, b.center.bounds):
            yield Finding(
                5,
                "warning",
                f"window {window.id}: center bounds of {a.id} and {b.id} overlap, so the "
                "peaks may swap; set ordered: true if they are listed in center order",
            )


def _rule6(spec: PriorsSpec) -> Iterator[Finding]:
    peak_ids = {p.id for w in spec.windows for p in w.peaks}
    owners: dict[str, list[str]] = {}
    for member in spec.end_members:
        for peak_id in member.peaks:
            owners.setdefault(peak_id, []).append(member.id)
            if peak_id not in peak_ids:
                yield Finding(6, "error", f"end member {member.id}: unknown peak {peak_id!r}")
    for peak_id, members in owners.items():
        if len(members) > 1:
            yield Finding(
                6, "error", f"peak {peak_id} is in more than one end member: {', '.join(members)}"
            )


def _rule7(spec: PriorsSpec) -> Iterator[Finding]:
    pre = spec.preprocessing
    if pre.margin_cm1 < 0:
        yield Finding(7, "error", f"preprocessing.margin_cm1 {pre.margin_cm1} must be >= 0")
    if pre.despike is not None and pre.despike.kernel % 2 == 0:
        yield Finding(7, "error", f"preprocessing.despike.kernel {pre.despike.kernel} must be odd")
    smoothing = pre.smoothing
    if smoothing is None:
        return
    if smoothing.window % 2 == 0 or smoothing.window < MIN_SMOOTHING_WINDOW:
        yield Finding(
            7, "error", f"preprocessing.smoothing.window {smoothing.window} must be odd and >= 3"
        )
    if not 0 <= smoothing.polyorder < smoothing.window:
        yield Finding(
            7,
            "error",
            f"preprocessing.smoothing.polyorder {smoothing.polyorder} must be >= 0 and below "
            f"the window ({smoothing.window})",
        )
