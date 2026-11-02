"""The file-level priors rules (DESIGN.md §5 rules 1-7) and the priors hash (rule 9).

Rule 8 needs a dataset axis and runs at plan time. Every rule reports every violation it finds,
so a user fixes a file in one pass. Ids are unique within their kind (windows, peaks across all
windows, end members): the §5 example names both a window and an end member ``carbon``.
"""

import hashlib
import itertools
import json
import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

from raman_bayes.domain.priors import Bounds, Peak, PriorsSpec, Window

__all__ = [
    "ID_PATTERN",
    "KNOWN_SCHEMA_VERSIONS",
    "MIN_SMOOTHING_WINDOW",
    "Finding",
    "RuleReport",
    "Severity",
    "canonical_json",
    "check_rules",
    "priors_hash",
]

KNOWN_SCHEMA_VERSIONS = frozenset({"1.0"})
MIN_SMOOTHING_WINDOW = 3
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
    """Apply rules 1-7 to ``spec`` and return every finding; never raises for a bad recipe."""
    checks = (_rule1, _rule2, _rule3, _rule4, _rule5, _rule6, _rule7)
    return RuleReport(tuple(itertools.chain.from_iterable(check(spec) for check in checks)))


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


def _peaks(spec: PriorsSpec) -> Iterator[tuple[Window, Peak]]:
    return ((window, peak) for window in spec.windows for peak in window.peaks)


def _overlap(a: Bounds, b: Bounds) -> bool:
    # Open intervals: windows or bounds that only touch do not overlap.
    return a[0] < b[1] and b[0] < a[1]


def _rule1(spec: PriorsSpec) -> Iterator[Finding]:
    if spec.schema_version not in KNOWN_SCHEMA_VERSIONS:
        known = ", ".join(sorted(KNOWN_SCHEMA_VERSIONS))
        yield Finding(1, "error", f"schema_version {spec.schema_version!r} is not one of: {known}")
    kinds = {
        "window": [w.id for w in spec.windows],
        "peak": [p.id for _, p in _peaks(spec)],
        "end member": [m.id for m in spec.end_members],
    }
    for kind, ids in kinds.items():
        for id_ in ids:
            if ID_PATTERN.fullmatch(id_) is None:
                yield Finding(
                    1,
                    "error",
                    f"{kind} id {id_!r} must be a lowercase letter followed by up to 31 "
                    "lowercase letters, digits or underscores",
                )
        for id_, count in Counter(ids).items():
            if count > 1:
                yield Finding(1, "error", f"{kind} id {id_!r} is used {count} times")


def _rule2(spec: PriorsSpec) -> Iterator[Finding]:
    for window, peak in _peaks(spec):
        (lo, hi), (c_lo, c_hi) = window.range_cm1, peak.center.bounds
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


def _rule3(spec: PriorsSpec) -> Iterator[Finding]:
    for window in spec.windows:
        if not window.range_cm1[0] < window.range_cm1[1]:
            yield Finding(3, "error", f"window {window.id}: range_cm1 lower must be below upper")
    for window, peak in _peaks(spec):
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


def _rule5(spec: PriorsSpec) -> Iterator[Finding]:
    for window in spec.windows:
        if window.ordered:
            continue
        for a, b in itertools.combinations(window.peaks, 2):
            if _overlap(a.center.bounds, b.center.bounds):
                yield Finding(
                    5,
                    "warning",
                    f"window {window.id}: center bounds of {a.id} and {b.id} overlap, so the "
                    "peaks may swap; set ordered: true if they are listed in center order",
                )


def _rule6(spec: PriorsSpec) -> Iterator[Finding]:
    peak_ids = {p.id for _, p in _peaks(spec)}
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
