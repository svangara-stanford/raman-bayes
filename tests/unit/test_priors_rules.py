"""PriorsSpec rules 1-7 and 9 (DESIGN.md §5) on the §5 example and edited copies."""

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from hypothesis import given
from hypothesis import strategies as st

from raman_bayes.domain.priors import PriorsSpec, center_sd
from raman_bayes.domain.priors_rules import RuleReport, check_rules, priors_hash

FIXTURE = Path(__file__).parents[1] / "fixtures" / "priors" / "design_example.yaml"
RAW: dict[str, Any] = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
SAVGOL = {"method": "savgol", "window": 11, "polyorder": 3}


def _raw() -> dict[str, Any]:
    return copy.deepcopy(RAW)


def _report(raw: dict[str, Any]) -> RuleReport:
    return check_rules(PriorsSpec.model_validate(raw))


def _kinds(report: RuleReport) -> set[tuple[int, str]]:
    return {(f.rule, f.severity) for f in report.findings}


def _u1(raw: dict[str, Any]) -> dict[str, Any]:
    peak: dict[str, Any] = raw["windows"][0]["peaks"][0]
    return peak


def test_design_example_is_valid_without_warnings() -> None:
    assert _report(_raw()).findings == ()


# ---------------------------------------------------------------- rule 1


def test_rule1_unknown_schema_version_rejected() -> None:
    raw = _raw()
    raw["schema_version"] = "2.0"
    report = _report(raw)
    assert _kinds(report) == {(1, "error")}
    assert "2.0" in report.errors[0]


@pytest.mark.parametrize(
    ("where", "value", "valid"),
    [
        ("window", "Lithiation", False),
        ("window", "_lithiation", False),
        ("window", "lithiation\n", False),
        ("peak", "1u", False),
        ("peak", "u-1", False),
        ("end_member", "carbon black", False),
        ("end_member", "a" * 33, False),
        ("window", "a" * 32, True),
        ("end_member", "c9_x", True),
    ],
)
def test_rule1_id_pattern(where: str, value: str, valid: bool) -> None:
    raw = _raw()
    if where == "window":
        raw["windows"][0]["id"] = value
    elif where == "peak":
        _u1(raw)["id"] = value
        raw["end_members"][0]["peaks"] = [value, "u2"]
    else:
        raw["end_members"][1]["id"] = value
    report = _report(raw)
    if valid:
        assert report.findings == ()
    else:
        assert _kinds(report) == {(1, "error")}
        assert repr(value) in report.errors[0]


def test_rule1_ids_unique_across_file() -> None:
    raw = _raw()
    raw["windows"][1]["peaks"][0]["id"] = "u1"  # a peak id reused in another window
    raw["end_members"][1]["peaks"] = ["u1", "g"]
    raw["end_members"][0]["peaks"] = ["u2"]
    report = _report(raw)
    assert _kinds(report) == {(1, "error")}
    assert "'u1'" in report.errors[0]

    raw = _raw()
    raw["windows"][1]["id"] = "lithiation"
    assert _kinds(_report(raw)) >= {(1, "error")}

    raw = _raw()
    raw["end_members"][1]["id"] = "lithiated_oxide"
    assert _kinds(_report(raw)) == {(1, "error")}


def test_rule1_same_id_for_window_and_end_member_allowed() -> None:
    # The §5 example itself names both a window and an end member "carbon".
    assert {w["id"] for w in RAW["windows"]} & {m["id"] for m in RAW["end_members"]} == {"carbon"}
    assert _report(_raw()).findings == ()


# ---------------------------------------------------------------- rule 2


def test_rule2_center_bounds_outside_window_rejected() -> None:
    for bounds in ([340, 485], [445, 710]):
        raw = _raw()
        _u1(raw)["center"]["bounds"] = bounds
        report = _report(raw)
        assert (2, "error") in _kinds(report), bounds
        assert "u1" in report.errors[0]


def test_rule2_center_bound_near_window_edge_warns() -> None:
    raw = _raw()
    _u1(raw)["center"]["bounds"] = [360, 485]  # 10 cm^-1 from the edge; fwhm lower is 35
    report = _report(raw)
    assert _kinds(report) == {(2, "warning")}
    assert report.errors == ()
    assert "u1" in report.warnings[0]


# ---------------------------------------------------------------- rule 3

coordinate = st.floats(min_value=-1e4, max_value=1e4)
gap = st.floats(min_value=1e-3, max_value=1e3)


@given(lower=coordinate, shortfall=st.floats(0, 1e3), target=st.sampled_from(["center", "fwhm"]))
def test_rule3_lower_not_below_upper_rejected(lower: float, shortfall: float, target: str) -> None:
    raw = _raw()
    _u1(raw)[target]["bounds"] = [lower, lower - shortfall]
    assert (3, "error") in _kinds(_report(raw))


@given(delta=gap, below=st.booleans())
def test_rule3_center_mean_outside_bounds_rejected(delta: float, below: bool) -> None:
    raw = _raw()
    _u1(raw)["center"]["mean"] = 445 - delta if below else 485 + delta
    assert _kinds(_report(raw)) == {(3, "error")}


@given(lower=st.floats(min_value=-1e3, max_value=0), width=gap)
def test_rule3_non_positive_fwhm_bounds_rejected(lower: float, width: float) -> None:
    raw = _raw()
    _u1(raw)["fwhm"]["bounds"] = [lower, min(lower + width, 350)]
    assert (3, "error") in _kinds(_report(raw))


def test_rule3_fwhm_upper_above_window_width_rejected() -> None:
    raw = _raw()
    _u1(raw)["fwhm"]["bounds"] = [35, 350]  # lithiation is 350 cm^-1 wide
    assert _report(raw).findings == ()
    _u1(raw)["fwhm"]["bounds"] = [35, 350.5]
    report = _report(raw)
    assert _kinds(report) == {(3, "error")}
    assert "350" in report.errors[0]


@st.composite
def single_peak_specs(draw: st.DrawFn) -> dict[str, Any]:
    # Margins of 2x the thresholds keep float rounding from landing a bound on a rule's edge.
    low = draw(st.floats(0, 3000))
    width = draw(st.floats(100, 1000))
    fwhm_lower = draw(st.floats(1, 20))
    fwhm_upper = draw(st.floats(fwhm_lower + 1, width / 2))
    center_lower = draw(st.floats(low + 2 * fwhm_lower, low + width / 2 - 1))
    center_upper = draw(st.floats(center_lower + 1, low + width - 2 * fwhm_lower))
    mean = draw(st.floats(center_lower, center_upper))
    raw = _raw()
    raw["end_members"] = []
    raw["windows"] = [
        {
            "id": "w",
            "range_cm1": [low, low + width],
            "peaks": [
                {
                    "id": "p",
                    "center": {"mean": mean, "bounds": [center_lower, center_upper]},
                    "fwhm": {"bounds": [fwhm_lower, fwhm_upper]},
                }
            ],
        }
    ]
    return raw


@given(single_peak_specs())
def test_rule3_valid_bounds_accepted(raw: dict[str, Any]) -> None:
    assert _report(raw).findings == ()


# ---------------------------------------------------------------- rules 4-7


@pytest.mark.parametrize(
    ("flags", "error"),
    [((False, False), True), ((True, False), True), ((False, True), True), ((True, True), False)],
)
def test_rule4_overlapping_windows(flags: tuple[bool, bool], error: bool) -> None:
    raw = _raw()
    raw["windows"][1]["range_cm1"] = [650, 1750]
    for window, flag in zip(raw["windows"], flags, strict=True):
        window["allow_overlap"] = flag
    report = _report(raw)
    assert _kinds(report) == ({(4, "error")} if error else set())
    if error:
        assert "lithiation" in report.errors[0]
        assert "carbon" in report.errors[0]


def test_rule4_touching_windows_do_not_overlap() -> None:
    raw = _raw()
    raw["windows"][1]["range_cm1"] = [700, 1750]
    assert _report(raw).findings == ()


def test_rule5_overlapping_centers_warn_unless_ordered() -> None:
    raw = _raw()
    raw["windows"][0]["peaks"][1]["center"]["bounds"] = [480, 585]
    report = _report(raw)
    assert _kinds(report) == {(5, "warning")}
    assert "u1" in report.warnings[0]
    assert "u2" in report.warnings[0]
    raw["windows"][0]["ordered"] = True
    assert _report(raw).findings == ()


def test_rule6_unknown_end_member_peak_rejected() -> None:
    raw = _raw()
    raw["end_members"][1]["peaks"] = ["d", "g", "x9"]
    report = _report(raw)
    assert _kinds(report) == {(6, "error")}
    assert "x9" in report.errors[0]


def test_rule6_peak_in_two_end_members_rejected() -> None:
    raw = _raw()
    raw["end_members"][1]["peaks"] = ["d", "g", "u1"]
    report = _report(raw)
    assert _kinds(report) == {(6, "error")}
    assert "u1" in report.errors[0]


def test_rule7_negative_margin_rejected() -> None:
    raw = _raw()
    raw["preprocessing"]["margin_cm1"] = 0
    assert _report(raw).findings == ()
    raw["preprocessing"]["margin_cm1"] = -1
    assert _kinds(_report(raw)) == {(7, "error")}


def test_rule7_even_despike_kernel_rejected() -> None:
    raw = _raw()
    raw["preprocessing"]["despike"]["kernel"] = 4
    report = _report(raw)
    assert _kinds(report) == {(7, "error")}
    assert "kernel" in report.errors[0]
    raw["preprocessing"]["despike"] = None
    assert _report(raw).findings == ()


@pytest.mark.parametrize(
    "smoothing",
    [
        {**SAVGOL, "window": 10},
        {**SAVGOL, "window": 1, "polyorder": 0},
        {**SAVGOL, "polyorder": 11},
        {**SAVGOL, "polyorder": -1},
    ],
    ids=["even", "below-3", "polyorder-ge-window", "negative-polyorder"],
)
def test_rule7_smoothing_window_and_polyorder(smoothing: dict[str, Any]) -> None:
    raw = _raw()
    raw["preprocessing"]["smoothing"] = SAVGOL
    assert _report(raw).findings == ()
    raw["preprocessing"]["smoothing"] = smoothing
    assert _kinds(_report(raw)) == {(7, "error")}


# ---------------------------------------------------------------- named rules and hash


def test_quarter_range_sd_and_explicit_sd_wins() -> None:
    spec = PriorsSpec.model_validate(_raw())
    u1 = spec.windows[0].peaks[0]
    assert center_sd(u1, spec.defaults.center.sd_rule) == (485 - 445) / 4
    raw = _raw()
    _u1(raw)["center"]["sd"] = 7.0
    explicit = PriorsSpec.model_validate(raw).windows[0].peaks[0]
    assert center_sd(explicit, "quarter_range") == 7.0


def test_priors_hash_is_sha256_of_canonical_json() -> None:
    spec = PriorsSpec.model_validate(_raw())
    canonical = json.dumps(
        spec.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    assert priors_hash(spec) == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert len(priors_hash(spec)) == 64


def test_priors_hash_ignores_yaml_formatting_and_key_order() -> None:
    reference = priors_hash(PriorsSpec.model_validate(_raw()))
    reordered = {key: RAW[key] for key in reversed(list(RAW))}
    block_style = yaml.safe_dump(reordered, default_flow_style=False, allow_unicode=True)
    reparsed = PriorsSpec.model_validate(yaml.safe_load("# a comment\n" + block_style))
    assert priors_hash(reparsed) == reference
    raw = _raw()
    _u1(raw)["center"]["mean"] = 465.0  # was the integer 465
    raw["windows"][0]["allow_overlap"] = False  # the default, now explicit
    assert priors_hash(PriorsSpec.model_validate(raw)) == reference


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("spec", "version"), "1.0.1"),
        (("spec", "status"), "draft"),
        (("spec", "source"), None),
        (("defaults", "presence", "p"), 0.36),
        (("preprocessing", "margin_cm1"), 26),
        (("preprocessing", "smoothing"), SAVGOL),
        (("end_members", 0, "label"), "NMC"),
        (("windows", 0, "peaks", 0, "label"), "nu1"),
        (("windows", 0, "peaks", 0, "center", "bounds"), [445, 486]),
        (("windows", 1, "ordered"), True),
    ],
    ids=lambda v: "/".join(map(str, v)) if isinstance(v, tuple) else "",
)
def test_priors_hash_changes_with_any_value(path: tuple[str | int, ...], value: Any) -> None:
    raw = _raw()
    node: Any = raw
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    reference = priors_hash(PriorsSpec.model_validate(_raw()))
    assert priors_hash(PriorsSpec.model_validate(raw)) != reference
