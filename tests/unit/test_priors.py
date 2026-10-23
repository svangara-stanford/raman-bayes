"""PriorsSpec models (DESIGN.md §5): structure and key names only; rules 1-9 are Phase 1."""

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import BaseModel, ValidationError

from raman_bayes.domain import priors
from raman_bayes.domain._base import DomainModel
from raman_bayes.domain.priors import (
    BaselineSpec,
    CenterDefault,
    CenterPrior,
    Defaults,
    DespikeSpec,
    EndMember,
    FwhmDefault,
    FwhmPrior,
    LikelihoodSpec,
    Peak,
    PreprocessingSpec,
    PresenceDefault,
    PriorsSpec,
    SmoothingSpec,
    SpecMeta,
    Window,
)

FIXTURE = Path(__file__).parents[1] / "fixtures" / "priors" / "design_example.yaml"
DESIGN_MD = Path(__file__).parents[2] / "docs" / "DESIGN.md"

# Key names copied by hand from the §5 YAML and text; never generated from the code under test.
SECTION5_KEYS: list[tuple[type[BaseModel], set[str]]] = [
    (
        PriorsSpec,
        {"schema_version", "spec", "defaults", "preprocessing", "end_members", "windows"},
    ),
    (SpecMeta, {"name", "version", "status", "source"}),
    (Defaults, {"presence", "lineshape", "center", "fwhm", "likelihood"}),
    (PresenceDefault, {"prior", "p"}),
    (CenterDefault, {"prior", "sd_rule"}),
    (FwhmDefault, {"prior"}),
    (LikelihoodSpec, {"model", "nu"}),
    (
        PreprocessingSpec,
        {"margin_cm1", "despike", "baseline", "smoothing", "clamp_nonnegative"},
    ),
    (DespikeSpec, {"method", "kernel", "thresh"}),
    (BaselineSpec, {"method", "lam", "p"}),
    (SmoothingSpec, {"method", "window", "polyorder"}),
    (EndMember, {"id", "label", "peaks"}),
    (Window, {"id", "range_cm1", "peaks", "allow_overlap", "ordered"}),
    (Peak, {"id", "label", "center", "fwhm"}),
    (CenterPrior, {"mean", "bounds", "sd"}),
    (FwhmPrior, {"bounds"}),
]

# Every mapping in the §5 example, addressed by its path from the document root.
NESTED_PATHS: list[tuple[str | int, ...]] = [
    (),
    ("spec",),
    ("defaults",),
    ("defaults", "presence"),
    ("defaults", "center"),
    ("defaults", "fwhm"),
    ("defaults", "likelihood"),
    ("preprocessing",),
    ("preprocessing", "despike"),
    ("preprocessing", "baseline"),
    ("end_members", 0),
    ("windows", 0),
    ("windows", 0, "peaks", 0),
    ("windows", 0, "peaks", 0, "center"),
    ("windows", 0, "peaks", 0, "fwhm"),
]

SAVGOL = {"method": "savgol", "window": 11, "polyorder": 3}


@pytest.fixture
def raw() -> dict[str, Any]:
    data = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


@pytest.fixture
def spec(raw: dict[str, Any]) -> PriorsSpec:
    return PriorsSpec.model_validate(raw)


def _at(data: Any, path: tuple[str | int, ...]) -> Any:
    for key in path:
        data = data[key]
    return data


def test_fixture_matches_design_md_example() -> None:
    design = DESIGN_MD.read_text(encoding="utf-8")
    block = design.split("# configs/priors/nmc_commercial.yaml\n", 1)[1].split("```", 1)[0]
    fixture_body = FIXTURE.read_text(encoding="utf-8").split("\n", 1)[1]
    assert fixture_body == block


def test_design_example_parses(spec: PriorsSpec) -> None:
    assert spec.schema_version == "1.0"
    assert spec.spec.name == "nmc_commercial"
    assert spec.spec.version == "1.0.0"
    assert spec.spec.status == "approved"
    assert spec.defaults.presence.p == 0.35
    assert spec.defaults.center.sd_rule == "quarter_range"
    assert spec.defaults.likelihood.nu == 5
    assert spec.preprocessing.margin_cm1 == 25
    assert spec.preprocessing.despike is not None
    assert spec.preprocessing.despike.kernel == 3
    assert spec.preprocessing.baseline is not None
    assert spec.preprocessing.baseline.lam == 1.0e5
    assert spec.preprocessing.smoothing is None
    assert spec.preprocessing.clamp_nonnegative is True
    assert [m.peaks for m in spec.end_members] == [("u1", "u2"), ("d", "g")]
    assert [w.id for w in spec.windows] == ["lithiation", "carbon"]
    lithiation = spec.windows[0]
    assert lithiation.range_cm1 == (350.0, 700.0)
    assert lithiation.allow_overlap is False
    assert lithiation.ordered is False
    u1 = lithiation.peaks[0]
    assert u1.label == "\N{GREEK SMALL LETTER NU}1"
    assert u1.center.mean == 465.0
    assert u1.center.bounds == (445.0, 485.0)
    assert u1.center.sd is None
    assert u1.fwhm.bounds == (35.0, 70.0)


def test_json_round_trip(spec: PriorsSpec) -> None:
    assert PriorsSpec.model_validate_json(spec.model_dump_json()) == spec


def test_yaml_round_trip(spec: PriorsSpec) -> None:
    text = yaml.safe_dump(spec.model_dump(mode="json"), allow_unicode=True, sort_keys=False)
    assert PriorsSpec.model_validate(yaml.safe_load(text)) == spec


def test_spec_is_hashable(spec: PriorsSpec) -> None:
    assert hash(spec) == hash(PriorsSpec.model_validate(spec.model_dump()))


@pytest.mark.parametrize(("model", "keys"), SECTION5_KEYS, ids=lambda v: getattr(v, "__name__", ""))
def test_key_names_match_section5(model: type[BaseModel], keys: set[str]) -> None:
    assert set(model.model_fields) == keys


def test_every_model_is_frozen_and_forbids_extra() -> None:
    models = [
        obj
        for obj in vars(priors).values()
        if isinstance(obj, type) and issubclass(obj, DomainModel) and obj is not DomainModel
    ]
    assert {m for m, _ in SECTION5_KEYS} == set(models)
    for model in models:
        assert model.model_config.get("frozen") is True, model.__name__
        assert model.model_config.get("extra") == "forbid", model.__name__


def test_assignment_raises(spec: PriorsSpec) -> None:
    with pytest.raises(ValidationError, match="frozen"):
        spec.spec.name = "other"  # type: ignore[misc]
    with pytest.raises(ValidationError, match="frozen"):
        spec.windows[0].peaks[0].center.mean = 470.0  # type: ignore[misc]


@pytest.mark.parametrize("path", NESTED_PATHS, ids=lambda p: "/".join(map(str, p)) or "root")
def test_unknown_key_rejected_at_every_level(
    raw: dict[str, Any], path: tuple[str | int, ...]
) -> None:
    bad = copy.deepcopy(raw)
    _at(bad, path)["unexpected"] = 1
    with pytest.raises(ValidationError) as exc:
        PriorsSpec.model_validate(bad)
    (error,) = exc.value.errors()
    assert error["type"] == "extra_forbidden"
    assert error["loc"] == (*path, "unexpected")


def test_derived_section_rejected(raw: dict[str, Any]) -> None:
    raw["derived"] = {"ratio": "u1.area / u2.area"}
    with pytest.raises(ValidationError) as exc:
        PriorsSpec.model_validate(raw)
    (error,) = exc.value.errors()
    assert error["type"] == "extra_forbidden"
    assert error["loc"] == ("derived",)


@pytest.mark.parametrize(
    "path",
    [
        ("windows", 0, "peaks", 0, "center", "mean"),
        ("preprocessing", "margin_cm1"),
        ("defaults", "presence", "p"),
    ],
    ids=lambda p: "/".join(map(str, p)),
)
def test_auto_rejected_for_numbers(raw: dict[str, Any], path: tuple[str | int, ...]) -> None:
    _at(raw, path[:-1])[path[-1]] = "auto"
    with pytest.raises(ValidationError) as exc:
        PriorsSpec.model_validate(raw)
    assert exc.value.errors()[0]["loc"] == path


def test_auto_rejected_as_sd_rule(raw: dict[str, Any]) -> None:
    raw["defaults"]["center"]["sd_rule"] = "auto"
    with pytest.raises(ValidationError):
        PriorsSpec.model_validate(raw)


def test_unknown_status_rejected(raw: dict[str, Any]) -> None:
    raw["spec"]["status"] = "final"
    with pytest.raises(ValidationError):
        PriorsSpec.model_validate(raw)


def test_smoothing_placeholder_parses(raw: dict[str, Any]) -> None:
    raw["preprocessing"]["smoothing"] = dict(SAVGOL)
    smoothing = PriorsSpec.model_validate(raw).preprocessing.smoothing
    assert smoothing == SmoothingSpec(method="savgol", window=11, polyorder=3)


def test_smoothing_rejects_unknown_key(raw: dict[str, Any]) -> None:
    raw["preprocessing"]["smoothing"] = {**SAVGOL, "deriv": 1}
    with pytest.raises(ValidationError) as exc:
        PriorsSpec.model_validate(raw)
    assert exc.value.errors()[0]["loc"] == ("preprocessing", "smoothing", "deriv")


def test_smoothing_spec_is_marked_provisional() -> None:
    assert SmoothingSpec.__doc__ is not None
    assert "provisional" in SmoothingSpec.__doc__.lower()
