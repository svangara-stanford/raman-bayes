"""``nmc_commercial.yaml``: the §5 recipe, with windows from the legacy CSVs (§14)."""

from pathlib import Path

import yaml

from raman_bayes.app.validate import validate_priors_file
from raman_bayes.domain.priors import PriorsSpec
from raman_bayes.domain.priors_rules import priors_hash
from raman_bayes.io.legacy_csv import read_legacy_csv

ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "configs" / "priors" / "nmc_commercial.yaml"
LEGACY = ROOT / "configs" / "priors" / "legacy"
FIXTURE = ROOT / "tests" / "fixtures" / "priors" / "design_example.yaml"


def _recipe() -> PriorsSpec:
    return PriorsSpec.model_validate(yaml.safe_load(RECIPE.read_text(encoding="utf-8")))


def test_nmc_commercial_is_valid_and_approved() -> None:
    result = validate_priors_file(RECIPE)
    assert result.errors == ()
    assert result.warnings == ()
    assert result.spec is not None
    assert result.spec.spec.status == "approved"


def test_nmc_commercial_equals_section5_example() -> None:
    example = PriorsSpec.model_validate(yaml.safe_load(FIXTURE.read_text(encoding="utf-8")))
    assert _recipe() == example
    assert priors_hash(_recipe()) == priors_hash(example)


def test_nmc_commercial_peaks_match_legacy_csvs() -> None:
    windows = {w.id: w for w in _recipe().windows}
    for window_id, name in (("lithiation", "nmc_lithiation.csv"), ("carbon", "nmc_carbon.csv")):
        recipe = windows[window_id]
        imported = read_legacy_csv(LEGACY / name, window_id, recipe.range_cm1)
        assert len(recipe.peaks) == len(imported.peaks)
        for ours, theirs in zip(recipe.peaks, imported.peaks, strict=True):
            assert ours.id == theirs.id
            assert ours.center == theirs.center
            assert ours.fwhm == theirs.fwhm
