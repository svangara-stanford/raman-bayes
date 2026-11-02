"""``app.validate.validate_priors_file``: YAML loading, structure errors and the §5 rules."""

from pathlib import Path

from raman_bayes.app.validate import validate_priors_file
from raman_bayes.domain.priors import PriorsSpec
from raman_bayes.domain.priors_rules import priors_hash

FIXTURE = Path(__file__).parents[1] / "fixtures" / "priors" / "design_example.yaml"


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "priors.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _edited(tmp_path: Path, *replacements: tuple[str, str]) -> Path:
    text = FIXTURE.read_text(encoding="utf-8")
    for old, new in replacements:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    return _write(tmp_path, text)


def test_valid_file_returns_spec_and_hash(tmp_path: Path) -> None:
    result = validate_priors_file(_edited(tmp_path))
    assert result.errors == ()
    assert result.warnings == ()
    assert result.spec is not None
    assert result.priors_hash == priors_hash(result.spec)


def test_all_errors_reported_not_only_the_first(tmp_path: Path) -> None:
    path = _edited(
        tmp_path,
        ("margin_cm1: 25", "margin_cm1: -5"),
        ("kernel: 3", "kernel: 4"),
        ("peaks: [d, g]", "peaks: [d, g, x9]"),
    )
    result = validate_priors_file(path)
    assert len(result.errors) == 3
    joined = "\n".join(result.errors)
    for fragment in ("margin_cm1", "kernel", "x9"):
        assert fragment in joined
    assert result.spec is None
    assert result.priors_hash is None

    structural = validate_priors_file(
        _edited(tmp_path, ("p: 0.35", "p: auto"), ("nu: 5", "nu: auto"))
    )
    assert len(structural.errors) == 2


def test_auto_rejected_through_validate_file(tmp_path: Path) -> None:
    path = _edited(tmp_path, ("center: {mean: 465,", "center: {mean: auto,"))
    (error,) = validate_priors_file(path).errors
    assert "windows[0].peaks[0].center.mean" in error
    assert "'auto'" in error


def test_auto_rejected_as_sd_rule_through_validate_file(tmp_path: Path) -> None:
    path = _edited(tmp_path, ("sd_rule: quarter_range", "sd_rule: auto"))
    (error,) = validate_priors_file(path).errors
    assert "defaults.center.sd_rule" in error
    assert "'auto'" in error


def test_derived_rejected_through_validate_file(tmp_path: Path) -> None:
    text = FIXTURE.read_text(encoding="utf-8") + "\nderived:\n  ratio: u1.area / u2.area\n"
    (error,) = validate_priors_file(_write(tmp_path, text)).errors
    assert error.startswith("derived:")
    assert "not allowed" in error


def test_unsafe_yaml_tag_rejected(tmp_path: Path) -> None:
    marker = tmp_path / "pwned"
    text = f'schema_version: !!python/object/apply:os.system ["touch {marker}"]\n'
    result = validate_priors_file(_write(tmp_path, text))
    (error,) = result.errors
    assert "line 1" in error
    assert "python/object/apply" in error
    assert not marker.exists()
    assert result.spec is None


def test_warnings_do_not_block_a_valid_result(tmp_path: Path) -> None:
    path = _edited(tmp_path, ("bounds: [445, 485]", "bounds: [360, 485]"))
    result = validate_priors_file(path)
    assert result.errors == ()
    (warning,) = result.warnings
    assert "u1" in warning
    assert isinstance(result.spec, PriorsSpec)


def test_unreadable_and_non_mapping_files(tmp_path: Path) -> None:
    (missing,) = validate_priors_file(tmp_path / "absent.yaml").errors
    assert "absent.yaml" in missing
    (listing,) = validate_priors_file(_write(tmp_path, "- just\n- a list\n")).errors
    assert "mapping" in listing
    (empty,) = validate_priors_file(_write(tmp_path, "")).errors
    assert "mapping" in empty
