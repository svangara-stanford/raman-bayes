"""``raman validate-priors``: exit codes, envelope contents and where text goes (§5, §10)."""

from pathlib import Path

import yaml
from typer.testing import CliRunner

from raman_bayes.cli.main import app
from raman_bayes.cli.output import CliEnvelope, ExitCode
from raman_bayes.domain.priors import PriorsSpec
from raman_bayes.domain.priors_rules import priors_hash

FIXTURE = Path(__file__).parents[1] / "fixtures" / "priors" / "design_example.yaml"
# User-visible text must not cite repo paths or internal docs; installed users have neither.
REPO_REFS = ("DESIGN.md", "§", "schemas/", "configs/", "docs/", "tests/", "Phase")

runner = CliRunner()


def _copy(tmp_path: Path, *replacements: tuple[str, str]) -> Path:
    text = FIXTURE.read_text(encoding="utf-8")
    for old, new in replacements:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "priors.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _json(path: Path) -> tuple[int, CliEnvelope, str]:
    result = runner.invoke(app, ["validate-priors", str(path), "--json"])
    (line,) = result.stdout.splitlines()
    return result.exit_code, CliEnvelope.model_validate_json(line), result.stderr


def test_cli_valid_file_exits_0_with_hash(tmp_path: Path) -> None:
    path = _copy(tmp_path)
    expected = priors_hash(PriorsSpec.model_validate(yaml.safe_load(FIXTURE.read_text("utf-8"))))
    code, env, stderr = _json(path)
    assert code == ExitCode.OK
    assert env.ok
    assert env.command == "validate-priors"
    assert env.data == {
        "priors_hash": expected,
        "name": "nmc_commercial",
        "version": "1.0.0",
        "status": "approved",
    }
    assert stderr == ""
    human = runner.invoke(app, ["validate-priors", str(path)])
    assert human.exit_code == 0
    assert expected in human.stdout
    assert human.stderr == ""


def test_cli_invalid_file_exits_2_listing_every_error(tmp_path: Path) -> None:
    path = _copy(tmp_path, ("margin_cm1: 25", "margin_cm1: -5"), ("kernel: 3", "kernel: 4"))
    code, env, _ = _json(path)
    assert code == ExitCode.VALIDATION
    assert not env.ok
    assert len(env.errors) == 2
    assert env.data is None
    human = runner.invoke(app, ["validate-priors", str(path)])
    assert human.exit_code == 2
    assert human.stdout == ""
    assert human.stderr.count("error: ") == 2


def test_cli_warnings_to_stderr_and_in_envelope(tmp_path: Path) -> None:
    path = _copy(tmp_path, ("bounds: [445, 485]", "bounds: [360, 485]"))
    code, env, json_stderr = _json(path)
    assert code == ExitCode.OK
    (warning,) = env.warnings
    assert "u1" in warning
    assert json_stderr == ""
    human = runner.invoke(app, ["validate-priors", str(path)])
    assert human.exit_code == 0
    assert f"warning: {warning}" in human.stderr
    assert "warning: " not in human.stdout


def test_cli_missing_file_exits_2(tmp_path: Path) -> None:
    code, env, _ = _json(tmp_path / "absent.yaml")
    assert code == ExitCode.VALIDATION
    (error,) = env.errors
    assert "absent.yaml" in error


def test_cli_yaml_syntax_error_exits_2_with_line(tmp_path: Path) -> None:
    path = _copy(tmp_path, ("  lineshape: pseudo_voigt\n", "  lineshape: [pseudo_voigt\n"))
    code, env, _ = _json(path)
    assert code == ExitCode.VALIDATION
    (error,) = env.errors
    assert "line " in error


def test_cli_messages_cite_no_repo_paths(tmp_path: Path) -> None:
    cases = [
        _copy(tmp_path / "ok"),
        _copy(tmp_path / "warn", ("bounds: [445, 485]", "bounds: [360, 485]")),
        _copy(tmp_path / "bad", ("margin_cm1: 25", "margin_cm1: -5"), ("p: 0.35", "p: x")),
        tmp_path / "absent.yaml",
    ]
    for path in cases:
        for argv in (["validate-priors", str(path)], ["validate-priors", str(path), "--json"]):
            result = runner.invoke(app, argv)
            for ref in REPO_REFS:
                assert ref not in result.output, (path, ref)
