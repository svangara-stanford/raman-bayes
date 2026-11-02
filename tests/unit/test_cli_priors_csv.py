"""``raman priors import-csv`` and ``export-csv`` (§5): output, envelopes and exit codes."""

from pathlib import Path

import yaml
from typer.testing import CliRunner

from raman_bayes.cli.main import app
from raman_bayes.cli.output import CliEnvelope, ExitCode
from raman_bayes.domain.priors import Window

ROOT = Path(__file__).resolve().parents[2]
CARBON = ROOT / "configs" / "priors" / "legacy" / "nmc_carbon.csv"
RECIPE = ROOT / "configs" / "priors" / "nmc_commercial.yaml"
HEADER_LINE = ",x (center),x (lower limit),x (upper limit),FWHM (lower limit),FWHM (upper limit)"

runner = CliRunner()


def _import_argv(lo: str = "1150", hi: str = "1750") -> list[str]:
    return ["priors", "import-csv", str(CARBON), "--window-id", "carbon", "--range", lo, hi]


def _envelope(argv: list[str]) -> tuple[int, CliEnvelope]:
    result = runner.invoke(app, [*argv, "--json"])
    (line,) = result.stdout.splitlines()
    return result.exit_code, CliEnvelope.model_validate_json(line)


def test_cli_import_csv_prints_fragment() -> None:
    result = runner.invoke(app, _import_argv())
    assert result.exit_code == 0, result.output
    (window,) = (Window.model_validate(w) for w in yaml.safe_load(result.stdout))
    assert window.id == "carbon"
    assert window.range_cm1 == (1150.0, 1750.0)
    assert [(p.id, p.label) for p in window.peaks] == [("d", "D"), ("g", "G")]
    assert "allow_overlap" not in result.stdout


def test_cli_import_csv_json() -> None:
    code, env = _envelope(_import_argv())
    assert code == ExitCode.OK
    assert env.data is not None
    window = Window.model_validate(env.data["window"])
    assert [p.id for p in window.peaks] == ["d", "g"]
    assert yaml.safe_load(env.data["fragment"]) == [
        window.model_dump(mode="json", exclude_defaults=True)
    ]


def test_cli_import_csv_bad_range_exits_2() -> None:
    for lo, hi in (("1750", "1150"), ("1400", "1750")):
        code, env = _envelope(_import_argv(lo, hi))
        assert code == ExitCode.VALIDATION, (lo, hi)
        assert env.errors
    human = runner.invoke(app, _import_argv("1750", "1150"))
    assert human.exit_code == 2
    assert human.stdout == ""
    assert "range_cm1" in human.stderr


def test_cli_import_csv_missing_file_exits_2(tmp_path: Path) -> None:
    argv = ["priors", "import-csv", str(tmp_path / "absent.csv"), "--window-id", "w"]
    code, env = _envelope([*argv, "--range", "0", "10"])
    assert code == ExitCode.VALIDATION
    assert "absent.csv" in env.errors[0]


def _export_argv(window_id: str = "carbon") -> list[str]:
    return ["priors", "export-csv", str(RECIPE), "--window-id", window_id]


def test_cli_export_csv_prints_csv() -> None:
    result = runner.invoke(app, _export_argv())
    assert result.exit_code == 0, result.output
    assert result.stdout == (
        f"{HEADER_LINE}\nd,1335.0,1310.0,1370.0,90.0,220.0\ng,1584.0,1548.0,1620.0,35.0,125.0\n"
    )
    assert result.stderr == ""


def test_cli_export_csv_json() -> None:
    code, env = _envelope(_export_argv("lithiation"))
    assert code == ExitCode.OK
    assert env.data is not None
    assert env.data["window_id"] == "lithiation"
    assert env.data["csv"].splitlines() == [
        HEADER_LINE,
        "u1,465.0,445.0,485.0,35.0,70.0",
        "u2,568.0,550.0,585.0,30.0,60.0",
    ]


def test_cli_export_csv_unknown_window_exits_2() -> None:
    code, env = _envelope(_export_argv("raman"))
    assert code == ExitCode.VALIDATION
    (error,) = env.errors
    assert "'raman'" in error
    assert "lithiation" in error
    assert "carbon" in error


def test_cli_export_csv_invalid_priors_exits_2(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(RECIPE.read_text(encoding="utf-8").replace("margin_cm1: 25", "margin_cm1: -1"))
    code, env = _envelope(["priors", "export-csv", str(bad), "--window-id", "carbon"])
    assert code == ExitCode.VALIDATION
    assert "margin_cm1" in env.errors[0]
