"""CLI output layer (DESIGN.md §10): exit codes, the ``--json`` envelope, and the Rich console."""

import io
import json

import pytest
from pydantic import ValidationError

from raman_bayes.cli.output import CliEnvelope, ExitCode, emit, make_console

ESC = "\x1b"


class _TtyStream(io.StringIO):
    def isatty(self) -> bool:
        return True


def _envelope(code: ExitCode, **kwargs: object) -> CliEnvelope:
    return CliEnvelope.model_validate(
        {"command": "doctor", "exit_code": code, "ok": code is ExitCode.OK, **kwargs}
    )


# ---------------------------------------------------------------- exit codes


def test_exit_codes_are_section_10() -> None:
    assert {c.name: c.value for c in ExitCode} == {
        "OK": 0,
        "VALIDATION": 2,
        "NOTHING_TO_DO": 3,
        "BACKEND": 4,
    }


# ---------------------------------------------------------------- envelope


def test_envelope_defaults() -> None:
    env = _envelope(ExitCode.NOTHING_TO_DO)
    assert env.schema_version == "1.0"
    assert env.message is None
    assert env.data is None
    assert env.warnings == ()
    assert env.errors == ()


@pytest.mark.parametrize(
    ("code", "ok"),
    [(ExitCode.OK, False), (ExitCode.VALIDATION, True), (ExitCode.NOTHING_TO_DO, True)],
)
def test_envelope_ok_must_match_exit_code(code: ExitCode, ok: bool) -> None:
    with pytest.raises(ValidationError, match="ok"):
        CliEnvelope(command="doctor", exit_code=code, ok=ok)


def test_envelope_rejects_unknown_exit_code() -> None:
    with pytest.raises(ValidationError):
        CliEnvelope.model_validate({"command": "doctor", "exit_code": 1, "ok": False})


def test_envelope_is_frozen_and_strict() -> None:
    env = _envelope(ExitCode.OK)
    with pytest.raises(ValidationError):
        env.command = "plan"  # type: ignore[misc]
    with pytest.raises(ValidationError, match="extra"):
        CliEnvelope.model_validate({"command": "doctor", "exit_code": 0, "ok": True, "surprise": 1})


def test_envelope_json_round_trip() -> None:
    env = _envelope(
        ExitCode.VALIDATION,
        message="bad priors",
        data={"n_windows": 2},
        warnings=["near edge"],
        errors=["overlap"],
    )
    raw = env.model_dump_json()
    assert json.loads(raw)["exit_code"] == 2
    assert CliEnvelope.model_validate_json(raw) == env


# ---------------------------------------------------------------- emit


def test_emit_json_writes_one_line_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    env = _envelope(ExitCode.NOTHING_TO_DO, message="nothing")
    emit(env, as_json=True)
    out, err = capsys.readouterr()
    assert err == ""
    assert out.endswith("\n")
    assert out.count("\n") == 1
    assert CliEnvelope.model_validate_json(out) == env


@pytest.mark.parametrize("code", [ExitCode.OK, ExitCode.NOTHING_TO_DO])
def test_emit_human_success_goes_to_stdout(
    code: ExitCode, capsys: pytest.CaptureFixture[str]
) -> None:
    emit(_envelope(code, message="all good", warnings=["careful"]), as_json=False)
    out, err = capsys.readouterr()
    assert "all good" in out
    assert "warning: careful" in out
    assert err == ""


@pytest.mark.parametrize("code", [ExitCode.VALIDATION, ExitCode.BACKEND])
def test_emit_human_failure_goes_to_stderr(
    code: ExitCode, capsys: pytest.CaptureFixture[str]
) -> None:
    emit(_envelope(code, message="it broke", errors=["details"]), as_json=False)
    out, err = capsys.readouterr()
    assert out == ""
    assert "it broke" in err
    assert "error: details" in err


def test_emit_human_prints_markup_verbatim(capsys: pytest.CaptureFixture[str]) -> None:
    emit(_envelope(ExitCode.OK, message="path [bold]x[/] kept"), as_json=False)
    assert "path [bold]x[/] kept" in capsys.readouterr().out


def test_emit_human_does_not_wrap_long_lines(capsys: pytest.CaptureFixture[str]) -> None:
    message = "x" * 300
    emit(_envelope(ExitCode.OK, message=message), as_json=False)
    assert capsys.readouterr().out == message + "\n"


# ---------------------------------------------------------------- console


def test_console_has_no_ansi_off_a_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    stream = io.StringIO()
    make_console(stream).print("styled", style="bold yellow")
    assert stream.getvalue() == "styled\n"


def test_console_uses_colour_on_a_tty() -> None:
    stream = _TtyStream()
    make_console(stream).print("styled", style="bold yellow")
    assert ESC in stream.getvalue()
