"""Exit codes, the ``--json`` envelope and the Rich console every command writes through (§10).

``CliEnvelope`` is provisional: DESIGN.md §10 only says ``--json`` output validates against
``schemas/cli/``. Regenerate the schema with ``pixi run gen-schemas`` after changing it.
"""

import sys
from enum import IntEnum
from typing import Any, Literal, Self, TextIO

import typer
from pydantic import BaseModel, ConfigDict, model_validator
from rich.console import Console

ENVELOPE_SCHEMA_VERSION: Literal["1.0"] = "1.0"


class ExitCode(IntEnum):
    """Process exit codes (§10). Click usage errors also exit 2."""

    OK = 0
    VALIDATION = 2
    NOTHING_TO_DO = 3  # nothing to do / already running
    BACKEND = 4  # backend or infrastructure failure


class EnvelopeStatusError(ValueError):
    """An envelope's ``ok`` disagrees with its ``exit_code``."""

    def __init__(self, ok: bool, exit_code: ExitCode) -> None:
        super().__init__(f"ok={ok} contradicts exit_code={int(exit_code)}")


class CliEnvelope(BaseModel):
    """The single JSON object a command writes to stdout under ``--json`` (provisional)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"] = ENVELOPE_SCHEMA_VERSION
    command: str
    exit_code: ExitCode
    ok: bool
    message: str | None = None
    data: dict[str, Any] | None = None
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _ok_matches_exit_code(self) -> Self:
        if self.ok != (self.exit_code is ExitCode.OK):
            raise EnvelopeStatusError(self.ok, self.exit_code)
        return self


_STYLE = {
    ExitCode.OK: "green",
    ExitCode.VALIDATION: "red",
    ExitCode.NOTHING_TO_DO: "yellow",
    ExitCode.BACKEND: "red",
}


def make_console(stream: TextIO) -> Console:
    """Return a Rich console that emits ANSI only when ``stream`` is a TTY, despite FORCE_COLOR."""
    return Console(file=stream, force_terminal=stream.isatty(), highlight=False)


def emit(envelope: CliEnvelope, *, as_json: bool, warnings_to_stderr: bool = False) -> None:
    """Write ``envelope``: one JSON line on stdout, or Rich text (stdout for 0/3, else stderr).

    In text mode, ``warnings_to_stderr`` sends warning lines to stderr even on success (on
    failure everything goes there already); under ``--json`` warnings are in the envelope only.
    """
    if as_json:
        typer.echo(envelope.model_dump_json())
        return
    ok = envelope.exit_code in (ExitCode.OK, ExitCode.NOTHING_TO_DO)
    style = _STYLE[envelope.exit_code]
    message = [envelope.message] if envelope.message is not None else []
    warnings = [f"warning: {w}" for w in envelope.warnings]
    errors = [f"error: {e}" for e in envelope.errors]
    if ok and warnings_to_stderr:
        _print(sys.stderr, warnings, _STYLE[ExitCode.NOTHING_TO_DO])
        warnings = []
    _print(sys.stdout if ok else sys.stderr, [*message, *warnings, *errors], style)


def _print(stream: TextIO, lines: list[str], style: str) -> None:
    console = make_console(stream)
    for line in lines:
        console.print(line, style=style, markup=False, soft_wrap=True)
