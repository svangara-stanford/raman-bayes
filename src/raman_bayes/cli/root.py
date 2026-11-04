"""The root command group: every outcome under ``--json`` is one envelope (DESIGN.md §10).

Without ``--json`` Typer handles usage errors as usual: usage text on stderr, exit 2. With it,
a usage error is an envelope with exit 2 instead. A ``RamanBayesError`` or ``OSError`` that a
command does not handle itself is a storage or backend problem: exit 4, with an envelope
under ``--json`` and a message on stderr otherwise. Any other exception is a bug and is not
caught, so it ends with a traceback and exit 1.
"""

import sys
from collections.abc import Sequence
from typing import Any

import typer
from typer.core import TyperGroup

from raman_bayes.cli.output import CliEnvelope, ExitCode, emit
from raman_bayes.domain.errors import RamanBayesError

__all__ = ["JSON_FLAG", "RootGroup"]

JSON_FLAG = "--json"


class RootGroup(TyperGroup):
    """The ``raman`` group, wrapping Typer's ``main`` with the envelope rules above."""

    def main(
        self,
        args: Sequence[str] | None = None,
        prog_name: str | None = None,
        complete_var: str | None = None,
        standalone_mode: bool = True,
        windows_expand_args: bool = True,
        **extra: Any,
    ) -> Any:
        """Run the command line ``args`` (default: ``sys.argv[1:]``) and exit with its code.

        With ``standalone_mode`` false, return the exit code instead of exiting.
        """
        argv = list(sys.argv[1:] if args is None else args)
        as_json = JSON_FLAG in argv
        command = _command_path(self, argv)
        try:
            # Under --json Typer must raise usage errors rather than print them.
            code = super().main(
                argv,
                prog_name,
                complete_var,
                standalone_mode=standalone_mode and not as_json,
                windows_expand_args=windows_expand_args,
                **extra,
            )
        except typer.TyperException as exc:  # raised only under --json: a usage error
            code = _report(command, ExitCode.VALIDATION, "usage error", exc, as_json=True)
        except typer.Abort:
            typer.echo("Aborted!", err=True)
            code = 1
        except (RamanBayesError, OSError) as exc:
            problem = "storage or backend problem"
            code = _report(command, ExitCode.BACKEND, problem, exc, as_json=as_json)
        code = code if isinstance(code, int) else 0
        if not standalone_mode:
            return code
        sys.exit(code)


def _report(command: str, code: ExitCode, problem: str, error: Exception, *, as_json: bool) -> int:
    message = error.format_message() if isinstance(error, typer.TyperException) else str(error)
    envelope = CliEnvelope(
        command=command,
        exit_code=code,
        ok=False,
        message=f"raman {command}: {problem}",
        errors=(message,),
    )
    emit(envelope, as_json=as_json)
    return int(code)


def _command_path(group: TyperGroup, argv: Sequence[str]) -> str:
    # The leading words of argv that name commands, as the envelope's command ("runs show").
    names: list[str] = []
    current: Any = group
    for token in argv:
        found = getattr(current, "commands", {}).get(token)
        if found is None:
            break
        names.append(token)
        current = found
    return " ".join(names) or "raman"
