"""The installed ``raman`` console script, run as a real subprocess with stdout piped."""

import os
import shutil
import subprocess

from raman_bayes.cli.output import CliEnvelope, ExitCode

ESC = "\x1b"


def test_console_script_without_a_tty() -> None:
    exe = shutil.which("raman")
    assert exe is not None, "raman not on PATH; run via pixi"
    # Typer forces a Rich terminal when any of these is set; stdout is a pipe, so no ANSI.
    env = os.environ | {"GITHUB_ACTIONS": "true", "FORCE_COLOR": "1", "PY_COLORS": "1"}

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [exe, *args], capture_output=True, text=True, env=env, check=False, timeout=60
        )

    root_help = run("--help")
    assert root_help.returncode == 0, root_help.stderr
    assert "run-task" in root_help.stdout
    assert ESC not in root_help.stdout + root_help.stderr

    sub_help = run("priors", "export-csv", "--help")
    assert sub_help.returncode == 0, sub_help.stderr
    assert "--window-id" in sub_help.stdout
    assert ESC not in sub_help.stdout + sub_help.stderr

    stub = run("doctor")
    assert stub.returncode == ExitCode.NOTHING_TO_DO
    assert "not implemented yet" in stub.stdout
    assert ESC not in stub.stdout + stub.stderr

    as_json = run("doctor", "--json")
    assert as_json.returncode == ExitCode.NOTHING_TO_DO, as_json.stderr
    env_out = CliEnvelope.model_validate_json(as_json.stdout)
    assert env_out.command == "doctor"
    assert env_out.exit_code is ExitCode.NOTHING_TO_DO
