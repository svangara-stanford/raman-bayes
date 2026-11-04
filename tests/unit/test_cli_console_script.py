"""The installed ``raman`` console script, run as a real subprocess with stdout piped."""

import os
import shutil
import subprocess
from pathlib import Path

from raman_bayes.cli.output import CliEnvelope, ExitCode

ESC = "\x1b"


def test_console_script_without_a_tty(tmp_path: Path) -> None:
    exe = shutil.which("raman")
    assert exe is not None, "raman not on PATH; run via pixi"
    # Typer forces a Rich terminal when any of these is set; stdout is a pipe, so no ANSI.
    # No RAMAN_* variables and an empty working directory, so doctor checks an empty local
    # setup and gives the same result on a laptop with data/ as in CI.
    base = {k: v for k, v in os.environ.items() if not k.upper().startswith("RAMAN_")}
    env = base | {"GITHUB_ACTIONS": "true", "FORCE_COLOR": "1", "PY_COLORS": "1"}

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [exe, *args],
            capture_output=True,
            text=True,
            env=env,
            cwd=tmp_path,
            check=False,
            timeout=60,
        )

    root_help = run("--help")
    assert root_help.returncode == 0, root_help.stderr
    assert "run-task" in root_help.stdout
    assert ESC not in root_help.stdout + root_help.stderr

    sub_help = run("priors", "export-csv", "--help")
    assert sub_help.returncode == 0, sub_help.stderr
    assert "--window-id" in sub_help.stdout
    assert ESC not in sub_help.stdout + sub_help.stderr

    doctor = run("doctor")
    assert doctor.returncode == ExitCode.OK, doctor.stderr
    assert "profile local" in doctor.stdout
    assert ESC not in doctor.stdout + doctor.stderr

    as_json = run("doctor", "--json")
    assert as_json.returncode == ExitCode.OK, as_json.stderr
    env_out = CliEnvelope.model_validate_json(as_json.stdout)
    assert env_out.command == "doctor"
    assert env_out.exit_code is ExitCode.OK
