"""Import contracts (DESIGN.md §2), enforced by import-linter in ``pixi run lint``.

Checks that the four contracts pass on the real tree, that their module lists cover every
top-level module (so a new module must be explicitly allowed or forbidden), and that each
contract actually catches a planted violation.
"""

import os
import pkgutil
import shutil
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import pytest

import raman_bayes

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_IDS = ["cli-imports", "nobody-imports-cli", "domain-pure", "app-never-prints"]
# Top-level modules cli/ may import (DESIGN.md §2 contract 1). Copied by hand.
CLI_ALLOWED = {"app", "domain", "config", "logging"}


def _contracts() -> dict[str, dict[str, Any]]:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    contracts = config["tool"]["importlinter"]["contracts"]
    return {c["id"]: c for c in contracts}


def _top_level() -> set[str]:
    return {m.name for m in pkgutil.iter_modules(raman_bayes.__path__)}


def _qualified(names: set[str]) -> set[str]:
    return {f"raman_bayes.{n}" for n in names}


def _lint_imports(cwd: Path, pythonpath: Path | None = None) -> subprocess.CompletedProcess[str]:
    exe = shutil.which("lint-imports")
    assert exe is not None, "lint-imports not on PATH; run via pixi"
    env = None if pythonpath is None else os.environ | {"PYTHONPATH": str(pythonpath)}
    return subprocess.run(
        [exe, "--no-cache", "--no-logo"],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


# ---------------------------------------------------------------- configuration


def test_four_contracts_configured() -> None:
    contracts = _contracts()
    assert list(contracts) == CONTRACT_IDS
    assert all(c["type"] == "forbidden" for c in contracts.values())


def test_cli_contract_forbids_everything_not_allowed() -> None:
    contract = _contracts()["cli-imports"]
    assert contract["source_modules"] == ["raman_bayes.cli"]
    assert contract["allow_indirect_imports"] is True
    expected = _qualified(_top_level() - {"cli"} - CLI_ALLOWED)
    assert set(contract["forbidden_modules"]) == expected


def test_cli_contract_sources_cover_every_other_module() -> None:
    contract = _contracts()["nobody-imports-cli"]
    assert set(contract["source_modules"]) == _qualified(_top_level() - {"cli"})
    assert contract["forbidden_modules"] == ["raman_bayes.cli"]


def test_domain_contract_forbids_every_other_module() -> None:
    contract = _contracts()["domain-pure"]
    assert contract["source_modules"] == ["raman_bayes.domain"]
    assert set(contract["forbidden_modules"]) == _qualified(_top_level() - {"domain"})


def test_app_contract_forbids_rich_and_typer() -> None:
    contract = _contracts()["app-never-prints"]
    assert contract["source_modules"] == ["raman_bayes.app"]
    assert set(contract["forbidden_modules"]) == {"rich", "typer"}


# ---------------------------------------------------------------- behaviour


def test_real_tree_passes() -> None:
    result = _lint_imports(ROOT)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("KEPT") == len(CONTRACT_IDS)


# (contract id, package the violating module is planted in, import it makes)
VIOLATIONS = [
    ("cli-imports", "cli", "import raman_bayes.storage"),
    ("nobody-imports-cli", "storage", "import raman_bayes.cli"),
    ("domain-pure", "domain", "import raman_bayes.io"),
    ("app-never-prints", "app", "import rich"),
]


@pytest.mark.parametrize(("contract_id", "package", "statement"), VIOLATIONS, ids=CONTRACT_IDS)
def test_planted_violation_breaks_exactly_its_contract(
    tmp_path: Path, contract_id: str, package: str, statement: str
) -> None:
    src = tmp_path / "src"
    shutil.copytree(
        ROOT / "src" / "raman_bayes",
        src / "raman_bayes",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copy(ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    (src / "raman_bayes" / package / "_planted.py").write_text(f"{statement}\n", encoding="utf-8")

    result = _lint_imports(tmp_path, pythonpath=src)
    assert result.returncode != 0
    names = {c["id"]: c["name"] for c in _contracts().values()}
    broken = {cid for cid, name in names.items() if f"{name} BROKEN" in result.stdout}
    assert broken == {contract_id}, result.stdout
    assert "_planted" in result.stdout
