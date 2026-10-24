"""Drift checks for repo-level project files: CI, pre-commit, CITATION.cff, CHANGELOG."""

import re
import tomllib
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _yaml(rel: str) -> Any:
    return yaml.safe_load((ROOT / rel).read_text(encoding="utf-8"))


def _pyproject() -> dict[str, Any]:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _pixi() -> dict[str, Any]:
    return tomllib.loads((ROOT / "pixi.toml").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- CI


def _ci() -> dict[str, Any]:
    ci: dict[Any, Any] = _yaml(".github/workflows/ci.yml")
    # PyYAML (YAML 1.1) reads the bare key `on` as the boolean True.
    ci["on"] = ci.pop(True)
    return ci


def test_ci_runs_on_pull_requests_and_pushes_to_main_only() -> None:
    on = _ci()["on"]
    assert set(on) == {"pull_request", "push"}
    assert on["push"] == {"branches": ["main"]}


def test_ci_matrix_covers_every_python_env() -> None:
    job = _ci()["jobs"]["lint-test"]
    py_envs = sorted(e for e in _pixi()["environments"] if re.fullmatch(r"py3\d+", e))
    assert job["strategy"]["fail-fast"] is False
    assert job["strategy"]["matrix"] == {
        "os": ["ubuntu-latest", "macos-latest"],
        "env": py_envs,
    }
    runs = [step.get("run") for step in job["steps"]]
    assert "pixi run -e ${{ matrix.env }} lint" in runs
    assert "pixi run -e ${{ matrix.env }} test" in runs


def test_ci_pins_pixi_and_uses_the_lock() -> None:
    for name in ("lint-test", "pre-commit"):
        (setup,) = [s for s in _ci()["jobs"][name]["steps"] if "setup-pixi" in s.get("uses", "")]
        assert setup["with"]["pixi-version"] == "v0.83.0"
        assert setup["with"]["locked"] is True


def test_ci_wheel_job_installs_without_pixi() -> None:
    job = _ci()["jobs"]["wheel-install"]
    assert job["runs-on"] == "ubuntu-latest"
    text = yaml.safe_dump(job)
    assert "pixi" not in text
    assert "python -m build --wheel" in text
    assert re.search(r'raman"? --help', text)
    assert "import raman_bayes" in text


def test_ci_jobs_have_a_timeout() -> None:
    jobs = _ci()["jobs"]
    assert set(jobs) == {"lint-test", "pre-commit", "wheel-install"}
    assert all(job["timeout-minutes"] == 30 for job in jobs.values())


def test_ci_actions_are_pinned_to_versions() -> None:
    for job in _ci()["jobs"].values():
        for step in job["steps"]:
            if "uses" in step:
                assert re.fullmatch(r"[\w.-]+/[\w.-]+@v\d+(\.\d+)*", step["uses"]), step["uses"]


# ---------------------------------------------------------------- pre-commit


def test_pre_commit_hooks() -> None:
    config = _yaml(".pre-commit-config.yaml")
    ids: set[str] = set()
    for repo in config["repos"]:
        if repo["repo"] != "local":
            assert re.fullmatch(r"v\d+(\.\d+)+", repo["rev"]), repo
        for hook in repo["hooks"]:
            ids.add(hook["id"])
            if repo["repo"] == "local":
                assert hook["language"] == "system"
                assert hook["entry"].startswith("pixi run "), hook
    assert {
        "check-yaml",
        "check-toml",
        "end-of-file-fixer",
        "trailing-whitespace",
        "check-added-large-files",
        "gitleaks",
        "ruff-check",
        "ruff-format",
        "mypy",
        "lint-imports",
    } <= ids


# ---------------------------------------------------------------- CITATION / CHANGELOG


def test_citation_matches_pyproject() -> None:
    citation = _yaml("CITATION.cff")
    project = _pyproject()["project"]
    assert citation["cff-version"] == "1.2.0"
    assert citation["title"] == project["name"]
    assert citation["version"] == project["version"]
    assert "license" not in citation  # pending OTL (DESIGN.md §2)
    (author,) = citation["authors"]
    assert f"{author['given-names']} {author['family-names']}" == project["authors"][0]["name"]
    assert re.fullmatch(r"https://orcid\.org/\d{4}-\d{4}-\d{4}-\d{3}[\dX]", author["orcid"])


def test_changelog_has_unreleased_section() -> None:
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "## [Unreleased]" in text
