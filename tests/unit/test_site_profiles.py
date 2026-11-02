"""The repo-level site profiles ``labpc`` and ``sherlock`` (DESIGN.md §7.5, §9)."""

import os
from pathlib import Path

import pytest
import yaml

from raman_bayes.config.settings import load_settings
from raman_bayes.domain.errors import UnsetVariableError

ROOT = Path(__file__).resolve().parents[2]
PROFILES = ROOT / "configs" / "profiles"
DESIGN_MD = ROOT / "docs" / "DESIGN.md"
SHERLOCK_ENV = {
    "SCRATCH": "/scratch/users/me",
    "GROUP_HOME": "/home/groups/lab",
    "HOME": "/home/me",
}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.upper().startswith("RAMAN_"):
            monkeypatch.delenv(name)


def _design_block(name: str) -> str:
    text = DESIGN_MD.read_text(encoding="utf-8")
    return text.split(f"# configs/profiles/{name}.yaml\n", 1)[1].split("```", 1)[0]


@pytest.mark.parametrize("name", ["labpc", "sherlock"])
def test_site_profiles_validate_with_env_set(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    for var, value in SHERLOCK_ENV.items():
        monkeypatch.setenv(var, value)
    loaded = load_settings(name, cwd=ROOT)
    assert loaded.source == str(PROFILES / f"{name}.yaml")
    settings = loaded.settings
    assert settings.profile == name
    if name == "labpc":
        assert settings.backend == "local"
        assert settings.local.workers == 48
        assert settings.paths.archive_root == Path("/archive/raman")
        return
    assert settings.backend == "slurm_array"
    assert settings.paths.data_root == Path("/scratch/users/me/raman/datasets")
    assert settings.paths.catalog == Path("/home/groups/lab/svangara/raman/catalog.sqlite")
    assert settings.slurm is not None
    assert settings.slurm.partitions == ("normal", "owners")
    assert settings.slurm.pixi_bin == Path("/home/me/.pixi/bin/pixi")
    assert settings.slurm.lock_stale_minutes == 30
    assert settings.limits is not None
    assert settings.limits.max_core_hours_without_confirm == 2000


def test_sherlock_profile_without_scratch_names_scratch(monkeypatch: pytest.MonkeyPatch) -> None:
    for var, value in SHERLOCK_ENV.items():
        monkeypatch.setenv(var, value)
    monkeypatch.delenv("SCRATCH")
    with pytest.raises(UnsetVariableError) as exc:
        load_settings("sherlock", cwd=ROOT)
    assert "SCRATCH" in str(exc.value)
    assert "paths.data_root" in str(exc.value)


@pytest.mark.parametrize("name", ["labpc", "sherlock"])
def test_site_profiles_match_design_md(name: str) -> None:
    committed = yaml.safe_load((PROFILES / f"{name}.yaml").read_text(encoding="utf-8"))
    assert committed == yaml.safe_load(_design_block(name))
