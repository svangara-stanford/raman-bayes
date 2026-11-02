"""Settings loading (DESIGN.md §9): profiles, ${VAR} expansion, RAMAN_* and CLI precedence."""

import json
import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from raman_bayes.config.expansion import expand_variables
from raman_bayes.config.settings import load_settings
from raman_bayes.domain.errors import (
    InvalidProfileError,
    MalformedVariableError,
    UnknownProfileError,
    UnsetVariableError,
)

PATHS = {"data_root": "data/datasets", "runs_root": "data/runs", "catalog": "data/catalog.sqlite"}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # A developer's own RAMAN_* variables must not leak into these tests.
    for name in list(os.environ):
        if name.upper().startswith("RAMAN_"):
            monkeypatch.delenv(name)


def _profile(tmp_path: Path, body: dict[str, Any], name: str = "test.yaml") -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    return path


def _minimal(**extra: Any) -> dict[str, Any]:
    return {"profile": "test", "paths": dict(PATHS), **extra}


# ---------------------------------------------------------------- ${VAR} expansion


def test_expands_braced_var() -> None:
    assert expand_variables({"a": "${HOME}/x"}, {"HOME": "/home/me"}, source="p") == {
        "a": "/home/me/x"
    }


def test_unset_var_error_names_variable_and_key() -> None:
    with pytest.raises(UnsetVariableError) as exc:
        expand_variables({"paths": {"data_root": "${SCRATCH}/d"}}, {}, source="site.yaml")
    message = str(exc.value)
    assert "SCRATCH" in message
    assert "paths.data_root" in message
    assert "site.yaml" in message


def test_only_braced_syntax_is_expanded() -> None:
    env = {"HOME": "/home/me"}
    assert expand_variables({"a": "$HOME/x", "b": "100$"}, env, source="p") == {
        "a": "$HOME/x",
        "b": "100$",
    }
    for bad in ("${HOME", "${}", "${1X}", "x ${HOME } y"):
        with pytest.raises(MalformedVariableError, match=r"paths\.catalog"):
            expand_variables({"paths": {"catalog": bad}}, env, source="p")


def test_several_vars_in_one_value() -> None:
    env = {"GROUP_HOME": "/g", "USER": "me"}
    result = expand_variables({"a": "${GROUP_HOME}/${USER}/${USER}"}, env, source="p")
    assert result == {"a": "/g/me/me"}


def test_expansion_in_nested_values_and_lists() -> None:
    env = {"P": "normal", "N": "4"}
    data = {"slurm": {"partitions": ["${P}", "owners"], "nested": {"deep": ["x${N}"]}}, "n": 3}
    assert expand_variables(data, env, source="p") == {
        "slurm": {"partitions": ["normal", "owners"], "nested": {"deep": ["x4"]}},
        "n": 3,
    }
    with pytest.raises(UnsetVariableError, match=r"slurm\.partitions\[1\]"):
        expand_variables({"slurm": {"partitions": ["a", "${MISSING}"]}}, env, source="p")


def test_expanded_values_are_not_expanded_again() -> None:
    assert expand_variables({"a": "${V}"}, {"V": "${W}", "W": "no"}, source="p") == {"a": "${W}"}


# ---------------------------------------------------------------- layers


def test_env_override_values_are_not_expanded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAMAN_PATHS__RUNS_ROOT", "/runs/${HOME}")
    loaded = load_settings(str(_profile(tmp_path, _minimal())), cwd=tmp_path)
    assert loaded.settings.paths.runs_root == Path("/runs/${HOME}")


@pytest.mark.parametrize(
    ("layer", "workers"), [("defaults", 1), ("profile", 3), ("env", 5), ("cli", 7)]
)
def test_precedence_defaults_profile_env_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layer: str, workers: int
) -> None:
    layers = ["defaults", "profile", "env", "cli"]
    active = layers[: layers.index(layer) + 1]
    body = _minimal(local={"workers": 3}) if "profile" in active else _minimal()
    if "env" in active:
        monkeypatch.setenv("RAMAN_LOCAL__WORKERS", "5")
    overrides = {"local": {"workers": 7}} if "cli" in active else None
    loaded = load_settings(str(_profile(tmp_path, body)), cwd=tmp_path, overrides=overrides)
    assert loaded.settings.local.workers == workers


def test_nested_env_override_double_underscore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAMAN_PATHS__RUNS_ROOT", "/tmp/runs")
    monkeypatch.setenv("raman_engine__name", "other")
    monkeypatch.setenv("RAMAN_SLURM__PARTITIONS", '["normal", "owners"]')
    slurm = {
        "partitions": ["normal"],
        "cpus_per_task": 16,
        "mem_per_cpu": "4G",
        "target_task_minutes": 60,
        "max_array_tasks": 500,
        "max_concurrent": 200,
        "pixi_bin": "/home/me/.pixi/bin/pixi",
    }
    path = _profile(tmp_path, _minimal(slurm=slurm, backend="slurm_array"))
    settings = load_settings(str(path), cwd=tmp_path).settings
    assert settings.paths.runs_root == Path("/tmp/runs")
    assert settings.paths.data_root == tmp_path / "data/datasets"
    assert settings.engine.name == "other"
    assert settings.slurm is not None
    assert settings.slurm.partitions == ("normal", "owners")
    assert settings.slurm.cpus_per_task == 16


# ---------------------------------------------------------------- profile resolution


def test_profile_name_resolves_packaged_first(tmp_path: Path) -> None:
    _profile(tmp_path, _minimal(profile="shadow"), "configs/profiles/local.yaml")
    loaded = load_settings("local", cwd=tmp_path)
    assert loaded.source == "package:local.yaml"
    assert loaded.settings.profile == "local"


def test_profile_name_falls_back_to_cwd_configs_profiles(tmp_path: Path) -> None:
    path = _profile(tmp_path, _minimal(profile="lab"), "configs/profiles/lab.yaml")
    loaded = load_settings("lab", cwd=tmp_path)
    assert loaded.source == str(path)
    assert loaded.settings.profile == "lab"


def test_profile_path_used_as_given(tmp_path: Path) -> None:
    path = _profile(tmp_path, _minimal(profile="mine"), "elsewhere/mine.yaml")
    absolute = load_settings(str(path), cwd=tmp_path)
    relative = load_settings("elsewhere/mine.yaml", cwd=tmp_path)
    assert absolute.source == relative.source == str(path)
    assert absolute.settings == relative.settings
    named_like_package = _profile(tmp_path, _minimal(profile="x"), "local.yaml")
    assert load_settings("local.yaml", cwd=tmp_path).source == str(named_like_package)


def test_raman_profile_env_selects_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _profile(tmp_path, _minimal(profile="from_env"))
    monkeypatch.setenv("RAMAN_PROFILE", str(path))
    loaded = load_settings(cwd=tmp_path)
    assert loaded.settings.profile == "from_env"
    assert load_settings("local", cwd=tmp_path).settings.profile == "local"
    monkeypatch.delenv("RAMAN_PROFILE")
    assert load_settings(cwd=tmp_path).settings.profile == "local"


def test_unknown_profile_error_names_searched_locations(tmp_path: Path) -> None:
    with pytest.raises(UnknownProfileError) as exc:
        load_settings("nope", cwd=tmp_path)
    message = str(exc.value)
    assert "'nope'" in message
    assert "built-in profiles" in message
    assert str(tmp_path / "configs" / "profiles" / "nope.yaml") in message
    with pytest.raises(UnknownProfileError, match=r"missing\.yaml"):
        load_settings(str(tmp_path / "missing.yaml"), cwd=tmp_path)


@pytest.mark.parametrize(
    ("body", "key"),
    [
        (_minimal(local={"workers": 2, "threads": 4}), "local.threads"),
        (_minimal(gpu=True), "gpu"),
        (_minimal(local={"workers": 0}), "local.workers"),
    ],
)
def test_unknown_profile_key_rejected_naming_it(
    tmp_path: Path, body: dict[str, Any], key: str
) -> None:
    path = _profile(tmp_path, body)
    with pytest.raises(InvalidProfileError) as exc:
        load_settings(str(path), cwd=tmp_path)
    assert key in str(exc.value)
    assert str(path) in str(exc.value)


def test_slurm_backend_requires_slurm_section(tmp_path: Path) -> None:
    with pytest.raises(InvalidProfileError, match="slurm"):
        load_settings(str(_profile(tmp_path, _minimal(backend="slurm_array"))), cwd=tmp_path)


def test_relative_paths_resolve_against_cwd(tmp_path: Path) -> None:
    paths = {"data_root": "d", "runs_root": "../up/runs", "catalog": "/abs/c.sqlite"}
    path = _profile(tmp_path, {"profile": "t", "paths": paths})
    work = tmp_path / "work"
    work.mkdir()
    settings = load_settings(str(path), cwd=work).settings
    assert settings.paths.data_root == work / "d"
    assert settings.paths.runs_root == tmp_path / "up" / "runs"
    assert settings.paths.catalog == Path("/abs/c.sqlite")
    assert settings.paths.archive_root is None


def test_resolved_config_has_absolute_paths_and_profile_source(tmp_path: Path) -> None:
    resolved = load_settings("local", cwd=tmp_path).resolved_config()
    assert resolved["profile_source"] == "package:local.yaml"
    assert json.loads(json.dumps(resolved)) == resolved
    paths = resolved["paths"]
    assert isinstance(paths, dict)
    for key in ("data_root", "runs_root", "catalog"):
        value = paths[key]
        assert isinstance(value, str)
        assert Path(value).is_absolute(), key
    assert paths["catalog"] == str(tmp_path / "data" / "catalog.sqlite")
    custom = _profile(tmp_path, _minimal())
    assert load_settings(str(custom), cwd=tmp_path).resolved_config()["profile_source"] == str(
        custom
    )


def test_packaged_local_profile_validates(tmp_path: Path) -> None:
    settings = load_settings("local", cwd=tmp_path).settings
    assert settings.profile == "local"
    assert settings.backend == "local"
    assert settings.local.workers == 8
    assert settings.engine.name == "mock"
    assert settings.slurm is None
    assert settings.paths.data_root == tmp_path / "data" / "datasets"
