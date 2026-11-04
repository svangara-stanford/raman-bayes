"""Settings (DESIGN.md §9): defaults < profile YAML < ``RAMAN_*`` environment < CLI flags.

``load_settings`` builds the layers explicitly rather than through pydantic-settings' implicit
sources, because the profile layer depends on the call (which profile, which working directory)
and must sit between the environment and the defaults. pydantic-settings still parses the
environment: ``RAMAN_LOCAL__WORKERS=4`` sets ``local.workers``, list values are JSON. Relative
``paths.*`` values resolve against the working directory after the layers are merged, so
values from every layer are treated alike.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, model_validator
from pydantic_core import ErrorDetails, PydanticCustomError
from pydantic_settings import (
    BaseSettings,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from raman_bayes.config.expansion import expand_variables
from raman_bayes.config.profiles import resolve_profile
from raman_bayes.domain.errors import InvalidProfileError

__all__ = [
    "DEFAULT_LOCK_STALE_MINUTES",
    "DEFAULT_PROFILE",
    "ENV_PREFIX",
    "EngineSettings",
    "LimitsSettings",
    "LoadedSettings",
    "LocalSettings",
    "PathsSettings",
    "Settings",
    "SlurmSettings",
    "load_settings",
    "requested_profile",
]

DEFAULT_PROFILE = "local"
DEFAULT_LOCK_STALE_MINUTES = 30.0
ENV_PREFIX = "RAMAN_"
_PROFILE_VAR = f"{ENV_PREFIX}PROFILE"  # selects the profile; never a settings value
_STRICT = ConfigDict(frozen=True, extra="forbid")


class PathsSettings(BaseModel):
    """Where datasets, runs and the catalog live; absolute once loaded."""

    model_config = _STRICT

    data_root: Path
    runs_root: Path
    catalog: Path
    archive_root: Path | None = None
    repo_dir: Path | None = None


class LocalSettings(BaseModel):
    """The local process-pool backend."""

    model_config = _STRICT

    workers: Annotated[int, Field(ge=1)] = 1


class EngineSettings(BaseModel):
    """The peak engine a plan uses unless ``--engine`` says otherwise."""

    model_config = _STRICT

    name: str = "mock"


class SlurmSettings(BaseModel):
    """The Slurm array backend; ``mem_per_cpu`` is passed to sbatch as written (``4G``)."""

    model_config = _STRICT

    partitions: Annotated[tuple[str, ...], Field(min_length=1)]
    cpus_per_task: Annotated[int, Field(ge=1)]
    mem_per_cpu: Annotated[str, Field(pattern=r"^\d+[KMGT]?$")]
    target_task_minutes: Annotated[float, Field(gt=0)]
    max_array_tasks: Annotated[int, Field(ge=1)]
    max_concurrent: Annotated[int, Field(ge=1)]
    pixi_bin: Path
    lock_stale_minutes: Annotated[float, Field(gt=0)] = DEFAULT_LOCK_STALE_MINUTES


class LimitsSettings(BaseModel):
    """Guards that ask for confirmation before large submissions."""

    model_config = _STRICT

    max_core_hours_without_confirm: Annotated[float, Field(gt=0)]


class Settings(BaseSettings):
    """The resolved configuration. Build it with ``load_settings``, not directly."""

    model_config = SettingsConfigDict(
        frozen=True,
        extra="forbid",
        env_prefix=ENV_PREFIX,
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    profile: str
    paths: PathsSettings
    backend: Literal["local", "slurm_array"] = "local"
    local: LocalSettings = LocalSettings()
    engine: EngineSettings = EngineSettings()
    slurm: SlurmSettings | None = None
    limits: LimitsSettings | None = None

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Use the constructor's values only; ``load_settings`` has already layered them."""
        return (init_settings,)

    @property
    def lock_stale_minutes(self) -> float:
        """Minutes after a task lock's last heartbeat at which it is stale (§7.2).

        ``slurm.lock_stale_minutes``, or the default when the profile has no slurm section,
        as local profiles do.
        """
        return self.slurm.lock_stale_minutes if self.slurm else DEFAULT_LOCK_STALE_MINUTES

    @model_validator(mode="after")
    def _slurm_backend_has_slurm_section(self) -> Self:
        if self.backend == "slurm_array" and self.slurm is None:
            raise PydanticCustomError(
                "slurm_section_missing", "backend slurm_array needs a slurm section"
            )
        return self


@dataclass(frozen=True)
class LoadedSettings:
    """Settings and the profile they came from (``package:<name>.yaml`` or a file path)."""

    settings: Settings
    source: str

    def resolved_config(self) -> dict[str, JsonValue]:
        """Return the settings as JSON, every path absolute, plus ``profile_source``."""
        resolved: dict[str, JsonValue] = self.settings.model_dump(mode="json")
        resolved["profile_source"] = self.source
        return resolved


def requested_profile(profile: str | None = None) -> str:
    """Return the profile to load: ``profile`` if given, else ``RAMAN_PROFILE``, else ``local``."""
    return profile or os.environ.get(_PROFILE_VAR, DEFAULT_PROFILE)


def load_settings(
    profile: str | None = None,
    *,
    cwd: Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> LoadedSettings:
    """Load the settings for ``profile`` (a name or a path).

    ``profile`` defaults to ``RAMAN_PROFILE``, then ``local``. ``overrides`` are CLI flag values
    as a nested mapping, applied last; ``cwd`` (default: the process working directory)
    anchors relative profile paths and relative ``paths.*`` values. Raises UnknownProfileError,
    UnsetVariableError, MalformedVariableError or InvalidProfileError.
    """
    cwd = cwd or Path.cwd()
    found = resolve_profile(requested_profile(profile), cwd)
    try:
        data = yaml.safe_load(found.text)
    except yaml.YAMLError as exc:
        raise InvalidProfileError(found.label, [f"not valid YAML: {exc}"]) from exc
    if not isinstance(data, dict):
        raise InvalidProfileError(found.label, ["the file must hold a mapping of settings"])
    profile_layer = expand_variables(data, os.environ, source=found.label)
    merged = _merge(profile_layer, _environment(), overrides or {})
    if isinstance(merged.get("paths"), dict):
        merged["paths"] = _absolute(merged["paths"], cwd)
    try:
        settings = Settings(**merged)
    except ValidationError as exc:
        raise InvalidProfileError(found.label, [_describe(e) for e in exc.errors()]) from exc
    return LoadedSettings(settings=settings, source=found.label)


def _environment() -> dict[str, Any]:
    values = EnvSettingsSource(
        Settings, env_prefix=ENV_PREFIX, env_nested_delimiter="__", case_sensitive=False
    )()
    values.pop("profile", None)  # RAMAN_PROFILE chose the file; it is not the profile's name
    return values


def _merge(*layers: Mapping[str, Any]) -> dict[str, Any]:
    # Later layers win; nested mappings merge key by key instead of replacing each other.
    merged: dict[str, Any] = {}
    for layer in layers:
        for key, value in layer.items():
            if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
                merged[key] = _merge(merged[key], value)
            else:
                merged[key] = value
    return merged


def _absolute(paths: Mapping[str, Any], cwd: Path) -> dict[str, Any]:
    return {
        key: os.path.normpath(cwd / value) if isinstance(value, str | Path) else value
        for key, value in paths.items()
    }


def _describe(error: ErrorDetails) -> str:
    where = ".".join(str(part) for part in error["loc"]) or "settings"
    if error["type"] == "extra_forbidden":
        return f"{where}: unknown key"
    return f"{where}: {error['msg']}"
