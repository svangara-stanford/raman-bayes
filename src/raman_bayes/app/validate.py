"""Validate a priors YAML file: safe loading, structure, then the §5 file-level rules.

Problems come back as messages, never as exceptions, so ``validate-priors`` can list them all.
Structure errors stop before the rules, which need a parsed recipe.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError
from pydantic_core import ErrorDetails

from raman_bayes.domain.priors import PriorsSpec
from raman_bayes.domain.priors_rules import check_rules, priors_hash

__all__ = ["PriorsValidation", "validate_priors_file"]


@dataclass(frozen=True)
class PriorsValidation:
    """Outcome of validating one priors file.

    ``spec`` and ``priors_hash`` are set exactly when ``errors`` is empty.
    """

    errors: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    spec: PriorsSpec | None = None
    priors_hash: str | None = None


def validate_priors_file(path: Path) -> PriorsValidation:
    """Load ``path`` with the safe YAML loader and check it; return every error and warning."""
    data = _load(path)
    if isinstance(data, str):
        return PriorsValidation(errors=(data,))
    try:
        spec = PriorsSpec.model_validate(data)
    except ValidationError as exc:
        return PriorsValidation(errors=tuple(_describe(error) for error in exc.errors()))
    report = check_rules(spec)
    if report.errors:
        return PriorsValidation(errors=report.errors, warnings=report.warnings)
    return PriorsValidation(
        errors=(), warnings=report.warnings, spec=spec, priors_hash=priors_hash(spec)
    )


def _load(path: Path) -> dict[str, Any] | str:
    # Returns the parsed mapping, or the one message saying why there is none.
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return f"{path}: cannot read file ({exc.strerror})"
    except UnicodeDecodeError:
        return f"{path}: not UTF-8 text"
    try:
        data = yaml.safe_load(text)
    except yaml.MarkedYAMLError as exc:
        return _yaml_problem(exc)
    except yaml.YAMLError as exc:
        return f"not valid YAML: {exc}"
    if not isinstance(data, dict):
        return "the file must hold a mapping of priors sections"
    return data


def _yaml_problem(exc: yaml.MarkedYAMLError) -> str:
    mark = exc.problem_mark or exc.context_mark
    where = f"line {mark.line + 1}, column {mark.column + 1}: " if mark is not None else ""
    return f"{where}{exc.problem or exc.context}"


def _location(loc: tuple[int | str, ...]) -> str:
    text = ""
    for part in loc:
        text += f"[{part}]" if isinstance(part, int) else f".{part}"
    return text.removeprefix(".")


def _describe(error: ErrorDetails) -> str:
    where = _location(error["loc"])
    if error["type"] == "extra_forbidden":
        return f"{where}: unknown key, not allowed in this schema version"
    given: Any = error["input"]
    shown = f" (got {given!r})" if isinstance(given, str | int | float | bool) else ""
    return f"{where}: {error['msg']}{shown}"
