"""Strict ``${NAME}`` expansion of profile values (DESIGN.md §9).

Only the braced form is expanded, and only in profile values; ``$NAME`` stays literal and
``RAMAN_*`` overrides are never expanded. An unset variable or a stray ``${`` is an error that
names the profile key, because a silently empty path on a cluster is worse than a refusal.
Expanded text is not scanned again.
"""

import re
from collections.abc import Mapping
from typing import Any

from raman_bayes.domain.errors import MalformedVariableError, UnsetVariableError

__all__ = ["expand_variables"]

_VARIABLE = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)\}")


def expand_variables(data: Any, environ: Mapping[str, str], *, source: str) -> Any:
    """Return ``data`` with every ``${NAME}`` in its strings replaced from ``environ``.

    Mappings and lists are walked; other values are returned as they are. ``source`` names the
    profile in error messages. Raises UnsetVariableError for a variable missing from
    ``environ`` and MalformedVariableError for a ``${`` that does not open a valid name.
    """
    return _expand(data, environ, source, key="")


def _expand(data: Any, environ: Mapping[str, str], source: str, key: str) -> Any:
    if isinstance(data, Mapping):
        return {
            name: _expand(value, environ, source, f"{key}.{name}" if key else str(name))
            for name, value in data.items()
        }
    if isinstance(data, list):
        return [_expand(item, environ, source, f"{key}[{i}]") for i, item in enumerate(data)]
    if isinstance(data, str):
        return _expand_text(data, environ, source, key)
    return data


def _expand_text(text: str, environ: Mapping[str, str], source: str, key: str) -> str:
    if "${" in _VARIABLE.sub("", text):
        raise MalformedVariableError(text, key, source)

    def replace(match: re.Match[str]) -> str:
        name = match["name"]
        if name not in environ:
            raise UnsetVariableError(name, key, source)
        return environ[name]

    return _VARIABLE.sub(replace, text)
