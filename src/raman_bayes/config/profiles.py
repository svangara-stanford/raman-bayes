"""Find a profile by name or path (DESIGN.md §9).

A value that looks like a file (contains a path separator or ends in ``.yaml``/``.yml``) is a
path, used as given relative to the working directory. Anything else is a name, looked up first
among the profiles packaged with raman-bayes and then in ``./configs/profiles/``.
"""

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from raman_bayes.domain.errors import UnknownProfileError

__all__ = ["ProfileSource", "resolve_profile"]

_PACKAGED = files("raman_bayes.config").joinpath("profiles")
_SITE_DIR = Path("configs") / "profiles"


@dataclass(frozen=True)
class ProfileSource:
    """A located profile: ``label`` is ``package:<name>.yaml`` or the absolute file path."""

    label: str
    text: str


def resolve_profile(profile: str, cwd: Path) -> ProfileSource:
    """Return the profile named or pointed to by ``profile``.

    Raises UnknownProfileError, naming every location searched, when no file exists.
    """
    if _is_path(profile):
        path = cwd / profile
        if not path.is_file():
            raise UnknownProfileError(profile, [str(path)])
        return ProfileSource(label=str(path), text=path.read_text(encoding="utf-8"))
    packaged = _PACKAGED.joinpath(f"{profile}.yaml")
    if packaged.is_file():
        return ProfileSource(label=f"package:{profile}.yaml", text=packaged.read_text("utf-8"))
    site = cwd / _SITE_DIR / f"{profile}.yaml"
    if site.is_file():
        return ProfileSource(label=str(site), text=site.read_text(encoding="utf-8"))
    raise UnknownProfileError(profile, ["the built-in profiles", str(site)])


def _is_path(profile: str) -> bool:
    return "/" in profile or "\\" in profile or profile.endswith((".yaml", ".yml"))
