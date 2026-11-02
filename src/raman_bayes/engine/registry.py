"""Find peak engines by name through the ``raman_bayes.engines`` entry-point group (DESIGN.md §6).

An engine package declares ``<name> = "<module>:<class>"`` in that group; the class is built
with no arguments.
"""

from importlib.metadata import entry_points

from raman_bayes.domain.errors import InvalidEngineError, UnknownEngineError
from raman_bayes.ports.engine import PeakEngine

__all__ = ["ENGINE_GROUP", "available_engines", "load_engine"]

ENGINE_GROUP = "raman_bayes.engines"


def available_engines() -> tuple[str, ...]:
    """Return the names of the installed engines, sorted."""
    return tuple(sorted({ep.name for ep in entry_points(group=ENGINE_GROUP)}))


def load_engine(name: str) -> PeakEngine:
    """Import and build the engine registered as ``name``.

    Raises UnknownEngineError, listing the installed engines, when none has that name, and
    InvalidEngineError when the registered object does not build a PeakEngine.
    """
    found = sorted(entry_points(group=ENGINE_GROUP, name=name), key=lambda ep: ep.value)
    if not found:
        raise UnknownEngineError(name, available_engines())
    entry_point = found[0]
    engine = entry_point.load()()
    if not isinstance(engine, PeakEngine):
        raise InvalidEngineError(name, entry_point.value)
    return engine
