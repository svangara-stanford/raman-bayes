"""Engine discovery through the ``raman_bayes.engines`` entry-point group (DESIGN.md §6)."""

import zipfile
from importlib.metadata import EntryPoint, EntryPoints, entry_points

import pytest

from raman_bayes.domain.errors import InvalidEngineError, UnknownEngineError
from raman_bayes.engine import registry
from raman_bayes.engine.mock import MockEngine
from raman_bayes.engine.registry import ENGINE_GROUP, available_engines, load_engine
from raman_bayes.ports.engine import PeakEngine


def test_mock_registered_in_entry_point_group() -> None:
    assert ENGINE_GROUP == "raman_bayes.engines"
    (mock,) = entry_points(group="raman_bayes.engines", name="mock")
    assert mock.value == "raman_bayes.engine.mock:MockEngine"
    assert "mock" in available_engines()


def test_unknown_engine_error_lists_available() -> None:
    with pytest.raises(UnknownEngineError) as exc:
        load_engine("bayes_v9")
    message = str(exc.value)
    assert "'bayes_v9'" in message
    assert "mock" in message


def test_loaded_engine_satisfies_peak_engine() -> None:
    engine = load_engine("mock")
    assert isinstance(engine, PeakEngine)
    assert isinstance(engine, MockEngine)
    assert (engine.name, engine.version) == ("mock", "0.1.0")


def test_wheel_declares_engine_entry_point(wheel: zipfile.ZipFile) -> None:
    (name,) = [n for n in wheel.namelist() if n.endswith(".dist-info/entry_points.txt")]
    text = wheel.read(name).decode()
    assert "[raman_bayes.engines]" in text
    assert "mock = raman_bayes.engine.mock:MockEngine" in text


def test_entry_point_that_is_not_an_engine_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = EntryPoint(name="not_engine", value="collections:OrderedDict", group=ENGINE_GROUP)

    def only_fake(*, group: str, name: str | None = None) -> EntryPoints:
        assert group == ENGINE_GROUP
        return EntryPoints(ep for ep in [fake] if name in (None, ep.name))

    monkeypatch.setattr(registry, "entry_points", only_fake)
    assert available_engines() == ("not_engine",)
    with pytest.raises(InvalidEngineError, match="collections:OrderedDict"):
        load_engine("not_engine")
