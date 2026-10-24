"""``scripts/gen_schemas.py``: the committed CLI envelope schema matches a fresh generation."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
COMMITTED = ROOT / "schemas/cli/envelope.schema.json"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("gen_schemas", ROOT / "scripts/gen_schemas.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_schema_matches_fresh_generation() -> None:
    fresh = _load_script().render_envelope_schema()
    assert COMMITTED.read_text(encoding="utf-8") == fresh, "run `pixi run gen-schemas`"


def test_main_writes_the_schema(tmp_path: Path) -> None:
    gen = _load_script()
    out = tmp_path / "cli" / "envelope.schema.json"
    gen.main(["--out", str(out)])
    assert out.read_text(encoding="utf-8") == gen.render_envelope_schema()


def test_schema_shape() -> None:
    schema = json.loads(COMMITTED.read_text(encoding="utf-8"))
    assert schema["title"] == "CliEnvelope"
    assert schema["additionalProperties"] is False
    assert {"command", "exit_code", "ok"} <= set(schema["required"])
    exit_code = schema["$defs"]["ExitCode"]
    assert exit_code["enum"] == [0, 2, 3, 4]
