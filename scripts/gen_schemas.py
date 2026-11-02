"""Generate the JSON schemas under schemas/ from the Pydantic models (DESIGN.md §2, §10).

Run with ``pixi run gen-schemas``; tests/unit/test_gen_schemas.py fails if the committed
file is stale.
"""

import argparse
import json
from pathlib import Path

from raman_bayes.cli.output import CliEnvelope

ROOT = Path(__file__).resolve().parents[1]
ENVELOPE_SCHEMA = ROOT / "schemas" / "cli" / "envelope.schema.json"


def render_envelope_schema() -> str:
    """Return the CliEnvelope JSON schema as sorted, indented JSON ending in a newline."""
    return json.dumps(CliEnvelope.model_json_schema(), indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> None:
    """Write the envelope schema to ``--out``, creating parent directories."""
    parser = argparse.ArgumentParser(description="Generate the CLI JSON schemas.")
    parser.add_argument("--out", type=Path, default=ENVELOPE_SCHEMA)
    args = parser.parse_args(argv)
    out: Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_envelope_schema(), encoding="utf-8")


if __name__ == "__main__":
    main()
