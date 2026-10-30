"""Carve the committed test fixtures from the raw Nanofinder archives (DESIGN.md §11, §14).

Developer-only: reads ``data/raw/``, which is gitignored, and writes under ``tests/fixtures/``.
Every copy is byte-for-byte, so CRLF line endings and the single-space blank lines of Info.txt
survive. Run with ``pixi run python scripts/make_fixture.py``.
"""

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MINI_SCAN = "NMC811_v0"
CORNER = range(5)
INFO_SCANS = ("NMC111_v1", "NMC111_v2", "NMC811_v0", "NMC811_v1", "NMC811_v2")
# §11 asks for NMC111 reader tests and the mini fixture is NMC811 only. These two NMC111_v1
# files are 19920 and 19922 bytes: same axis, different number formatting.
NMC111_SCAN = "NMC111_v1"
NMC111_SPECTRA = ("NMC111_v1-z0y0x0.txt", "NMC111_v1-z0y11x0.txt")


class FixtureError(Exception):
    """The fixtures cannot be written; nothing was changed."""


class MissingSourceError(FixtureError):
    """A scan folder or file the fixtures need is not in the raw tree."""

    def __init__(self, path: Path) -> None:
        super().__init__(f"missing source: {path}")


class FixtureConflictError(FixtureError):
    """Existing fixture files differ from their sources and ``force`` was not given."""

    def __init__(self, paths: list[Path]) -> None:
        listed = "\n  ".join(str(p) for p in paths)
        super().__init__(f"fixture files differ from the raw data (use --force):\n  {listed}")


def _scan_dir(raw: Path, scan: str) -> Path:
    # The archives unzip to raw/<archive>/<scan>/; the archive folder name is not fixed.
    matches = sorted(p for p in raw.glob(f"*/{scan}") if p.is_dir())
    if not matches:
        raise MissingSourceError(raw / "*" / scan)
    return matches[0]


def plan_copies(raw: Path, out: Path) -> list[tuple[Path, Path]]:
    """Return the ``(source, destination)`` pair of every fixture file.

    Raises MissingSourceError when a scan folder is not under ``raw``.
    """
    mini_src = _scan_dir(raw, MINI_SCAN)
    mini_dst = out / "nanofinder_mini" / MINI_SCAN
    mini_names = [f"{MINI_SCAN}-Info.txt"] + [
        f"{MINI_SCAN}-z0y{iy}x{ix}.txt" for iy in CORNER for ix in CORNER
    ]
    pairs = [(mini_src / name, mini_dst / name) for name in mini_names]
    for scan in INFO_SCANS:
        name = f"{scan}-Info.txt"
        pairs.append((_scan_dir(raw, scan) / name, out / "info_txt" / name))
    nmc111 = _scan_dir(raw, NMC111_SCAN)
    pairs += [(nmc111 / name, out / "spectra" / name) for name in NMC111_SPECTRA]
    return pairs


def make_fixtures(raw: Path, out: Path, *, force: bool = False) -> list[Path]:
    """Copy every fixture file from ``raw`` into ``out``; return the destinations written.

    Files already identical to their source are left alone. Raises MissingSourceError when a
    source is absent and FixtureConflictError when a destination differs and ``force`` is
    false; in both cases nothing is written.
    """
    pairs = plan_copies(raw, out)
    for src, _ in pairs:
        if not src.is_file():
            raise MissingSourceError(src)
    stale = [(s, d) for s, d in pairs if not (d.is_file() and filecmp.cmp(s, d, shallow=False))]
    conflicts = [d for _, d in stale if d.exists()]
    if conflicts and not force:
        raise FixtureConflictError(conflicts)
    for src, dst in stale:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    return [dst for _, dst in stale]


def main(argv: list[str] | None = None) -> int:
    """Write the fixtures; return 0 on success and 1, with the reason on stderr, otherwise."""
    parser = argparse.ArgumentParser(description="Carve the test fixtures from the raw archives.")
    parser.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--out", type=Path, default=ROOT / "tests" / "fixtures")
    parser.add_argument("--force", action="store_true", help="Overwrite differing fixtures.")
    args = parser.parse_args(argv)
    try:
        written = make_fixtures(args.raw, args.out, force=args.force)
    except FixtureError as exc:
        print(f"make_fixture: {exc}", file=sys.stderr)
        return 1
    print(f"make_fixture: {len(written)} file(s) written under {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
