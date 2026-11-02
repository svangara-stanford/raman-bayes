"""``scripts/make_fixture.py`` on a synthetic raw tree: selection, byte-exact copies, overwrites."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
INFO_SCANS = ("NMC111_v1", "NMC111_v2", "NMC811_v0", "NMC811_v1", "NMC811_v2")
NMC111_SPECTRA = ("NMC111_v1-z0y0x0.txt", "NMC111_v1-z0y11x0.txt")


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("make_fixture", ROOT / "scripts/make_fixture.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


make_fixture = _load_script()


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _archive(scan: str) -> str:
    return f"{scan.split('_')[0]}-Exported-Data"


@pytest.fixture
def raw(tmp_path: Path) -> Path:
    """Build a raw tree shaped like the unzipped archives, every file with unique bytes."""
    root = tmp_path / "raw"
    for scan in INFO_SCANS:
        _write(root / _archive(scan) / scan / f"{scan}-Info.txt", f"{scan} info\r\n \r\n".encode())
    mini = root / "NMC811-Exported-Data" / "NMC811_v0"
    for iz in range(2):
        for iy in range(7):
            for ix in range(7):
                name = f"NMC811_v0-z{iz}y{iy}x{ix}.txt"
                _write(mini / name, f"-41.5\t{iz}{iy}{ix}.00\r\n".encode())
    nmc111 = root / "NMC111-Exported-Data" / "NMC111_v1"
    _write(nmc111 / NMC111_SPECTRA[0], b"-41.55539\t6042.00\r\n")
    _write(nmc111 / NMC111_SPECTRA[1], b"-41.55539\t16042.00\r\n")
    _write(nmc111 / "NMC111_v1-z0y1x0.txt", b"-41.55539\t6043.00\r\n")
    return root


def _corner() -> set[str]:
    return {f"NMC811_v0-z0y{iy}x{ix}.txt" for iy in range(5) for ix in range(5)}


def _run(raw: Path, out: Path, *, force: bool = False) -> list[Path]:
    written: list[Path] = make_fixture.make_fixtures(raw, out, force=force)
    return written


def test_selects_exactly_the_corner_files(raw: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    _run(raw, out)
    mini = out / "nanofinder_mini" / "NMC811_v0"
    assert {p.name for p in mini.iterdir()} == _corner() | {"NMC811_v0-Info.txt"}


def test_copies_are_byte_identical(raw: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    written = _run(raw, out)
    assert len(written) == 26 + 5 + 2
    for dst in written:
        (src,) = raw.glob(f"*/*/{dst.name}")
        assert dst.read_bytes() == src.read_bytes(), dst


def test_collects_all_five_info_files(raw: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    _run(raw, out)
    info = out / "info_txt"
    assert sorted(p.name for p in info.iterdir()) == [f"{s}-Info.txt" for s in INFO_SCANS]
    for scan in INFO_SCANS:
        src = raw / _archive(scan) / scan / f"{scan}-Info.txt"
        assert (info / f"{scan}-Info.txt").read_bytes() == src.read_bytes()


def test_copies_two_nmc111_spectra_of_different_sizes(raw: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    _run(raw, out)
    spectra = sorted((out / "spectra").iterdir())
    assert sorted(p.name for p in spectra) == sorted(NMC111_SPECTRA)
    assert len({p.stat().st_size for p in spectra}) == 2


def test_refuses_to_overwrite_differing_fixture_without_force(raw: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    _run(raw, out)
    edited = out / "info_txt" / "NMC811_v0-Info.txt"
    edited.write_bytes(b"edited\r\n")
    with pytest.raises(make_fixture.FixtureConflictError, match=r"NMC811_v0-Info\.txt"):
        _run(raw, out)
    assert edited.read_bytes() == b"edited\r\n"


def test_force_overwrites_differing_fixture(raw: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    _run(raw, out)
    edited = out / "info_txt" / "NMC811_v0-Info.txt"
    edited.write_bytes(b"edited\r\n")
    assert _run(raw, out, force=True) == [edited]
    assert (
        edited.read_bytes()
        == (raw / "NMC811-Exported-Data/NMC811_v0/NMC811_v0-Info.txt").read_bytes()
    )


def test_rerun_on_identical_fixture_is_a_noop(raw: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    _run(raw, out)
    before = {p: p.stat().st_mtime_ns for p in out.rglob("*") if p.is_file()}
    assert _run(raw, out) == []
    assert {p: p.stat().st_mtime_ns for p in out.rglob("*") if p.is_file()} == before


def test_missing_source_file_is_a_clear_error(
    raw: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (raw / "NMC811-Exported-Data/NMC811_v0/NMC811_v0-z0y4x4.txt").unlink()
    out = tmp_path / "out"
    assert make_fixture.main(["--raw", str(raw), "--out", str(out)]) == 1
    err = capsys.readouterr().err
    assert "missing" in err
    assert "NMC811_v0-z0y4x4.txt" in err
    assert not out.exists()
