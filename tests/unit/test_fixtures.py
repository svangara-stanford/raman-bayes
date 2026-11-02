"""The committed Nanofinder fixtures (DESIGN.md §11) and the settings that keep them verbatim."""

import re
from pathlib import Path

import yaml

from raman_bayes.domain.dataset import FILENAME_RE

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
MINI = FIXTURES / "nanofinder_mini" / "NMC811_v0"
INFO = FIXTURES / "info_txt"
SPECTRA = FIXTURES / "spectra"
INFO_SCANS = ("NMC111_v1", "NMC111_v2", "NMC811_v0", "NMC811_v1", "NMC811_v2")


def _mini_spectra() -> list[Path]:
    return sorted(p for p in MINI.iterdir() if p.name != "NMC811_v0-Info.txt")


def _crlf_lines(path: Path) -> list[bytes]:
    data = path.read_bytes()
    assert data.endswith(b"\r\n"), path
    lines = data[:-2].split(b"\r\n")
    assert not any(b"\n" in line or b"\r" in line for line in lines), path
    return lines


def test_mini_has_info_plus_25_spectra() -> None:
    names = {p.name for p in MINI.iterdir()}
    assert "NMC811_v0-Info.txt" in names
    assert len(names) == 26


def test_mini_filenames_match_filename_re_within_corner() -> None:
    indices = set()
    for path in _mini_spectra():
        match = re.match(FILENAME_RE, path.name)
        assert match is not None, path.name
        assert match["scan"] == "NMC811_v0"
        indices.add((int(match["iz"]), int(match["iy"]), int(match["ix"])))
    assert indices == {(0, iy, ix) for iy in range(5) for ix in range(5)}


def test_mini_spectra_are_1024_crlf_two_column_lines() -> None:
    for path in _mini_spectra():
        lines = _crlf_lines(path)
        assert len(lines) == 1024, path.name
        for line in lines:
            shift, intensity = line.split(b"\t")
            float(shift)
            float(intensity)


def test_mini_info_equals_info_txt_copy() -> None:
    mini_info = (MINI / "NMC811_v0-Info.txt").read_bytes()
    assert mini_info == (INFO / "NMC811_v0-Info.txt").read_bytes()


def test_five_info_files_present() -> None:
    assert sorted(p.name for p in INFO.iterdir()) == [f"{s}-Info.txt" for s in INFO_SCANS]


def test_info_files_keep_crlf_and_single_space_blank_lines() -> None:
    for path in INFO.iterdir():
        lines = _crlf_lines(path)
        assert lines.count(b" ") == 7, path.name
        assert b"" not in lines, path.name


def test_info_line_counts_are_102_for_nmc111_and_104_for_nmc811() -> None:
    counts = {p.name.removesuffix("-Info.txt"): len(_crlf_lines(p)) for p in INFO.iterdir()}
    assert counts == {
        "NMC111_v1": 102,
        "NMC111_v2": 102,
        "NMC811_v0": 104,
        "NMC811_v1": 104,
        "NMC811_v2": 104,
    }


def test_nmc111_spectra_differ_in_byte_size() -> None:
    spectra = sorted(SPECTRA.iterdir())
    assert [p.name for p in spectra] == ["NMC111_v1-z0y0x0.txt", "NMC111_v1-z0y11x0.txt"]
    assert [p.stat().st_size for p in spectra] == [19920, 19922]
    for path in spectra:
        assert len(_crlf_lines(path)) == 1024, path.name


def test_gitattributes_marks_fixtures_and_legacy_csvs_minus_text() -> None:
    lines = (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()
    assert "tests/fixtures/** -text" in lines
    assert "configs/priors/legacy/*.csv -text" in lines


def test_precommit_excludes_fixtures_from_whitespace_fixers() -> None:
    config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"))
    hooks = {hook["id"]: hook for repo in config["repos"] for hook in repo["hooks"]}
    for hook_id in ("end-of-file-fixer", "trailing-whitespace"):
        exclude = re.compile(hooks[hook_id]["exclude"])
        assert exclude.search("tests/fixtures/info_txt/NMC811_v0-Info.txt"), hook_id
        assert exclude.search("tests/fixtures/nanofinder_mini/NMC811_v0/NMC811_v0-z0y0x0.txt")
        assert exclude.search("configs/priors/legacy/nmc_carbon.csv"), hook_id
        assert not exclude.search("src/raman_bayes/__init__.py"), hook_id
        assert not exclude.search("docs/fixtures-notes.md"), hook_id
