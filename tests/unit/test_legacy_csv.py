"""Legacy priors CSV import and export (DESIGN.md §5) on both committed files."""

import copy
import csv
from pathlib import Path
from typing import Any

import pytest
import yaml

from raman_bayes.domain.errors import LegacyCsvHeaderError, LegacyCsvRowError
from raman_bayes.domain.priors import Bounds, PriorsSpec, Window
from raman_bayes.domain.priors_rules import check_rules
from raman_bayes.io.legacy_csv import HEADER, format_legacy_csv, read_legacy_csv

ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT / "configs" / "priors" / "legacy"
LITHIATION = LEGACY / "nmc_lithiation.csv"
CARBON = LEGACY / "nmc_carbon.csv"
FILES: dict[str, tuple[Path, Bounds]] = {
    "lithiation": (LITHIATION, (350.0, 700.0)),
    "carbon": (CARBON, (1150.0, 1750.0)),
}
FIXTURE = ROOT / "tests" / "fixtures" / "priors" / "design_example.yaml"
BOM = b"\xef\xbb\xbf"
HEADER_LINE = ",x (center),x (lower limit),x (upper limit),FWHM (lower limit),FWHM (upper limit)"


def _import(window_id: str) -> Window:
    path, range_cm1 = FILES[window_id]
    return read_legacy_csv(path, window_id, range_cm1)


def _numbers(window: Window) -> list[tuple[str, float, float, float, float, float]]:
    return [(p.id, p.center.mean, *p.center.bounds, *p.fwhm.bounds) for p in window.peaks]


def _copy(tmp_path: Path, source: Path, data: bytes) -> Path:
    path = tmp_path / source.name
    path.write_bytes(data)
    return path


def test_import_lithiation_csv() -> None:
    raw = LITHIATION.read_bytes()
    assert raw.startswith(BOM)
    assert b"\r\n" in raw
    window = _import("lithiation")
    assert window.id == "lithiation"
    assert window.range_cm1 == (350.0, 700.0)
    assert _numbers(window) == [
        ("u1", 465.0, 445.0, 485.0, 35.0, 70.0),
        ("u2", 568.0, 550.0, 585.0, 30.0, 60.0),
    ]
    assert [p.label for p in window.peaks] == [None, None]


def test_import_carbon_csv() -> None:
    raw = CARBON.read_bytes()
    assert raw.isascii()
    assert b"\r" not in raw
    window = _import("carbon")
    assert _numbers(window) == [
        ("d", 1335.0, 1310.0, 1370.0, 90.0, 220.0),
        ("g", 1584.0, 1548.0, 1620.0, 35.0, 125.0),
    ]


def test_import_column_mapping(tmp_path: Path) -> None:
    text = f"{HEADER_LINE}\np1,10,1,20,2,30\n"
    path = _copy(tmp_path, Path("map.csv"), text.encode())
    (peak,) = read_legacy_csv(path, "w", (0.0, 100.0)).peaks
    assert peak.center.mean == 10.0
    assert peak.center.bounds == (1.0, 20.0)
    assert peak.fwhm.bounds == (2.0, 30.0)
    assert peak.center.sd is None


@pytest.mark.parametrize("window_id", list(FILES))
def test_import_lf_and_crlf_identical(tmp_path: Path, window_id: str) -> None:
    source, range_cm1 = FILES[window_id]
    lf = source.read_bytes().replace(b"\r\n", b"\n")
    crlf = lf.replace(b"\n", b"\r\n")
    (tmp_path / "lf").mkdir()
    (tmp_path / "crlf").mkdir()
    from_lf = read_legacy_csv(_copy(tmp_path / "lf", source, lf), window_id, range_cm1)
    from_crlf = read_legacy_csv(_copy(tmp_path / "crlf", source, crlf), window_id, range_cm1)
    assert from_lf == from_crlf == _import(window_id)


@pytest.mark.parametrize("window_id", list(FILES))
def test_import_with_and_without_bom_identical(tmp_path: Path, window_id: str) -> None:
    source, range_cm1 = FILES[window_id]
    bare = source.read_bytes().removeprefix(BOM)
    (tmp_path / "bare").mkdir()
    (tmp_path / "bom").mkdir()
    without = read_legacy_csv(_copy(tmp_path / "bare", source, bare), window_id, range_cm1)
    with_bom = read_legacy_csv(_copy(tmp_path / "bom", source, BOM + bare), window_id, range_cm1)
    assert without == with_bom == _import(window_id)


def test_import_lowercases_ids_and_keeps_label() -> None:
    window = _import("carbon")
    assert [(p.id, p.label) for p in window.peaks] == [("d", "D"), ("g", "G")]


@pytest.mark.parametrize(
    "row",
    [
        "d,1335.0,1310.0,1370.0,90.0",
        "d,1335.0,1310.0,1370.0,90.0,220.0,1",
        "d,1335.0,abc,1370.0,90.0,220.0",
        ",1335.0,1310.0,1370.0,90.0,220.0",
        "D,1335.0,1310.0,1370.0,90.0,220.0",
    ],
    ids=["short", "long", "non-numeric", "no-name", "duplicate-modulo-case"],
)
def test_import_rejects_malformed_row_naming_it(tmp_path: Path, row: str) -> None:
    text = f"{HEADER_LINE}\nd,1,0,2,1,2\n{row}\n"
    path = _copy(tmp_path, Path("bad.csv"), text.encode())
    with pytest.raises(LegacyCsvRowError, match=r"bad\.csv") as exc:
        read_legacy_csv(path, "carbon", (1150.0, 1750.0))
    assert exc.value.line_number == 3
    assert "line 3" in str(exc.value)


def test_import_rejects_unexpected_header(tmp_path: Path) -> None:
    text = "name,center,lower,upper,fwhm_lower,fwhm_upper\nd,1,0,2,1,2\n"
    path = _copy(tmp_path, Path("header.csv"), text.encode())
    with pytest.raises(LegacyCsvHeaderError, match=r"header\.csv"):
        read_legacy_csv(path, "carbon", (1150.0, 1750.0))


@pytest.mark.parametrize("window_id", list(FILES))
def test_round_trip_import_export_import(tmp_path: Path, window_id: str) -> None:
    first = _import(window_id)
    exported = _copy(tmp_path, Path("x.csv"), format_legacy_csv(first).encode())
    second = read_legacy_csv(exported, window_id, first.range_cm1)
    # The CSV has no label column: ids survive, display labels do not.
    assert _numbers(second) == _numbers(first)
    assert second.model_copy(update={"peaks": ()}) == first.model_copy(update={"peaks": ()})


@pytest.mark.parametrize("window_id", list(FILES))
def test_export_import_export_is_byte_identical(tmp_path: Path, window_id: str) -> None:
    once = format_legacy_csv(_import(window_id))
    path = _copy(tmp_path, Path("x.csv"), once.encode())
    twice = format_legacy_csv(read_legacy_csv(path, window_id, FILES[window_id][1]))
    assert twice == once


def test_export_header_exact() -> None:
    assert ",".join(HEADER) == HEADER_LINE
    for window_id in FILES:
        assert format_legacy_csv(_import(window_id)).split("\n", 1)[0] == HEADER_LINE


def _rows(text: str) -> list[list[str]]:
    return list(csv.reader(text.removeprefix("﻿").splitlines()))


@pytest.mark.parametrize("window_id", list(FILES))
def test_export_matches_original_rows_modulo_case(window_id: str) -> None:
    source = FILES[window_id][0]
    original = _rows(source.read_bytes().decode("utf-8"))
    exported = _rows(format_legacy_csv(_import(window_id)))
    assert exported[0] == original[0]
    assert len(exported) == len(original)
    for ours, theirs in zip(exported[1:], original[1:], strict=True):
        assert ours[0].lower() == theirs[0].lower()
        assert [float(v) for v in ours[1:]] == [float(v) for v in theirs[1:]]


def test_export_uses_lf_and_no_bom() -> None:
    for window_id in FILES:
        data = format_legacy_csv(_import(window_id)).encode("utf-8")
        assert not data.startswith(BOM)
        assert b"\r" not in data
        assert data.endswith(b"\n")
        assert data.count(b"\n") == 1 + len(_import(window_id).peaks)


def test_imported_fragment_passes_validation() -> None:
    raw: dict[str, Any] = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    doc = copy.deepcopy(raw)
    doc["windows"] = [_import(w).model_dump(mode="json") for w in FILES]
    assert check_rules(PriorsSpec.model_validate(doc)).findings == ()
