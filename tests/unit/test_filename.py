"""Spectrum filename parsing (DESIGN.md §3 filename grammar)."""

import pytest

from raman_bayes.domain.errors import FilenamePatternGroupsError, FilenamePatternSyntaxError
from raman_bayes.io.filename import SpectrumName, compile_pattern, parse_filename


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("NMC811_v0-z0y24x49.txt", SpectrumName("NMC811_v0", 0, 24, 49)),
        ("NMC111_v2-z0y0x0.txt", SpectrumName("NMC111_v2", 0, 0, 0)),
        ("sample-a-b-z3y10x7.txt", SpectrumName("sample-a-b", 3, 10, 7)),
        ("odd-z9-z1y2x3.txt", SpectrumName("odd-z9", 1, 2, 3)),
    ],
)
def test_parse_valid_names(name: str, expected: SpectrumName) -> None:
    assert parse_filename(name) == expected


@pytest.mark.parametrize(
    "name",
    [
        "NMC811_v0-z0y0x0.TXT",
        "NMC811_v0-z0y0.txt",
        "NMC811_v0-z0x0y0.txt",
        "NMC811_v0-z0y0x0.txt.bak",
        "NMC811_v0-z0y0x0.txt\n",
        "-z0y0x0.txt",
        "NMC811_v0-z-1y0x0.txt",
        "NMC811_v0-zay0x0.txt",
        "",
    ],
)
def test_parse_rejects_malformed_names(name: str) -> None:
    assert parse_filename(name) is None


def test_info_file_is_not_a_spectrum_name() -> None:
    assert parse_filename("NMC811_v0-Info.txt") is None


def test_custom_pattern_for_another_instrument() -> None:
    pattern = compile_pattern(r"(?P<scan>\w+)_X(?P<ix>\d+)_Y(?P<iy>\d+)_Z(?P<iz>\d+)\.csv")
    assert parse_filename("map01_X3_Y4_Z0.csv", pattern) == SpectrumName("map01", 0, 4, 3)
    assert parse_filename("NMC811_v0-z0y0x0.txt", pattern) is None


def test_custom_pattern_missing_a_named_group_is_rejected() -> None:
    with pytest.raises(FilenamePatternGroupsError, match="iz") as exc:
        compile_pattern(r"(?P<scan>\w+)_X(?P<ix>\d+)_Y(?P<iy>\d+)\.csv")
    assert exc.value.missing == ("iz",)
    with pytest.raises(FilenamePatternSyntaxError):
        compile_pattern(r"(?P<scan>\w+")


def test_leading_zero_index_parses_to_same_integer() -> None:
    padded = parse_filename("NMC811_v0-z00y07x012.txt")
    assert padded == parse_filename("NMC811_v0-z0y7x12.txt") == SpectrumName("NMC811_v0", 0, 7, 12)
