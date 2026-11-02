"""Two-column spectrum reader and axis check (DESIGN.md §3) on NMC811 and NMC111 files."""

from pathlib import Path

import numpy as np
import pytest

from raman_bayes.domain.errors import SpectrumLineError, SpectrumRowCountError
from raman_bayes.io.nanofinder import AXIS_TOLERANCE_CM1, axis_matches, read_spectrum

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
NMC811 = FIXTURES / "nanofinder_mini" / "NMC811_v0" / "NMC811_v0-z0y0x0.txt"
NMC111 = (
    FIXTURES / "spectra" / "NMC111_v1-z0y0x0.txt",
    FIXTURES / "spectra" / "NMC111_v1-z0y11x0.txt",
)
BOM = b"\xef\xbb\xbf"


def _spectrum_files() -> list[Path]:
    mini = sorted((FIXTURES / "nanofinder_mini" / "NMC811_v0").glob("*-z*.txt"))
    return [*mini, *NMC111]


def test_reads_nmc811_spectrum() -> None:
    spectrum = read_spectrum(NMC811, 1024)
    assert spectrum.wavenumber.shape == spectrum.intensity.shape == (1024,)
    assert spectrum.wavenumber.dtype == spectrum.intensity.dtype == np.float64
    assert spectrum.wavenumber[0] == -41.55539
    assert spectrum.wavenumber[-1] == 1945.85360
    assert spectrum.intensity[0] == 15748.0
    assert spectrum.intensity[-1] == 17191.0


def test_reads_nmc111_spectra_of_differing_byte_sizes() -> None:
    small, large = (read_spectrum(path, 1024) for path in NMC111)
    assert NMC111[0].stat().st_size != NMC111[1].stat().st_size
    np.testing.assert_array_equal(small.wavenumber, large.wavenumber)
    assert small.intensity[0] == 6042.0
    np.testing.assert_array_equal(small.wavenumber, read_spectrum(NMC811, 1024).wavenumber)


@pytest.mark.parametrize("source", [NMC811, *NMC111], ids=lambda p: p.name)
def test_lf_and_crlf_spectra_give_identical_arrays(tmp_path: Path, source: Path) -> None:
    lf = tmp_path / source.name
    lf.write_bytes(source.read_bytes().replace(b"\r\n", b"\n"))
    crlf, plain = read_spectrum(source, 1024), read_spectrum(lf, 1024)
    np.testing.assert_array_equal(crlf.wavenumber, plain.wavenumber)
    np.testing.assert_array_equal(crlf.intensity, plain.intensity)


def test_fixture_spectra_have_no_bom() -> None:
    for path in _spectrum_files():
        assert not path.read_bytes().startswith(BOM), path.name


def test_utf8_bom_is_tolerated(tmp_path: Path) -> None:
    with_bom = tmp_path / NMC811.name
    with_bom.write_bytes(BOM + NMC811.read_bytes())
    np.testing.assert_array_equal(
        read_spectrum(with_bom, 1024).intensity, read_spectrum(NMC811, 1024).intensity
    )


@pytest.mark.parametrize("n_rows", [1023, 1025])
def test_wrong_row_count_names_the_file(tmp_path: Path, n_rows: int) -> None:
    lines = NMC811.read_bytes().splitlines(keepends=True)
    path = tmp_path / NMC811.name
    path.write_bytes(b"".join((lines * 2)[:n_rows]))
    with pytest.raises(SpectrumRowCountError, match=r"NMC811_v0-z0y0x0\.txt") as exc:
        read_spectrum(path, 1024)
    assert "1024" in str(exc.value)
    assert str(n_rows) in str(exc.value)


@pytest.mark.parametrize(
    "bad", [b"-37.17443\tabc\r\n", b"-37.17443\r\n", b"-37.17443\t1.0\t2.0\r\n", b"\r\n"]
)
def test_non_numeric_line_names_file_and_line(tmp_path: Path, bad: bytes) -> None:
    lines = NMC811.read_bytes().splitlines(keepends=True)
    lines[2] = bad
    path = tmp_path / NMC811.name
    path.write_bytes(b"".join(lines))
    with pytest.raises(SpectrumLineError, match=r"NMC811_v0-z0y0x0\.txt") as exc:
        read_spectrum(path, 1024)
    assert exc.value.line_number == 3
    assert "line 3" in str(exc.value)


def test_axis_within_tolerance_matches() -> None:
    reference = read_spectrum(NMC811, 1024).wavenumber
    assert AXIS_TOLERANCE_CM1 == 0.05
    assert axis_matches(reference, reference)
    assert axis_matches(reference + 0.049, reference)
    shifted = reference.copy()
    shifted[500] -= 0.049
    assert axis_matches(shifted, reference)


def test_axis_beyond_tolerance_is_a_mismatch() -> None:
    reference = read_spectrum(NMC811, 1024).wavenumber
    assert not axis_matches(reference + 0.051, reference)
    shifted = reference.copy()
    shifted[500] -= 0.051
    assert not axis_matches(shifted, reference)
    assert not axis_matches(reference[:-1], reference)
