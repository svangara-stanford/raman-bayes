"""Info.txt parsing on the five real files and edited copies (DESIGN.md §3)."""

from datetime import datetime
from pathlib import Path

import pytest

from raman_bayes.domain.dataset import ScanInfo
from raman_bayes.domain.errors import (
    InfoFilenameError,
    MissingInfoKeyError,
    NegativeDurationError,
    TimeSeriesNotSupportedError,
)
from raman_bayes.io.nanofinder import parse_info

ROOT = Path(__file__).resolve().parents[2]
INFO = ROOT / "tests" / "fixtures" / "info_txt"
SCANS = ("NMC111_v1", "NMC111_v2", "NMC811_v0", "NMC811_v1", "NMC811_v2")


def _info(scan: str) -> Path:
    return INFO / f"{scan}-Info.txt"


def _edited(tmp_path: Path, scan: str, old: bytes, new: bytes, *, name: str | None = None) -> Path:
    data = _info(scan).read_bytes()
    assert data.count(old) == 1, old
    path = tmp_path / (name or f"{scan}-Info.txt")
    path.write_bytes(data.replace(old, new))
    return path


@pytest.mark.parametrize(
    ("scan", "shape"),
    [
        ("NMC111_v1", (50, 50, 1)),
        ("NMC111_v2", (50, 50, 1)),
        ("NMC811_v0", (50, 25, 1)),
        ("NMC811_v1", (50, 50, 1)),
        ("NMC811_v2", (50, 50, 1)),
    ],
)
def test_grid_dimensions_per_file(scan: str, shape: tuple[int, int, int]) -> None:
    grid = parse_info(_info(scan)).grid
    assert (grid.nx, grid.ny, grid.nz) == shape
    assert grid.allow_partial is False


def test_frame_dimentions_wins_over_frame_stage_for_z() -> None:
    info = parse_info(_info("NMC811_v0"))
    # Frame Stage / Axis : 3 says 6 points from 75.06 to 79.05 um.
    assert info.grid.nz == 1
    assert (info.grid.z.start_um, info.grid.z.stop_um, info.grid.z.step_um) == (42.6788, 42.6788, 0)
    assert (info.grid.x.start_um, info.grid.x.stop_um, info.grid.x.step_um) == (
        28.5139,
        77.5378,
        1.0005,
    )
    assert (info.grid.y.start_um, info.grid.y.stop_um, info.grid.y.step_um) == (
        26.9586,
        50.9488,
        0.9996,
    )


@pytest.mark.parametrize(
    ("scan", "n_accum"),
    [("NMC111_v1", 1), ("NMC111_v2", 5), ("NMC811_v0", 5), ("NMC811_v1", 5), ("NMC811_v2", 10)],
)
def test_n_accum_per_file(scan: str, n_accum: int) -> None:
    assert parse_info(_info(scan)).acq.n_accum == n_accum


@pytest.mark.parametrize("scan", SCANS)
def test_n_series_is_1_in_all_files(scan: str) -> None:
    info = parse_info(_info(scan))
    assert info.n_series == 1
    assert info.n_wn == 1024


def test_n_series_2_copy_is_refused_as_time_series(tmp_path: Path) -> None:
    path = _edited(
        tmp_path,
        "NMC811_v0",
        b"Spectral series number (maximum) : 1",
        b"Spectral series number (maximum) : 2",
    )
    with pytest.raises(TimeSeriesNotSupportedError, match="time series not supported yet"):
        parse_info(path)


def test_exposure_read_from_line_shared_with_cycle_time() -> None:
    assert parse_info(_info("NMC811_v0")).acq.exposure_s == 0.2
    assert parse_info(_info("NMC111_v1")).acq.exposure_s == 1.0


def test_n_accum_read_from_line_shared_with_acquisition_mode(tmp_path: Path) -> None:
    path = _edited(
        tmp_path,
        "NMC811_v0",
        b"Acquisition mode: Accomulate. Number in Accumulation = 5",
        b"Acquisition mode: Accomulate. Number in Accumulation = 7",
    )
    assert parse_info(path).acq.n_accum == 7


def test_center_row_and_track_height_not_required(tmp_path: Path) -> None:
    path = _edited(tmp_path, "NMC811_v0", b"Center Row = 136\r\nTrack Height = 15\r\n", b"")
    assert parse_info(path) == parse_info(_info("NMC811_v0"))
    assert b"Center Row" not in _info("NMC111_v1").read_bytes()
    parse_info(_info("NMC111_v1"))


def test_header_names() -> None:
    instrument = parse_info(_info("NMC811_v0")).instrument
    assert instrument.system == "TII Nanofinder"
    assert instrument.detector == "Andor CCD"
    assert instrument.positioner == "XYZ NI-DACmx"


def test_acquired_at_and_duration() -> None:
    info = parse_info(_info("NMC811_v0"))
    assert info.acquired_at == datetime(2026, 5, 28, 15, 47, 23)
    assert info.acquired_at.tzinfo is None
    assert info.duration_s == 34 * 60 + 50


def test_twelve_hour_clock_across_noon() -> None:
    info = parse_info(_info("NMC811_v1"))
    assert info.acquired_at == datetime(2026, 5, 29, 10, 30, 50)
    assert info.duration_s == 2 * 3600 + 7 * 60 + 38


def test_negative_duration_is_an_error(tmp_path: Path) -> None:
    path = _edited(tmp_path, "NMC811_v0", b"Time End: 4:22:13 PM", b"Time End: 3:22:13 PM")
    with pytest.raises(NegativeDurationError, match=r"NMC811_v0-Info\.txt") as exc:
        parse_info(path)
    assert "midnight" in str(exc.value)


def test_channels_preamble_is_not_a_subsection(tmp_path: Path) -> None:
    # A key in the preamble (before "Channel : 0") must not be read as a channel setting.
    path = _edited(
        tmp_path, "NMC811_v0", b"Channels :1\r\n", b"Channels :1\r\nCentral Pixel = 9\r\n"
    )
    info = parse_info(path)
    assert info.acq.central_pixel == 505
    assert info.acq.head_model == "DU920P_BEX2D"
    assert info.acq.ccd_temp_c == -72.0


def test_raw_stage_extent_and_lf_endings() -> None:
    raw_stage = parse_info(_info("NMC811_v0")).instrument.raw_stage
    assert raw_stage.startswith("Axis : 1\nAxis is present\nAxis in use\n")
    assert raw_stage.endswith("Axis maker : XYZ NI-DACmx\nAxis is fast\n \n")
    assert "\r" not in raw_stage
    assert "Frame Stage" not in raw_stage
    assert "Scan detection channel information" not in raw_stage
    assert "Axis scan points : 6\n" in raw_stage
    assert raw_stage.count("\n \n") == 3
    assert len(raw_stage.splitlines()) == 51


@pytest.mark.parametrize("scan", SCANS)
def test_lf_and_crlf_info_parse_identically(tmp_path: Path, scan: str) -> None:
    lf = tmp_path / f"{scan}-Info.txt"
    lf.write_bytes(_info(scan).read_bytes().replace(b"\r\n", b"\n"))
    assert parse_info(lf) == parse_info(_info(scan))


def test_unknown_keys_are_ignored(tmp_path: Path) -> None:
    path = _edited(
        tmp_path,
        "NMC811_v0",
        b"Function name: Not specified\r\n",
        b"Function name: Not specified\r\nShutter delay = 12 ms\r\nFlag without separator\r\n",
    )
    assert parse_info(path) == parse_info(_info("NMC811_v0"))


@pytest.mark.parametrize(
    ("line", "section", "key"),
    [
        (b"Time End: 4:22:13 PM\r\n", "Frame Header", "Time End"),
        (b"Spectral size (number of pixels) : 1024\r\n", "Frame Dimentions", "Spectral size"),
        (b"Axis scan step, [um]: 0.9996\r\n", "Frame Dimentions", "Axis scan step"),
        (b"Head model = DU920P_BEX2D\r\n", "Channel : 0", "Head model"),
    ],
)
def test_missing_required_key_names_section_and_key(
    tmp_path: Path, line: bytes, section: str, key: str
) -> None:
    path = _edited(tmp_path, "NMC811_v0", line, b"")
    with pytest.raises(MissingInfoKeyError) as exc:
        parse_info(path)
    message = str(exc.value)
    assert section in message
    assert key in message
    assert "NMC811_v0-Info.txt" in message


def test_scan_name_from_info_filename(tmp_path: Path) -> None:
    assert parse_info(_info("NMC811_v0")).scan == "NMC811_v0"
    renamed = tmp_path / "copy-of-scan-Info.txt"
    renamed.write_bytes(_info("NMC811_v0").read_bytes())
    assert parse_info(renamed).scan == "copy-of-scan"
    for bad in ("NMC811_v0.txt", "-Info.txt"):
        path = tmp_path / bad
        path.write_bytes(_info("NMC811_v0").read_bytes())
        with pytest.raises(InfoFilenameError, match=r"-Info\.txt"):
            parse_info(path)


@pytest.mark.parametrize("scan", SCANS)
def test_real_info_scan_info_json_round_trip(scan: str) -> None:
    info = parse_info(_info(scan))
    assert ScanInfo.model_validate_json(info.model_dump_json()) == info
