"""Nanofinder export reader: the section-aware ``<scan>-Info.txt`` parser and spectrum files.

Info.txt layout (DESIGN.md §3 rule 1): two title lines, then the sections in ``_SECTIONS``.
Keys repeat across sections and separators vary (``:``, `` : ``, `` = ``), and two keys share
a line with another setting, so a key is found anywhere in its line and a numeric value is the
first token after its separator. ``Frame Dimentions`` is authoritative for the grid; the
``Frame Stage`` block is kept verbatim for provenance only. Lines are never addressed by number:
the channel subsection is shorter for Full Vertical Binning than for Single Track readout.
"""

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
from pydantic import ValidationError

from raman_bayes.domain.dataset import (
    AcquisitionInfo,
    AxisSpec,
    GridSpec,
    InstrumentInfo,
    ScanInfo,
)
from raman_bayes.domain.errors import (
    InfoFilenameError,
    InfoValueError,
    InvalidScanInfoError,
    MissingInfoKeyError,
    MissingInfoSectionError,
    NegativeDurationError,
    SpectrumLineError,
    SpectrumRowCountError,
    TimeSeriesNotSupportedError,
)

__all__ = [
    "AXIS_TOLERANCE_CM1",
    "INFO_SUFFIX",
    "Spectrum",
    "axis_matches",
    "parse_info",
    "read_spectrum",
]

INFO_SUFFIX = "-Info.txt"
AXIS_TOLERANCE_CM1 = 0.05

_TITLE_LINES = 2  # export timestamp, then "Scan common information"
_HEADER = "Frame Header"
_DIMENSIONS = "Frame Dimentions"  # sic, as the instrument writes it
_STAGE = "Frame Stage"
_CHANNELS = "Scan detection channel information"
_SECTIONS = (_HEADER, "Frame Options", _DIMENSIONS, _STAGE, _CHANNELS)
# "Channels :1" in the preamble must not open a subsection, hence the singular and the digits.
_CHANNEL_0 = re.compile(r"Channel\s*:\s*0")
_SCANNED_POINTS = re.compile(r"Scanned points ([XYZ])\s*:")
_DATE_FORMAT = "%m/%d/%Y"
_TIME_FORMAT = "%I:%M:%S %p"


@dataclass(frozen=True)
class _Section:
    """The lines of one section, with lookups that name the file and section on failure."""

    path: Path
    name: str
    lines: tuple[str, ...]

    def text(self, key: str) -> str:
        pattern = re.compile(rf"(?<!\w){re.escape(key)}\s*[:=]\s*(.*)")
        for line in self.lines:
            match = pattern.search(line)
            if match is not None:
                return match[1].strip()
        raise MissingInfoKeyError(self.path, self.name, key)

    def integer(self, key: str) -> int:
        value = self._token(key)
        try:
            return int(value)
        except ValueError:
            raise InfoValueError(self.path, self.name, key, value) from None

    def real(self, key: str) -> float:
        value = self._token(key)
        try:
            return float(value)
        except ValueError:
            raise InfoValueError(self.path, self.name, key, value) from None

    def moment(self, key: str, fmt: str) -> datetime:
        value = self.text(key)
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            raise InfoValueError(self.path, self.name, key, value) from None

    def _token(self, key: str) -> str:
        # The number right after the separator; the rest of the line may be another setting.
        tokens = self.text(key).split()
        return tokens[0] if tokens else ""


def parse_info(path: Path) -> ScanInfo:
    """Parse one ``<scan>-Info.txt`` (LF or CRLF, optional UTF-8 BOM) into a ScanInfo.

    Raises InfoFilenameError when the name does not end in ``-Info.txt``,
    MissingInfoSectionError or MissingInfoKeyError when a required line is absent,
    InfoValueError when a value is unreadable, TimeSeriesNotSupportedError when the scan has
    more than one spectral series, NegativeDurationError when it ends before it starts, and
    InvalidScanInfoError when the values break a ScanInfo rule.
    """
    scan = path.name.removesuffix(INFO_SUFFIX)
    if not path.name.endswith(INFO_SUFFIX) or not scan:
        raise InfoFilenameError(path)
    sections = _split_sections(path.read_bytes().decode("utf-8-sig").splitlines())
    header = _section(path, sections, _HEADER)
    dims = _section(path, sections, _DIMENSIONS)
    n_series = dims.integer("Spectral series number (maximum)")
    if n_series != 1:
        raise TimeSeriesNotSupportedError(path, n_series)
    acquired_at, duration_s = _timing(header)
    try:
        return ScanInfo(
            scan=scan,
            acquired_at=acquired_at,
            duration_s=duration_s,
            instrument=InstrumentInfo(
                system=header.text("System name"),
                detector=header.text("Detector name"),
                positioner=header.text("Positioner name"),
                raw_stage="".join(f"{line}\n" for line in _section(path, sections, _STAGE).lines),
            ),
            acq=_acquisition(_channel_0(_section(path, sections, _CHANNELS))),
            n_wn=dims.integer("Spectral size (number of pixels)"),
            n_series=n_series,
            grid=_grid(dims),
        )
    except ValidationError as exc:
        raise InvalidScanInfoError(path, exc) from exc


def _split_sections(lines: list[str]) -> dict[str, tuple[str, ...]]:
    bodies: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in lines[_TITLE_LINES:]:
        heading = line.strip().removesuffix(":")
        if heading in _SECTIONS:
            current = bodies.setdefault(heading, [])
        elif current is not None:
            current.append(line)
    return {name: tuple(body) for name, body in bodies.items()}


def _section(path: Path, sections: dict[str, tuple[str, ...]], name: str) -> _Section:
    if name not in sections:
        raise MissingInfoSectionError(path, name)
    return _Section(path, name, sections[name])


def _channel_0(channels: _Section) -> _Section:
    name = f"{channels.name}/Channel : 0"
    starts = [i for i, line in enumerate(channels.lines) if _CHANNEL_0.fullmatch(line.strip())]
    if not starts:
        raise MissingInfoSectionError(channels.path, name)
    return _Section(channels.path, name, channels.lines[starts[0] + 1 :])


def _timing(header: _Section) -> tuple[datetime, float]:
    day = header.moment("Date", _DATE_FORMAT).date()
    start = datetime.combine(day, header.moment("Time Start", _TIME_FORMAT).time())
    end = datetime.combine(day, header.moment("Time End", _TIME_FORMAT).time())
    if end < start:
        raise NegativeDurationError(header.path, header.text("Time Start"), header.text("Time End"))
    return start, (end - start).total_seconds()


def _acquisition(channel: _Section) -> AcquisitionInfo:
    return AcquisitionInfo(
        exposure_s=channel.real("Exposure time, [sec]"),
        n_accum=channel.integer("Number in Accumulation"),
        ccd_temp_c=channel.real("Temperature [grad C]"),
        central_pixel=channel.integer("Central Pixel"),
        head_model=channel.text("Head model"),
    )


def _grid(dims: _Section) -> GridSpec:
    # The per-axis keys repeat once per axis (X, Y, Z), so each axis reads only the lines from
    # its own "Scanned points" line up to the next one.
    starts = {
        match[1]: i
        for i, line in enumerate(dims.lines)
        if (match := _SCANNED_POINTS.match(line.strip())) is not None
    }
    bounds = [*sorted(starts.values()), len(dims.lines)]
    axes: dict[str, tuple[int, AxisSpec]] = {}
    for axis in "XYZ":
        if axis not in starts:
            raise MissingInfoKeyError(dims.path, dims.name, f"Scanned points {axis}")
        stop = next(b for b in bounds if b > starts[axis])
        block = _Section(
            dims.path, f"{dims.name}/Scanned points {axis}", dims.lines[starts[axis] : stop]
        )
        axes[axis] = _axis(block, axis)
    (nx, x), (ny, y), (nz, z) = axes["X"], axes["Y"], axes["Z"]
    return GridSpec(nx=nx, ny=ny, nz=nz, x=x, y=y, z=z)


def _axis(block: _Section, axis: str) -> tuple[int, AxisSpec]:
    n = block.integer(f"Scanned points {axis}")
    if n == 1:
        position = block.real("Axis position, [um]")
        return n, AxisSpec(start_um=position, stop_um=position, step_um=0.0)
    return n, AxisSpec(
        start_um=block.real("Axis scan start, [um]"),
        stop_um=block.real("Axis scan stop, [um]"),
        step_um=block.real("Axis scan step, [um]"),
    )


@dataclass(frozen=True)
class Spectrum:
    """One spectrum file's columns: Raman shift (cm^-1) and raw accumulated counts."""

    wavenumber: npt.NDArray[np.float64]
    intensity: npt.NDArray[np.float64]


def read_spectrum(path: Path, n_wn: int) -> Spectrum:
    """Read a two-column ``shift intensity`` file (LF or CRLF, optional UTF-8 BOM).

    Raises SpectrumRowCountError when the file does not have exactly ``n_wn`` rows and
    SpectrumLineError, naming the 1-based line, when a row is not two numbers.
    """
    lines = path.read_bytes().decode("utf-8-sig").splitlines()
    if len(lines) != n_wn:
        raise SpectrumRowCountError(path, n_wn, len(lines))
    # Parsed here rather than with np.loadtxt, whose error messages number rows inconsistently
    # (from 0 for a bad value, from 1 for a column-count change); the user needs the line.
    rows = [_row(path, n, line) for n, line in enumerate(lines, start=1)]
    values = np.array(rows, dtype=np.float64)
    return Spectrum(wavenumber=values[:, 0].copy(), intensity=values[:, 1].copy())


def _row(path: Path, line_number: int, line: str) -> tuple[float, float]:
    try:
        shift, intensity = (float(field) for field in line.split())
    except ValueError:
        raise SpectrumLineError(path, line_number, line) from None
    return shift, intensity


def axis_matches(
    axis: npt.NDArray[np.float64],
    reference: npt.NDArray[np.float64],
    tolerance_cm1: float = AXIS_TOLERANCE_CM1,
) -> bool:
    """Return whether ``axis`` has the shape of ``reference`` and is within tolerance everywhere."""
    return axis.shape == reference.shape and bool(
        np.allclose(axis, reference, rtol=0.0, atol=tolerance_cm1)
    )
