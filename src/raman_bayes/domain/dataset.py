"""Dataset models (DESIGN.md §3, §4): QC bits, filename grammar, ScanInfo and the manifest.

Manifest keys follow the §3 field table. Parsing Info.txt and matching filenames is io/'s job;
this module defines the shapes, constants and the §3/§4 consistency rules.
"""

from datetime import datetime
from enum import IntFlag
from typing import Annotated, Self

from pydantic import AwareDatetime, Field, field_validator, model_validator
from pydantic_core import PydanticCustomError

from raman_bayes.domain._base import DomainModel

__all__ = [
    "FILENAME_RE",
    "MANIFEST_SCHEMA_VERSION",
    "AcquisitionInfo",
    "AxisSpec",
    "DatasetManifest",
    "FilenamePattern",
    "GridSpec",
    "InstrumentInfo",
    "QCFlag",
    "ScanInfo",
]

FILENAME_RE = r"^(?P<scan>.+?)-z(?P<iz>\d+)y(?P<iy>\d+)x(?P<ix>\d+)\.txt$"
MANIFEST_SCHEMA_VERSION = "1.0"


class QCFlag(IntFlag):
    """Per-spectrum QC bits, stored as ``uint16`` in ``qc_flags``."""

    NAN_OR_INF = 1 << 0
    SATURATED = 1 << 1
    AXIS_MISMATCH = 1 << 2
    NEGATIVE = 1 << 3


class AxisSpec(DomainModel):
    """One ``Frame Dimentions`` axis. A single-point axis has start == stop == position, step 0."""

    start_um: float
    stop_um: float
    step_um: float


class GridSpec(DomainModel):
    """Scan grid from ``Frame Dimentions`` (authoritative over ``Frame Stage``).

    Each axis must agree with its point count: one point means start == stop and step 0;
    more means a non-zero step with ``start + (n - 1) * step`` within half a step of stop,
    because Info.txt rounds to 4 decimals.
    """

    nx: Annotated[int, Field(ge=1)]
    ny: Annotated[int, Field(ge=1)]
    nz: Annotated[int, Field(ge=1)]
    x: AxisSpec
    y: AxisSpec
    z: AxisSpec
    allow_partial: bool = False

    @property
    def size(self) -> int:
        """Number of grid positions, ``nx * ny * nz``."""
        return self.nx * self.ny * self.nz

    @model_validator(mode="after")
    def _axes_match_counts(self) -> Self:
        for name, n, axis in (
            ("x", self.nx, self.x),
            ("y", self.ny, self.y),
            ("z", self.nz, self.z),
        ):
            _check_axis(name, n, axis)
        return self


def _check_axis(name: str, n: int, axis: AxisSpec) -> None:
    context = {"axis": name, "n": n}
    if n == 1:
        if not (axis.step_um == 0 and axis.start_um == axis.stop_um):
            raise PydanticCustomError(
                "single_point_axis",
                "axis {axis} has 1 point, so start_um must equal stop_um and step_um must be 0",
                context,
            )
        return
    if axis.step_um == 0:
        raise PydanticCustomError(
            "multi_point_axis_zero_step",
            "axis {axis} has {n} points, so step_um must not be 0",
            context,
        )
    # Written as "not <=" so that NaN fails too.
    if not abs(axis.start_um + (n - 1) * axis.step_um - axis.stop_um) <= abs(axis.step_um) / 2:
        raise PydanticCustomError(
            "axis_stop_mismatch",
            "axis {axis}: start_um + (n - 1) * step_um is not within half a step of stop_um",
            context,
        )


class InstrumentInfo(DomainModel):
    """Instrument names from ``Frame Header``.

    ``raw_stage`` is the verbatim ``Frame Stage`` block text, kept for provenance only. Text
    rather than a mapping because the block repeats the same keys under each ``Axis : N``
    subsection and has keyless flag lines (``Axis is present``).
    """

    system: str
    detector: str
    positioner: str
    raw_stage: str


class AcquisitionInfo(DomainModel):
    """Detector channel settings from ``Scan detection channel information``."""

    exposure_s: Annotated[float, Field(gt=0)]
    n_accum: Annotated[int, Field(ge=1)]
    ccd_temp_c: float
    central_pixel: Annotated[int, Field(ge=0)]
    head_model: str


class FilenamePattern(DomainModel):
    """The spectrum filename grammar; ``regex`` needs groups ``scan``, ``iz``, ``iy``, ``ix``."""

    regex: str = FILENAME_RE


class ScanInfo(DomainModel):
    """Everything one ``<scan>-Info.txt`` yields, and nothing read from spectrum files (§3).

    ``scan`` is the filename without ``-Info.txt``. ``acquired_at`` is naive local time, as
    Info.txt records it. ``n_series`` must be 1: time series are not supported yet.
    """

    scan: Annotated[str, Field(min_length=1)]
    acquired_at: datetime
    duration_s: Annotated[float, Field(ge=0)]
    instrument: InstrumentInfo
    acq: AcquisitionInfo
    n_wn: Annotated[int, Field(ge=1)]
    n_series: int
    grid: GridSpec

    @field_validator("n_series")
    @classmethod
    def _single_series(cls, n_series: int) -> int:
        if n_series != 1:
            raise PydanticCustomError(
                "time_series_not_supported",
                "n_series is {n_series}, but time series are not supported yet (must be 1)",
                {"n_series": n_series},
            )
        return n_series


class DatasetManifest(ScanInfo):
    """The full manifest stored in ``dataset.zarr`` attrs and the catalog (§4, §8.5).

    ScanInfo plus identity and provenance, so the §3 keys stay flat. ``n_spectra`` is the
    number of files found: equal to the grid size, or at most it when ``grid.allow_partial``.
    ``created_at`` is pipeline-generated and must be timezone-aware (written in UTC);
    ``acquired_at`` stays naive because Info.txt records local time with no zone.
    """

    schema_version: str = MANIFEST_SCHEMA_VERSION
    dataset_id: str
    sample: str | None
    n_spectra: Annotated[int, Field(ge=1)]
    filename_pattern: FilenamePattern = FilenamePattern()
    content_hash: str
    created_at: AwareDatetime

    @model_validator(mode="after")
    def _n_spectra_fits_grid(self) -> Self:
        context = {"n_spectra": self.n_spectra, "size": self.grid.size}
        if self.grid.allow_partial:
            if self.n_spectra > self.grid.size:
                raise PydanticCustomError(
                    "n_spectra_exceeds_grid",
                    "n_spectra is {n_spectra}, more than the {size} grid positions",
                    context,
                )
        elif self.n_spectra != self.grid.size:
            raise PydanticCustomError(
                "n_spectra_grid_mismatch",
                "n_spectra is {n_spectra}, but the grid has {size} positions "
                "and grid.allow_partial is false",
                context,
            )
        return self
