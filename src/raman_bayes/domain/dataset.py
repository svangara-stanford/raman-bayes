"""Dataset models (DESIGN.md §3, §4): QC bits, filename grammar and the dataset manifest.

Manifest keys follow the §3 field table. Parsing Info.txt and matching filenames is io/'s job;
this module only defines the shapes and constants.
"""

from datetime import datetime
from enum import IntFlag

from pydantic import AwareDatetime

from raman_bayes.domain._base import DomainModel

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
    """Scan grid from ``Frame Dimentions`` (authoritative over ``Frame Stage``)."""

    nx: int
    ny: int
    nz: int
    x: AxisSpec
    y: AxisSpec
    z: AxisSpec
    allow_partial: bool = False


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

    exposure_s: float
    n_accum: int
    ccd_temp_c: float
    central_pixel: int
    head_model: str


class FilenamePattern(DomainModel):
    regex: str = FILENAME_RE


class ScanInfo(DomainModel):
    """Provisional: everything one ``<scan>-Info.txt`` yields.

    DESIGN.md §2 names ScanInfo without defining it; the field list is inferred from §3.
    """

    scan: str
    acquired_at: datetime
    duration_s: float
    instrument: InstrumentInfo
    acq: AcquisitionInfo
    n_wn: int
    n_series: int
    grid: GridSpec


class DatasetManifest(ScanInfo):
    """The full manifest stored in ``dataset.zarr`` attrs and the catalog (§4, §8.5).

    Extends ScanInfo so the §3 keys stay flat. The identity and provenance fields added here
    (``dataset_id``, ``sample``, ``content_hash``, ``created_at``) are provisional placement.
    ``created_at`` is pipeline-generated and must be timezone-aware (written in UTC);
    ``acquired_at`` stays naive because Info.txt records local time with no zone.
    """

    schema_version: str = MANIFEST_SCHEMA_VERSION
    dataset_id: str
    sample: str | None
    n_spectra: int
    filename_pattern: FilenamePattern = FilenamePattern()
    content_hash: str
    created_at: AwareDatetime
