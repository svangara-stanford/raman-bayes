"""Errors a user can act on, raised by io/ and storage/ and reported by the CLI (DESIGN.md §10).

They live in domain/ because cli/ may import only app/, domain/, config/ and logging. Every
message names the file and what is wrong with it; InputError and its subclasses exit 2.
"""

import re
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

__all__ = [
    "AxisMismatchError",
    "CatalogError",
    "ConfigError",
    "CountTooLargeError",
    "DatasetStoreError",
    "DuplicateSpectrumError",
    "FilenamePatternError",
    "FilenamePatternGroupsError",
    "FilenamePatternSyntaxError",
    "FolderScanMismatchError",
    "GridError",
    "IllegalTransitionError",
    "InfoFileCountError",
    "InfoFileError",
    "InfoFilenameError",
    "InfoValueError",
    "InputError",
    "InvalidEngineError",
    "InvalidProfileError",
    "InvalidScanInfoError",
    "LegacyCsvError",
    "LegacyCsvHeaderError",
    "LegacyCsvRowError",
    "MalformedVariableError",
    "MissingInfoKeyError",
    "MissingInfoSectionError",
    "MissingSpectraError",
    "NegativeDurationError",
    "NoScanFolderError",
    "OutOfGridSpectrumError",
    "RamanBayesError",
    "ScanFolderError",
    "SpectrumFileError",
    "SpectrumLineError",
    "SpectrumRowCountError",
    "SpectrumScanMismatchError",
    "StaleStatusError",
    "TimeSeriesNotSupportedError",
    "UnknownEngineError",
    "UnknownProfileError",
    "UnknownRunError",
    "UnknownTaskError",
    "UnsetVariableError",
    "UnsupportedCatalogError",
]


_SHOWN = 5  # names listed in a message before "and N more"


def _some(names: Sequence[str], total: int | None = None) -> str:
    total = len(names) if total is None else total
    shown = ", ".join(names[:_SHOWN])
    return f"{shown} and {total - _SHOWN} more" if total > _SHOWN else shown


class RamanBayesError(Exception):
    """Base of every error the CLI reports as a message rather than a traceback."""


class InputError(RamanBayesError):
    """An input file or setting is invalid."""


# ---------------------------------------------------------------- filename patterns


class FilenamePatternError(InputError):
    """A spectrum filename pattern cannot be used."""


class FilenamePatternSyntaxError(FilenamePatternError):
    """The pattern is not a valid regular expression."""

    def __init__(self, regex: str, error: re.error) -> None:
        super().__init__(f"filename pattern {regex!r} is not a valid regular expression: {error}")


class FilenamePatternGroupsError(FilenamePatternError):
    """The pattern lacks named groups that every spectrum filename must provide."""

    def __init__(self, regex: str, missing: Sequence[str]) -> None:
        self.missing = tuple(missing)
        super().__init__(
            f"filename pattern {regex!r} lacks the named group(s) {', '.join(self.missing)}; "
            "it needs scan, iz, iy and ix"
        )


# ---------------------------------------------------------------- Info.txt


class InfoFileError(InputError):
    """An Info.txt file cannot be read into a ScanInfo."""


class InfoFilenameError(InfoFileError):
    """The file is not named ``<scan>-Info.txt``."""

    def __init__(self, path: Path) -> None:
        super().__init__(f"{path}: expected a file named <scan>-Info.txt")


class MissingInfoSectionError(InfoFileError):
    """A required section heading is absent."""

    def __init__(self, path: Path, section: str) -> None:
        super().__init__(f"{path}: no '{section}' section")


class MissingInfoKeyError(InfoFileError):
    """A required key is absent from its section."""

    def __init__(self, path: Path, section: str, key: str) -> None:
        super().__init__(f"{path}: section '{section}' has no '{key}' line")


class InfoValueError(InfoFileError):
    """A key's value cannot be read as the expected type."""

    def __init__(self, path: Path, section: str, key: str, value: str) -> None:
        super().__init__(f"{path}: section '{section}', '{key}' has unreadable value {value!r}")


class NegativeDurationError(InfoFileError):
    """``Time End`` is earlier than ``Time Start``."""

    def __init__(self, path: Path, start: str, end: str) -> None:
        super().__init__(
            f"{path}: Time End {end} is before Time Start {start}; "
            "scans that cross midnight are not supported"
        )


class TimeSeriesNotSupportedError(InfoFileError):
    """The scan has more than one spectrum per position."""

    def __init__(self, path: Path, n_series: int) -> None:
        super().__init__(
            f"{path}: time series not supported yet "
            f"(Spectral series number (maximum) is {n_series}, expected 1)"
        )


class InvalidScanInfoError(InfoFileError):
    """The values read are individually readable but inconsistent."""

    def __init__(self, path: Path, error: ValidationError) -> None:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in e['loc']) or 'scan info'}: {e['msg']}"
            for e in error.errors()
        )
        super().__init__(f"{path}: {problems}")


# ---------------------------------------------------------------- spectrum files


class SpectrumFileError(InputError):
    """A spectrum file cannot be read."""


class SpectrumRowCountError(SpectrumFileError):
    """The file does not have one row per spectral pixel."""

    def __init__(self, path: Path, expected: int, found: int) -> None:
        super().__init__(f"{path}: expected {expected} rows, found {found}")


class SpectrumLineError(SpectrumFileError):
    """A line is not two numbers (shift and intensity)."""

    def __init__(self, path: Path, line_number: int, line: str) -> None:
        self.line_number = line_number
        super().__init__(f"{path}, line {line_number}: expected two numbers, got {line!r}")


class AxisMismatchError(SpectrumFileError):
    """Spectrum files whose wavenumber axis differs from the dataset axis beyond tolerance."""

    def __init__(self, folder: Path, files: Sequence[str], tolerance_cm1: float) -> None:
        super().__init__(
            f"{folder}: {len(files)} spectrum file(s) have a wavenumber axis more than "
            f"{tolerance_cm1} cm^-1 from the first file's: {_some(files)}; "
            "use --allow-axis-mismatch to ingest them with the axis_mismatch flag"
        )


class CountTooLargeError(SpectrumFileError):
    """A count that float32 cannot store exactly."""

    def __init__(self, file: str, value: float, limit: int) -> None:
        super().__init__(
            f"{file}: count {value:g} has magnitude {limit} or more, "
            "so the dataset's float32 intensities could not store every count exactly"
        )


# ---------------------------------------------------------------- scan folders and grids


class ScanFolderError(InputError):
    """A path is not a scan folder, or a scan folder is not laid out as ingest expects."""


class NoScanFolderError(ScanFolderError):
    """The path neither is a scan folder nor holds any."""

    def __init__(self, path: Path) -> None:
        super().__init__(
            f"{path}: no scan folder found; expected a folder holding <scan>-Info.txt, "
            "or a folder of such folders"
        )


class InfoFileCountError(ScanFolderError):
    """A scan folder holds more than one Info.txt."""

    def __init__(self, folder: Path, names: Sequence[str]) -> None:
        super().__init__(f"{folder}: expected one <scan>-Info.txt, found {', '.join(names)}")


class FolderScanMismatchError(ScanFolderError):
    """The folder name is not the scan named by its Info.txt."""

    def __init__(self, folder: Path, info_name: str, scan: str) -> None:
        super().__init__(
            f"{folder / info_name}: names scan {scan!r}, but its folder is {folder.name!r}; "
            "the folder must be named after the scan"
        )


class GridError(InputError):
    """A scan's spectrum files do not fit its Frame Dimentions grid."""


class SpectrumScanMismatchError(GridError):
    """A spectrum filename names another scan."""

    def __init__(self, scan: str, file: str, file_scan: str) -> None:
        super().__init__(f"scan {scan!r}: spectrum file {file} belongs to scan {file_scan!r}")


class OutOfGridSpectrumError(GridError):
    """Spectrum files whose indices lie outside the grid."""

    def __init__(self, scan: str, files: Sequence[str], shape: tuple[int, int, int]) -> None:
        nx, ny, nz = shape
        super().__init__(
            f"scan {scan!r}: {len(files)} spectrum file(s) lie outside the "
            f"{nx} x {ny} x {nz} grid: {_some(files)}"
        )


class DuplicateSpectrumError(GridError):
    """Two or more spectrum files claim one grid position."""

    def __init__(self, scan: str, position: str, files: Sequence[str]) -> None:
        super().__init__(
            f"scan {scan!r}: grid position {position} has {len(files)} files: {', '.join(files)}"
        )


class MissingSpectraError(GridError):
    """Grid positions without a spectrum file, when the grid must be complete."""

    def __init__(self, scan: str, found: int, size: int, missing: Sequence[str]) -> None:
        hint = "; use --allow-partial-grid to ingest a partial grid" if found else ""
        super().__init__(
            f"scan {scan!r}: found {found} of {size} spectrum files; "
            f"no file for {_some(missing, total=size - found)}{hint}"
        )


# ---------------------------------------------------------------- dataset store


class DatasetStoreError(RamanBayesError):
    """A dataset.zarr store is missing or inconsistent."""

    def __init__(self, path: Path, problem: str) -> None:
        super().__init__(f"{path}: {problem}")


# ---------------------------------------------------------------- engines


class UnknownEngineError(InputError):
    """No installed engine has this name."""

    def __init__(self, name: str, available: Sequence[str]) -> None:
        installed = ", ".join(available) or "none"
        super().__init__(f"no engine named {name!r}; installed engines: {installed}")


class InvalidEngineError(InputError):
    """An engine entry point does not provide a PeakEngine."""

    def __init__(self, name: str, target: str) -> None:
        super().__init__(f"engine {name!r} ({target}) does not provide a peak engine")


# ---------------------------------------------------------------- legacy priors CSV


class LegacyCsvError(InputError):
    """A legacy priors CSV cannot be read."""


class LegacyCsvHeaderError(LegacyCsvError):
    """The first row is not the legacy header."""

    def __init__(self, path: Path, expected: Sequence[str]) -> None:
        super().__init__(f"{path}, line 1: expected the header {','.join(expected)!r}")


class LegacyCsvRowError(LegacyCsvError):
    """A row is not a peak name followed by five numbers, or repeats a peak."""

    def __init__(self, path: Path, line_number: int, problem: str) -> None:
        self.line_number = line_number
        super().__init__(f"{path}, line {line_number}: {problem}")


# ---------------------------------------------------------------- settings and profiles


class ConfigError(InputError):
    """A profile cannot be found, expanded or validated."""


class UnknownProfileError(ConfigError):
    """No profile file exists for the given name or path."""

    def __init__(self, profile: str, searched: Sequence[str]) -> None:
        super().__init__(f"profile {profile!r} not found; looked in: {'; '.join(searched)}")


class UnsetVariableError(ConfigError):
    """A profile value names an environment variable that is not set."""

    def __init__(self, variable: str, key: str, source: str) -> None:
        super().__init__(f"{source}: {key} uses ${{{variable}}}, but {variable} is not set")


class MalformedVariableError(ConfigError):
    """A profile value contains ``${`` that is not a complete ``${NAME}``."""

    def __init__(self, value: str, key: str, source: str) -> None:
        super().__init__(
            f"{source}: {key} value {value!r} has a '${{' that is not a complete ${{NAME}}"
        )


class InvalidProfileError(ConfigError):
    """A profile file is not YAML, or its settings break the settings model."""

    def __init__(self, source: str, problems: Sequence[str]) -> None:
        super().__init__(f"{source}: {'; '.join(problems)}")


# ---------------------------------------------------------------- catalog


class CatalogError(RamanBayesError):
    """The catalog refused an operation or cannot be used."""


class UnsupportedCatalogError(CatalogError):
    """The catalog file was written by an incompatible version of raman-bayes."""

    def __init__(self, path: Path, version: int, supported: int) -> None:
        super().__init__(
            f"{path}: catalog schema version {version}, but this raman-bayes reads {supported}"
        )


class IllegalTransitionError(CatalogError):
    """A run status change that the run lifecycle does not allow."""

    def __init__(self, run_id: str, current: str, requested: str) -> None:
        super().__init__(f"run {run_id}: cannot go from {current} to {requested}")


class StaleStatusError(CatalogError):
    """The run's status changed since the caller read it."""

    def __init__(self, run_id: str, expected: str, actual: str) -> None:
        super().__init__(f"run {run_id}: expected status {expected}, but it is {actual}")


class UnknownRunError(CatalogError):
    """No run with this id is in the catalog."""

    def __init__(self, run_id: str) -> None:
        super().__init__(f"no run {run_id!r} in the catalog")


class UnknownTaskError(CatalogError):
    """The run has no task with this index."""

    def __init__(self, run_id: str, task_index: int) -> None:
        super().__init__(f"run {run_id} has no task {task_index}")
