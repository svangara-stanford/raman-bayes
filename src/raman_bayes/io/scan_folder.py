"""Scan folders (DESIGN.md §3): find them, list their files, and place spectra on the grid.

A scan folder holds ``<scan>-Info.txt`` and one file per grid position, and is named after the
scan. Ingest accepts a scan folder or a folder of scan folders. Placement checks, in this order,
that every spectrum belongs to the scan, lies inside the grid, has a position to itself, and,
unless the grid may be partial, that every position has a file; each failure is its own error.
"""

import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from raman_bayes.domain.dataset import GridSpec
from raman_bayes.domain.errors import (
    DuplicateSpectrumError,
    FolderScanMismatchError,
    InfoFileCountError,
    MissingSpectraError,
    NoScanFolderError,
    OutOfGridSpectrumError,
    SpectrumScanMismatchError,
)
from raman_bayes.io.filename import DEFAULT_PATTERN, SpectrumName, parse_filename
from raman_bayes.io.nanofinder import INFO_SUFFIX

__all__ = [
    "GridLayout",
    "ScanListing",
    "find_scan_folders",
    "layout_grid",
    "list_scan_folder",
]

_MISSING_SHOWN = 5  # missing positions named in the error; the rest are counted

Position = tuple[int, int, int]  # (iz, iy, ix)


@dataclass(frozen=True)
class ScanListing:
    """The files of one scan folder.

    ``spectra`` maps each spectrum filename to its parsed name; ``ignored`` lists the other
    files, which are neither the Info.txt nor hidden.
    """

    folder: Path
    info_path: Path
    spectra: Mapping[str, SpectrumName]
    ignored: tuple[str, ...]


@dataclass(frozen=True)
class GridLayout:
    """Spectrum files in raster order (iz, iy, ix), with their indices and the grid mask.

    ``indices`` is ``[n, 3]`` int32 (iz, iy, ix); ``mask`` is ``[nz, ny, nx]`` and True where a
    file was found.
    """

    files: tuple[str, ...]
    indices: npt.NDArray[np.int32]
    mask: npt.NDArray[np.bool_]


def find_scan_folders(path: Path) -> tuple[Path, ...]:
    """Return ``path`` if it is a scan folder, else its scan subfolders sorted by name.

    Raises NoScanFolderError when ``path`` is not a folder or holds no scan folder.
    """
    if not path.is_dir():
        raise NoScanFolderError(path)
    if _holds_info(path):
        return (path,)
    folders = tuple(sorted(p for p in path.iterdir() if p.is_dir() and _holds_info(p)))
    if not folders:
        raise NoScanFolderError(path)
    return folders


def _holds_info(folder: Path) -> bool:
    return any(folder.glob(f"*{INFO_SUFFIX}"))


def list_scan_folder(folder: Path, pattern: re.Pattern[str] = DEFAULT_PATTERN) -> ScanListing:
    """List the Info.txt and spectrum files of ``folder``; ``pattern`` from ``compile_pattern``.

    Raises InfoFileCountError unless there is exactly one Info.txt, and FolderScanMismatchError
    when the folder is not named after its scan.
    """
    infos = sorted(folder.glob(f"*{INFO_SUFFIX}"))
    if len(infos) != 1:
        raise InfoFileCountError(folder, [p.name for p in infos])
    (info,) = infos
    scan = info.name.removesuffix(INFO_SUFFIX)
    if scan != folder.name:
        raise FolderScanMismatchError(folder, info.name, scan)
    spectra: dict[str, SpectrumName] = {}
    ignored: list[str] = []
    for path in sorted(folder.iterdir()):
        if path == info or path.name.startswith("."):
            continue
        name = parse_filename(path.name, pattern)
        if name is None or not path.is_file():
            ignored.append(path.name)
        else:
            spectra[path.name] = name
    return ScanListing(folder=folder, info_path=info, spectra=spectra, ignored=tuple(ignored))


def layout_grid(scan: str, spectra: Mapping[str, SpectrumName], grid: GridSpec) -> GridLayout:
    """Place ``spectra`` (filename -> parsed name) on ``grid`` in raster order.

    Raises SpectrumScanMismatchError naming the first file (by name) of another scan,
    OutOfGridSpectrumError, DuplicateSpectrumError, and MissingSpectraError when a position has
    no file and ``grid.allow_partial`` is false, or when there are no files at all.
    """
    others = sorted(name for name, parsed in spectra.items() if parsed.scan != scan)
    if others:
        raise SpectrumScanMismatchError(scan, others[0], spectra[others[0]].scan)
    shape = (grid.nz, grid.ny, grid.nx)
    outside = sorted(
        name
        for name, p in spectra.items()
        if not (p.iz < grid.nz and p.iy < grid.ny and p.ix < grid.nx)
    )
    if outside:
        raise OutOfGridSpectrumError(scan, outside, (grid.nx, grid.ny, grid.nz))
    by_position: defaultdict[Position, list[str]] = defaultdict(list)
    for name, p in sorted(spectra.items()):
        by_position[(p.iz, p.iy, p.ix)].append(name)
    order = sorted(by_position)
    for position in order:
        if len(by_position[position]) > 1:
            raise DuplicateSpectrumError(scan, _position(position), by_position[position])
    mask = np.zeros(shape, dtype=np.bool_)
    for position in order:
        mask[position] = True
    if not order or (len(order) < grid.size and not grid.allow_partial):
        missing = [_position(tuple(int(i) for i in p)) for p in np.argwhere(~mask)[:_MISSING_SHOWN]]
        raise MissingSpectraError(scan, len(order), grid.size, missing)
    return GridLayout(
        files=tuple(by_position[p][0] for p in order),
        indices=np.array(order, dtype=np.int32).reshape(-1, 3),
        mask=mask,
    )


def _position(position: tuple[int, ...]) -> str:
    iz, iy, ix = position
    return f"z{iz}y{iy}x{ix}"
