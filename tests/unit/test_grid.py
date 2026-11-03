"""Placing a scan's spectrum files on its grid (DESIGN.md §3 rule 3, partial grids, §4 order)."""

import numpy as np
import pytest

from raman_bayes.domain.dataset import AxisSpec, GridSpec
from raman_bayes.domain.errors import (
    DuplicateSpectrumError,
    GridError,
    MissingSpectraError,
    OutOfGridSpectrumError,
    SpectrumScanMismatchError,
)
from raman_bayes.io.filename import SpectrumName, parse_filename
from raman_bayes.io.scan_folder import layout_grid

SCAN = "S1"


def _grid(*, allow_partial: bool = False) -> GridSpec:
    # nx = 3, ny = 2, nz = 1
    return GridSpec(
        nx=3,
        ny=2,
        nz=1,
        x=AxisSpec(start_um=0.0, stop_um=2.0, step_um=1.0),
        y=AxisSpec(start_um=0.0, stop_um=0.5, step_um=0.5),
        z=AxisSpec(start_um=7.0, stop_um=7.0, step_um=0.0),
        allow_partial=allow_partial,
    )


def _files(*names: str) -> dict[str, SpectrumName]:
    parsed = {name: parse_filename(name) for name in names}
    return {name: spec for name, spec in parsed.items() if spec is not None}


def _all_names() -> list[str]:
    return [f"{SCAN}-z0y{iy}x{ix}.txt" for iy in range(2) for ix in range(3)]


def test_complete_grid_ok() -> None:
    # Given in reverse order; the layout is raster order (iz, iy, ix) regardless.
    layout = layout_grid(SCAN, _files(*reversed(_all_names())), _grid())
    assert layout.files == tuple(_all_names())
    np.testing.assert_array_equal(
        layout.indices, [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 1], [0, 1, 2]]
    )
    assert layout.indices.dtype == np.int32
    assert layout.mask.shape == (1, 2, 3)
    assert layout.mask.dtype == np.bool_
    assert layout.mask.all()


def test_raster_order_is_numeric_not_lexical() -> None:
    grid = GridSpec(
        nx=12,
        ny=1,
        nz=1,
        x=AxisSpec(start_um=0.0, stop_um=11.0, step_um=1.0),
        y=AxisSpec(start_um=0.0, stop_um=0.0, step_um=0.0),
        z=AxisSpec(start_um=0.0, stop_um=0.0, step_um=0.0),
    )
    names = [f"{SCAN}-z0y0x{ix}.txt" for ix in range(12)]
    assert layout_grid(SCAN, _files(*sorted(names)), grid).files == tuple(names)


def test_missing_file_error() -> None:
    names = _all_names()
    names.remove(f"{SCAN}-z0y1x2.txt")
    with pytest.raises(MissingSpectraError) as exc:
        layout_grid(SCAN, _files(*names), _grid())
    message = str(exc.value)
    assert "5 of 6" in message
    assert "z0y1x2" in message
    assert "--allow-partial-grid" in message


def test_duplicate_index_error() -> None:
    # Leading zeros give the same integer index, so these two files claim one position.
    names = [*_all_names(), f"{SCAN}-z0y01x0.txt"]
    with pytest.raises(DuplicateSpectrumError) as exc:
        layout_grid(SCAN, _files(*names), _grid())
    assert f"{SCAN}-z0y01x0.txt" in str(exc.value)
    assert f"{SCAN}-z0y1x0.txt" in str(exc.value)


def test_out_of_grid_file_error() -> None:
    names = [*_all_names(), f"{SCAN}-z0y0x3.txt"]
    with pytest.raises(OutOfGridSpectrumError) as exc:
        layout_grid(SCAN, _files(*names), _grid())
    assert f"{SCAN}-z0y0x3.txt" in str(exc.value)
    assert "3 x 2 x 1" in str(exc.value)


def test_three_grid_errors_are_distinct() -> None:
    names = _all_names()
    cases = {
        MissingSpectraError: names[1:],
        DuplicateSpectrumError: [*names, f"{SCAN}-z00y0x0.txt"],
        OutOfGridSpectrumError: [*names, f"{SCAN}-z1y0x0.txt"],
    }
    messages = set()
    for error, case in cases.items():
        with pytest.raises(GridError) as exc:
            layout_grid(SCAN, _files(*case), _grid())
        assert type(exc.value) is error
        messages.add(str(exc.value))
    assert len(messages) == 3


def test_allow_partial_records_missing_in_mask() -> None:
    names = _all_names()
    names.remove(f"{SCAN}-z0y0x1.txt")
    names.remove(f"{SCAN}-z0y1x2.txt")
    layout = layout_grid(SCAN, _files(*names), _grid(allow_partial=True))
    assert layout.files == tuple(names)
    expected = np.ones((1, 2, 3), dtype=bool)
    expected[0, 0, 1] = expected[0, 1, 2] = False
    np.testing.assert_array_equal(layout.mask, expected)
    assert len(layout.indices) == 4


def test_allow_partial_still_rejects_duplicate_and_out_of_grid() -> None:
    grid = _grid(allow_partial=True)
    with pytest.raises(DuplicateSpectrumError):
        layout_grid(SCAN, _files(f"{SCAN}-z0y0x0.txt", f"{SCAN}-z0y0x00.txt"), grid)
    with pytest.raises(OutOfGridSpectrumError):
        layout_grid(SCAN, _files(f"{SCAN}-z0y0x0.txt", f"{SCAN}-z0y2x0.txt"), grid)


def test_allow_partial_needs_at_least_one_file() -> None:
    with pytest.raises(MissingSpectraError, match="0 of 6"):
        layout_grid(SCAN, {}, _grid(allow_partial=True))


def test_scan_name_mismatch_names_first_file() -> None:
    names = [*_all_names()[:-2], "S2-z0y1x1.txt", "S0-z0y1x2.txt"]
    with pytest.raises(SpectrumScanMismatchError) as exc:
        layout_grid(SCAN, _files(*names), _grid())
    message = str(exc.value)
    # Sorted by name, S0-... comes before S2-...: the first offending file is named.
    assert "S0-z0y1x2.txt" in message
    assert "S2-z0y1x1.txt" not in message
    assert repr(SCAN) in message
