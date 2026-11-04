"""``raman export`` (DESIGN.md §8.4): legacy per-window CSVs and the flat peaks CSV."""

import csv
import io
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
import yaml
from typer.testing import CliRunner

from raman_bayes.app.collect import CollectRequest, collect_run
from raman_bayes.app.plan import plan_run
from raman_bayes.app.run_task import TaskRequest, run_task
from raman_bayes.cli.main import app
from raman_bayes.cli.output import CliEnvelope, ExitCode
from raman_bayes.config.settings import PathsSettings, load_settings
from raman_bayes.domain.priors import PriorsSpec, Window
from raman_bayes.domain.results import PEAKS_SCHEMA
from raman_bayes.domain.run import RunConfig, RunStatus
from raman_bayes.io.results_csv import LegacyOptions, legacy_csv
from raman_bayes.storage.catalog import Catalog

PRIORS = Path(__file__).parents[1] / "fixtures" / "priors" / "design_example.yaml"
# No counts CSV from the previous pipeline was found, so the header is this committed string
# (DESIGN.md §11), with the recipe's lower-case peak ids (C11, A9).
LEGACY_HEADERS = {
    "lithiation": "x,y,u1 center,u1 height,u1 width,u1 confidence,"
    "u2 center,u2 height,u2 width,u2 confidence",
    "carbon": "x,y,d center,d height,d width,d confidence,g center,g height,g width,g confidence",
}
runner = CliRunner()


def design_window(window_id: str) -> Window:
    recipe = PriorsSpec.model_validate(yaml.safe_load(PRIORS.read_text(encoding="utf-8")))
    return next(w for w in recipe.windows if w.id == window_id)


@pytest.fixture
def run_id(mini_store: Path, raman_paths: PathsSettings) -> str:
    """Plan, run and collect the mini dataset; return the collected run's id."""
    config = RunConfig(
        dataset_id=mini_store.name.removesuffix(".zarr"),
        priors_path=PRIORS,
        engine="mock",
        spectra_per_task=16,
        seed=5,
        profile="local",
    )
    planned = plan_run(config, load_settings())
    assert planned.plan is not None
    run = planned.plan.run_id
    Catalog(raman_paths.catalog).transition(run, RunStatus.PLANNED, RunStatus.SUBMITTED)
    for task in (0, 1):
        run_task(TaskRequest(raman_paths.runs_root / run, task))
    collect_run(CollectRequest(run), load_settings())
    return run


def _export(run: str, *options: str) -> tuple[int, CliEnvelope]:
    result = runner.invoke(app, ["export", run, *options, "--json"])
    (line,) = result.stdout.splitlines()
    return result.exit_code, CliEnvelope.model_validate_json(line)


def _files(env: CliEnvelope) -> list[Path]:
    assert env.data is not None
    return [Path(f) for f in env.data["files"]]


def _rows(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))


def _peaks(paths: PathsSettings, run: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = pq.read_table(
        paths.runs_root / run / "results" / "peaks.parquet"
    ).to_pylist()
    return rows


def _f32(value: float) -> str:
    return str(np.float32(value))


# ---------------------------------------------------------------- legacy-csv


def test_legacy_csv_header_matches_committed_string(run_id: str) -> None:
    code, env = _export(run_id, "--format", "legacy-csv")
    assert code == ExitCode.OK, env.errors
    for path in _files(env):
        window = path.stem.removeprefix("legacy-")
        first = path.read_text(encoding="utf-8").splitlines()[0]
        assert first == LEGACY_HEADERS[window]


def test_legacy_csv_one_file_per_window_in_yaml_peak_order(
    run_id: str, raman_paths: PathsSettings
) -> None:
    _, env = _export(run_id, "--format", "legacy-csv")
    exports = raman_paths.runs_root / run_id / "exports"
    assert _files(env) == [exports / "legacy-lithiation.csv", exports / "legacy-carbon.csv"]
    for path in _files(env):
        text = path.read_text(encoding="utf-8")
        assert text.endswith("\n")
        assert "\r" not in text
        assert not text.startswith("﻿")
        assert len(text.splitlines()) == 1 + 25  # header, then one row per spectrum


def test_legacy_csv_min_presence(run_id: str, raman_paths: PathsSettings) -> None:
    peaks = [p for p in _peaks(raman_paths, run_id) if p["peak_id"] == "u1"]
    presences = sorted(p["presence_prob"] for p in peaks)
    threshold = presences[len(presences) // 2]  # about half the spectra pass
    _, env = _export(run_id, "--format", "legacy-csv", "--min-presence", str(threshold))
    by_spectrum = {(p["ix"], p["iy"]): p for p in peaks}
    for row in _rows(_files(env)[0]):
        peak = by_spectrum[(int(row["x"]), int(row["y"]))]
        present = peak["presence_prob"] >= threshold and peak["center_median"] is not None
        assert row["u1 center"] == (_f32(peak["center_median"]) if present else "0")
        assert row["u1 confidence"] == _f32(peak["presence_prob"])


def test_legacy_csv_absent_fill_default_zero(run_id: str) -> None:
    _, env = _export(run_id, "--format", "legacy-csv", "--window", "carbon")
    rows = _rows(_files(env)[0])
    # The scans have no detectable carbon: at the default 0.90 nothing is present.
    assert {row["d center"] for row in rows} == {"0"}
    assert {row["g width"] for row in rows} == {"0"}
    _, env = _export(run_id, "--format", "legacy-csv", "--window", "carbon", "--absent-fill", "-1")
    assert {row["d height"] for row in _rows(_files(env)[0])} == {"-1"}


def test_legacy_csv_height_source(run_id: str, raman_paths: PathsSettings) -> None:
    peaks = {(p["ix"], p["iy"]): p for p in _peaks(raman_paths, run_id) if p["peak_id"] == "u2"}
    for source, column in (("area", "area_median"), ("height", "height_median")):
        options = ("--min-presence", "0", "--height-source", source)
        _, env = _export(run_id, "--format", "legacy-csv", "--window", "lithiation", *options)
        for row in _rows(_files(env)[0]):
            peak = peaks[(int(row["x"]), int(row["y"]))]
            expected = _f32(peak[column]) if peak[column] is not None else "0"
            assert row["u2 height"] == expected
            width = _f32(peak["fwhm_median"]) if peak["fwhm_median"] is not None else "0"
            assert row["u2 width"] == width


def test_legacy_csv_coords_index_default_and_um(run_id: str, raman_paths: PathsSettings) -> None:
    peaks = sorted(_peaks(raman_paths, run_id), key=lambda p: p["spectrum_id"])[::4]
    _, env = _export(run_id, "--format", "legacy-csv", "--window", "lithiation")
    assert [(r["x"], r["y"]) for r in _rows(_files(env)[0])] == [
        (str(p["ix"]), str(p["iy"])) for p in peaks
    ]
    _, env = _export(run_id, "--format", "legacy-csv", "--window", "lithiation", "--coords", "um")
    assert [(r["x"], r["y"]) for r in _rows(_files(env)[0])] == [
        (repr(p["x_um"]), repr(p["y_um"])) for p in peaks
    ]


def test_legacy_csv_window_option(run_id: str, raman_paths: PathsSettings) -> None:
    _, env = _export(run_id, "--format", "legacy-csv", "--window", "lithiation")
    assert _files(env) == [raman_paths.runs_root / run_id / "exports" / "legacy-lithiation.csv"]
    code, env = _export(run_id, "--format", "legacy-csv", "--window", "nope")
    assert code == ExitCode.VALIDATION
    assert "lithiation, carbon" in env.errors[0]


# ---------------------------------------------------------------- csv


def test_csv_export_flattens_peaks_parquet(run_id: str, raman_paths: PathsSettings) -> None:
    code, env = _export(run_id, "--format", "csv")
    assert code == ExitCode.OK
    (path,) = _files(env)
    assert path == raman_paths.runs_root / run_id / "exports" / "peaks.csv"
    rows = _rows(path)
    assert list(rows[0]) == PEAKS_SCHEMA.names
    peaks = _peaks(raman_paths, run_id)
    assert len(rows) == len(peaks) == 100
    first, row = peaks[0], rows[0]
    assert (row["run_id"], row["peak_id"], row["fit_status"]) == (run_id, "u1", "ok")
    assert row["spectrum_id"] == str(first["spectrum_id"])
    assert row["x_um"] == repr(first["x_um"])
    assert row["presence_prob"] == _f32(first["presence_prob"])
    _, env = _export(run_id, "--format", "csv", "--window", "carbon")
    (path,) = _files(env)
    assert path.name == "peaks-carbon.csv"
    assert {r["window_id"] for r in _rows(path)} == {"carbon"}


def test_csv_export_nulls_are_empty_fields(run_id: str, raman_paths: PathsSettings) -> None:
    _, env = _export(run_id, "--format", "csv")
    rows = _rows(_files(env)[0])
    peaks = _peaks(raman_paths, run_id)
    absent = [i for i, p in enumerate(peaks) if p["center_mean"] is None]
    assert absent
    for i in absent:
        assert rows[i]["center_mean"] == ""
        assert rows[i]["end_member_id"] != ""
    assert all(row["center_at_bound_frac"] == "" for row in rows)  # the mock never sets it


# ---------------------------------------------------------------- refusals and DuckDB


def test_export_uncollected_run_exits_2(mini_store: Path) -> None:
    config = RunConfig(
        dataset_id=mini_store.name.removesuffix(".zarr"),
        priors_path=PRIORS,
        engine="mock",
        profile="local",
    )
    planned = plan_run(config, load_settings())
    assert planned.plan is not None
    code, env = _export(planned.plan.run_id, "--format", "csv")
    assert code == ExitCode.VALIDATION
    assert "planned" in env.errors[0]
    assert "collect" in env.errors[0]
    assert _export("0190b3d4-0000-7000-8000-000000000000", "--format", "csv")[0] == 2


def test_export_with_missing_results_exits_4(run_id: str, raman_paths: PathsSettings) -> None:
    (raman_paths.runs_root / run_id / "results" / "peaks.parquet").unlink()
    code, env = _export(run_id, "--format", "csv")
    assert code == ExitCode.BACKEND
    assert "peaks.parquet" in env.errors[0]


def test_duckdb_query_over_results(run_id: str, raman_paths: PathsSettings) -> None:
    pattern = raman_paths.runs_root / "*" / "results" / "peaks.parquet"
    rows, present, peaks = duckdb.sql(
        f"SELECT count(*), count(center_mean), count(DISTINCT peak_id) FROM '{pattern}'"
    ).fetchone()  # type: ignore[misc]
    stored = _peaks(raman_paths, run_id)
    assert (rows, peaks) == (len(stored), 4)
    assert present == sum(p["center_mean"] is not None for p in stored)
    window = duckdb.sql(f"SELECT DISTINCT window_id FROM '{pattern}' ORDER BY 1").fetchall()
    assert window == [("carbon",), ("lithiation",)]


def test_legacy_csv_spectrum_without_a_peak_row_is_filled(
    run_id: str, raman_paths: PathsSettings
) -> None:
    table = pq.read_table(raman_paths.runs_root / run_id / "results" / "peaks.parquet")
    keep = pc.invert(
        pc.and_(
            pc.equal(table.column("spectrum_id"), 0),
            pc.equal(table.column("peak_id").cast(pa.string()), "u2"),
        )
    )
    window = design_window("lithiation")
    text = legacy_csv(table.filter(keep), window, LegacyOptions(absent_fill=-1))
    first = text.splitlines()[1].split(",")
    assert first[-4:] == ["-1", "-1", "-1", "-1"]
