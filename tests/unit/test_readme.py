"""The README documents the full-data smoke (DESIGN.md §11) and the current status."""

from pathlib import Path

README = (Path(__file__).resolve().parents[2] / "README.md").read_text(encoding="utf-8")


def _section(title: str) -> str:
    start = README.index(f"## {title}\n")
    end = README.find("\n## ", start + 1)
    return README[start : end if end != -1 else len(README)]


def test_readme_documents_full_data_smoke() -> None:
    smoke = _section("Full-data smoke")
    for needed in (
        "NMC811-Exported-Data",
        "NMC111-Exported-Data",
        "5 datasets",
        "11,250 spectra",
        "under 3 minutes",
        "raman ingest",
        "raman run",
        "raman collect",
        "configs/priors/nmc_commercial.yaml",
        "presence",
        "not in CI",
    ):
        assert needed in smoke, needed
    assert "RAMAN_PATHS__RUNS_ROOT" in smoke  # the smoke keeps its state apart


def test_readme_status_is_phase_1() -> None:
    status = _section("Status")
    assert "Phase 1" in status
    assert "Phase 0" not in status
    assert "stub" not in status
