"""Port Protocols (DESIGN.md §6, §7.3): PeakEngine, its inputs, and ExecutionBackend."""

import dataclasses
import inspect
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import yaml

from raman_bayes.domain.priors import PriorsSpec, Window
from raman_bayes.domain.priors_rules import priors_hash
from raman_bayes.domain.results import WindowFitResult
from raman_bayes.domain.run import RunPlan, SubmissionRecord, TaskSpec, TaskState, TaskStatus
from raman_bayes.ports.backend import ExecutionBackend
from raman_bayes.ports.engine import EngineConfig, PeakEngine, PreparedSpectrum

PRIORS_FIXTURE = Path(__file__).parents[1] / "fixtures" / "priors" / "design_example.yaml"
WINDOW = Window.model_validate(
    {
        "id": "lithiation",
        "range_cm1": [350, 700],
        "peaks": [
            {
                "id": "u1",
                "center": {"mean": 465, "bounds": [445, 485]},
                "fwhm": {"bounds": [35, 70]},
            }
        ],
    }
)

# Parameter names copied by hand from DESIGN.md §6 / §7.3.
SIGNATURES: list[tuple[object, list[str]]] = [
    (PeakEngine.validate, ["self", "window", "cfg"]),
    (PeakEngine.fit, ["self", "s", "window", "cfg", "rng"]),
    (PeakEngine.fit_many, ["self", "ss", "window", "cfg", "seeds"]),
    (ExecutionBackend.submit, ["self", "plan", "dry_run"]),
    (ExecutionBackend.poll, ["self", "run_id"]),
    (ExecutionBackend.cancel, ["self", "run_id", "tasks"]),
    (ExecutionBackend.resubmit, ["self", "run_id", "tasks"]),
]


def _spectrum(spectrum_id: int = 0) -> PreparedSpectrum:
    wn = np.linspace(350.0, 700.0, 64)
    return PreparedSpectrum(
        spectrum_id=spectrum_id,
        wavenumber=wn,
        intensity=np.zeros_like(wn),
        noise_sd=1.0,
        qc_flags=0,
    )


def _result(s: PreparedSpectrum, window: Window, runtime_s: float) -> WindowFitResult:
    return WindowFitResult(
        spectrum_id=s.spectrum_id,
        window_id=window.id,
        status="ok",
        peaks=(),
        diagnostics={},
        runtime_s=runtime_s,
        engine="dummy",
        engine_version="0.0.0",
    )


class DummyEngine:
    """Structural PeakEngine: does not subclass the Protocol."""

    name = "dummy"
    version = "0.0.0"

    def validate(self, window: Window, cfg: EngineConfig) -> list[str]:
        return []

    def fit(
        self, s: PreparedSpectrum, window: Window, cfg: EngineConfig, rng: np.random.Generator
    ) -> WindowFitResult:
        return _result(s, window, runtime_s=float(rng.random()))

    def fit_many(
        self,
        ss: Sequence[PreparedSpectrum],
        window: Window,
        cfg: EngineConfig,
        seeds: Sequence[np.random.SeedSequence],
    ) -> list[WindowFitResult]:
        return [
            self.fit(s, window, cfg, np.random.default_rng(seed))
            for s, seed in zip(ss, seeds, strict=True)
        ]


class EngineWithoutFitMany:
    name = "partial"
    version = "0.0.0"

    def validate(self, window: Window, cfg: EngineConfig) -> list[str]:
        return []

    def fit(
        self, s: PreparedSpectrum, window: Window, cfg: EngineConfig, rng: np.random.Generator
    ) -> WindowFitResult:
        return _result(s, window, runtime_s=0.0)


class SubclassedEngine(PeakEngine):
    """Explicit subclass that implements only fit and inherits the default fit_many."""

    name = "subclassed"
    version = "0.0.0"

    def validate(self, window: Window, cfg: EngineConfig) -> list[str]:
        return []

    def fit(
        self, s: PreparedSpectrum, window: Window, cfg: EngineConfig, rng: np.random.Generator
    ) -> WindowFitResult:
        return _result(s, window, runtime_s=float(rng.random()))


class DummyBackend:
    name = "dummy"

    def submit(self, plan: RunPlan, *, dry_run: bool = False) -> SubmissionRecord:
        return SubmissionRecord(
            run_id=plan.run_id,
            backend=self.name,
            submitted_at=datetime(2026, 10, 23, tzinfo=UTC),
            dry_run=dry_run,
            task_indices=tuple(t.task_index for t in plan.tasks),
        )

    def poll(self, run_id: str) -> Mapping[int, TaskState]:
        return {0: TaskState(status=TaskStatus.DONE, attempts=1, has_done_marker=True)}

    def cancel(self, run_id: str, tasks: Sequence[int] | None = None) -> None:
        return None

    def resubmit(self, run_id: str, tasks: Sequence[int]) -> SubmissionRecord:
        return SubmissionRecord(
            run_id=run_id,
            backend=self.name,
            submitted_at=datetime(2026, 10, 23, tzinfo=UTC),
            dry_run=False,
            task_indices=tuple(tasks),
        )


class BackendWithoutResubmit:
    name = "partial"

    def submit(self, plan: RunPlan, *, dry_run: bool = False) -> SubmissionRecord:
        raise NotImplementedError

    def poll(self, run_id: str) -> Mapping[int, TaskState]:
        return {}

    def cancel(self, run_id: str, tasks: Sequence[int] | None = None) -> None:
        return None


# ---------------------------------------------------------------- engine inputs


def test_engine_config_defaults() -> None:
    cfg = EngineConfig()
    assert (cfg.n_iter, cfg.burn_in, cfg.thin) == (6000, 3000, 5)
    assert cfg.options == {}


def test_engine_config_options_not_shared() -> None:
    assert EngineConfig().options is not EngineConfig().options


def test_engine_inputs_are_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        EngineConfig().n_iter = 1  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        _spectrum().noise_sd = 2.0  # type: ignore[misc]


def test_prepared_spectrum_fields_match_design() -> None:
    names = [f.name for f in dataclasses.fields(PreparedSpectrum)]
    assert names == ["spectrum_id", "wavenumber", "intensity", "noise_sd", "qc_flags"]


# ---------------------------------------------------------------- PeakEngine


@pytest.mark.parametrize(
    ("func", "params"), SIGNATURES, ids=lambda v: getattr(v, "__qualname__", "")
)
def test_signatures_match_design(func: object, params: list[str]) -> None:
    assert callable(func)
    assert list(inspect.signature(func).parameters) == params


def test_structural_engine_satisfies_protocol() -> None:
    engine: PeakEngine = DummyEngine()  # checked by mypy in lint
    assert isinstance(engine, PeakEngine)


def test_engine_without_fit_many_is_rejected() -> None:
    assert not isinstance(EngineWithoutFitMany(), PeakEngine)


def test_default_fit_many_loops_over_fit_with_one_rng_per_seed() -> None:
    engine = SubclassedEngine()
    spectra = [_spectrum(i) for i in range(3)]
    seeds = [np.random.SeedSequence(7, spawn_key=(0, i)) for i in range(3)]
    results = engine.fit_many(spectra, WINDOW, EngineConfig(), seeds)
    expected = [
        engine.fit(s, WINDOW, EngineConfig(), np.random.default_rng(seed))
        for s, seed in zip(spectra, seeds, strict=True)
    ]
    assert results == expected
    assert [r.spectrum_id for r in results] == [0, 1, 2]


def test_default_fit_many_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="zip"):
        SubclassedEngine().fit_many(
            [_spectrum()], WINDOW, EngineConfig(), [np.random.SeedSequence(1)] * 2
        )


# ---------------------------------------------------------------- ExecutionBackend


def test_submit_dry_run_is_keyword_only() -> None:
    param = inspect.signature(ExecutionBackend.submit).parameters["dry_run"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is False


def test_structural_backend_satisfies_protocol() -> None:
    backend: ExecutionBackend = DummyBackend()  # checked by mypy in lint
    assert isinstance(backend, ExecutionBackend)


def test_backend_without_resubmit_is_rejected() -> None:
    assert not isinstance(BackendWithoutResubmit(), ExecutionBackend)


def test_dummy_backend_satisfies_protocol() -> None:
    backend: ExecutionBackend = DummyBackend()  # checked by mypy in lint
    priors = PriorsSpec.model_validate(yaml.safe_load(PRIORS_FIXTURE.read_text(encoding="utf-8")))
    plan = RunPlan(
        run_id="0190b3d4-0000-7000-8000-000000000000",
        created_at=datetime(2026, 10, 23, tzinfo=UTC),
        dataset_id="NMC811_v0-0123456789",
        dataset_hash="h",
        dataset_uri="/data/datasets/NMC811_v0-0123456789.zarr",
        n_spectra=25,
        priors=priors,
        priors_hash=priors_hash(priors),
        engine="mock",
        engine_version="0.1.0",
        engine_config={},
        backend="local",
        profile="local",
        resolved_config={},
        seed=1,
        spectra_per_task=1024,
        tasks=(TaskSpec(task_index=0, spec_start=0, spec_stop=25),),
    )
    record = backend.submit(plan, dry_run=True)
    assert (record.dry_run, record.task_indices) == (True, (0,))
    assert backend.poll(plan.run_id)[0].has_done_marker
