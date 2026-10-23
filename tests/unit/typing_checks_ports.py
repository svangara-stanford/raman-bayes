"""Signature guard for the port Protocols: checked by mypy (``pixi run lint``), never by pytest.

The filename does not match ``test_*.py``, so pytest does not collect it; mypy checks it
because ``tests`` is in ``[tool.mypy] files``.

The trick: each assignment below is a deliberate type error silenced with
``# type: ignore[assignment]``. Strict mypy sets ``warn_unused_ignores``, so if a Protocol
stops catching the mismatch, the ignore becomes unused and lint fails.

``isinstance`` on a ``@runtime_checkable`` Protocol checks member names only (test_ports.py
covers that); these assignments check signatures. Both are needed.
"""

from collections.abc import Sequence

import numpy as np

from raman_bayes.domain.priors import Window
from raman_bayes.domain.results import WindowFitResult
from raman_bayes.domain.run import RunPlan, SubmissionRecord
from raman_bayes.ports.backend import ExecutionBackend
from raman_bayes.ports.engine import EngineConfig, PeakEngine, PreparedSpectrum


class BackendWithWrongPoll:
    name = "wrong_poll"

    def submit(self, plan: RunPlan, *, dry_run: bool = False) -> SubmissionRecord:
        raise NotImplementedError

    def poll(self, run_id: str) -> list[str]:  # should be Mapping[int, TaskState]
        return []

    def cancel(self, run_id: str, tasks: Sequence[int] | None = None) -> None:
        return None

    def resubmit(self, run_id: str, tasks: Sequence[int]) -> SubmissionRecord:
        raise NotImplementedError


class EngineWithWrongFit:
    name = "wrong_fit"
    version = "0.0.0"

    def validate(self, window: Window, cfg: EngineConfig) -> list[str]:
        return []

    def fit(  # should return WindowFitResult
        self, s: PreparedSpectrum, window: Window, cfg: EngineConfig, rng: np.random.Generator
    ) -> str:
        return ""

    def fit_many(
        self,
        ss: Sequence[PreparedSpectrum],
        window: Window,
        cfg: EngineConfig,
        seeds: Sequence[np.random.SeedSequence],
    ) -> list[WindowFitResult]:
        return []


backend: ExecutionBackend = BackendWithWrongPoll()  # type: ignore[assignment]
engine: PeakEngine = EngineWithWrongFit()  # type: ignore[assignment]
