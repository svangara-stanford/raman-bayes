"""Plan a run (DESIGN.md §7.1): resolve a RunConfig into a frozen RunPlan and record it.

Everything is checked before anything is written: the recipe (§5 rules 1-7), the dataset, rule 8
against the dataset axis, the engine and its ``validate`` messages, and the task split. Then
``runs/<run_id>/run_plan.json`` is written (to a temporary name, then renamed) and the catalog
gets the recipe, the run as ``planned`` and its tasks as ``pending``. A draft recipe is planned
with a warning; production runs will require approved recipes.
"""

import dataclasses
import os
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import JsonValue, ValidationError

from raman_bayes.app.validate import validate_priors_file
from raman_bayes.config.settings import LoadedSettings, Settings
from raman_bayes.domain.errors import (
    InputError,
    InvalidRunPlanError,
    RunPlanNotFoundError,
    TaskSizeRequiredError,
)
from raman_bayes.domain.priors import PriorsSpec
from raman_bayes.domain.priors_rules import check_axis, priors_hash
from raman_bayes.domain.run import SEED_BITS, RunConfig, RunPlan, config_hash
from raman_bayes.domain.uuid7 import uuid7
from raman_bayes.engine.registry import load_engine
from raman_bayes.orchestration.chunking import Chunking, plan_tasks
from raman_bayes.ports.engine import EngineConfig, PeakEngine
from raman_bayes.storage.catalog import Catalog, RunRecord
from raman_bayes.storage.zarr_dataset import StoredDataset, open_dataset

__all__ = ["RUN_PLAN_FILE", "PlanResult", "plan_run", "read_plan"]

RUN_PLAN_FILE = "run_plan.json"


@dataclass(frozen=True)
class PlanResult:
    """Outcome of planning. ``plan`` and ``run_dir`` are set exactly when ``errors`` is empty."""

    errors: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    plan: RunPlan | None = None
    run_dir: Path | None = None


@dataclass(frozen=True)
class _Checked:
    """A recipe, dataset, engine and task split that passed every check."""

    spec: PriorsSpec
    yaml_text: str
    dataset: StoredDataset
    engine: PeakEngine
    chunking: Chunking


def plan_run(
    config: RunConfig, loaded: LoadedSettings, *, now: datetime | None = None
) -> PlanResult:
    """Plan ``config`` with ``loaded`` settings; ``now`` (default: the current time) stamps it.

    Problems with the request come back as errors and write nothing. Raises DatasetStoreError
    when the catalog names a store that is missing or inconsistent.
    """
    validation = validate_priors_file(config.priors_path)
    if validation.spec is None:
        return PlanResult(errors=validation.errors, warnings=validation.warnings)
    warnings = list(validation.warnings)
    checked = _check(config, loaded.settings, validation.spec)
    if isinstance(checked, PlanResult):
        return dataclasses.replace(checked, warnings=(*warnings, *checked.warnings))
    if checked.spec.spec.status == "draft":
        warnings.append(
            f"{config.priors_path}: recipe status is draft; production runs need approved"
        )
    plan = _plan(config, loaded, checked, now or datetime.now(UTC))
    run_dir = _write(plan, loaded.settings.paths.runs_root)
    _register(Catalog(loaded.settings.paths.catalog), plan, checked.yaml_text)
    return PlanResult(errors=(), warnings=tuple(warnings), plan=plan, run_dir=run_dir)


def _check(config: RunConfig, settings: Settings, spec: PriorsSpec) -> _Checked | PlanResult:
    row = Catalog(settings.paths.catalog).get_dataset(config.dataset_id)
    if row is None:
        return PlanResult(errors=(f"no dataset {config.dataset_id!r} in {settings.paths.catalog}",))
    dataset = open_dataset(Path(row.store_uri))
    axis = check_axis(spec, dataset.wavenumber())
    if axis.errors:
        return PlanResult(errors=axis.errors, warnings=axis.warnings)
    try:
        engine = load_engine(config.engine)
        chunking = _chunking(config, settings, dataset.manifest.n_spectra)
    except InputError as exc:
        return PlanResult(errors=(str(exc),))
    refusals = tuple(m for w in spec.windows for m in engine.validate(w, EngineConfig()))
    if refusals:
        return PlanResult(errors=refusals)
    yaml_text = config.priors_path.read_text(encoding="utf-8")
    return _Checked(spec, yaml_text, dataset, engine, chunking)


def _chunking(config: RunConfig, settings: Settings, n_spectra: int) -> Chunking:
    # Only the Slurm backend caps the task count; local runs take any number of tasks.
    slurm = settings.slurm if settings.backend == "slurm_array" else None
    if slurm is not None and config.spectra_per_task is None:
        raise TaskSizeRequiredError(settings.backend)
    return plan_tasks(
        n_spectra,
        explicit=config.spectra_per_task,
        workers=settings.local.workers,
        max_tasks=slurm.max_array_tasks if slurm is not None else None,
    )


def _plan(config: RunConfig, loaded: LoadedSettings, checked: _Checked, now: datetime) -> RunPlan:
    manifest = checked.dataset.manifest
    engine_config: dict[str, JsonValue] = dataclasses.asdict(EngineConfig())
    return RunPlan(
        run_id=str(uuid7()),
        created_at=now,
        dataset_id=manifest.dataset_id,
        dataset_hash=manifest.content_hash,
        dataset_uri=str(checked.dataset.path),
        n_spectra=manifest.n_spectra,
        priors=checked.spec,
        priors_hash=priors_hash(checked.spec),
        engine=checked.engine.name,
        engine_version=checked.engine.version,
        engine_config=engine_config,
        backend=loaded.settings.backend,
        profile=config.profile,
        resolved_config=loaded.resolved_config(),
        seed=config.seed if config.seed is not None else secrets.randbits(SEED_BITS),
        spectra_per_task=checked.chunking.spectra_per_task,
        tasks=checked.chunking.tasks,
    )


def _write(plan: RunPlan, runs_root: Path) -> Path:
    run_dir = runs_root / plan.run_id
    run_dir.mkdir(parents=True)
    staged = run_dir / f".{RUN_PLAN_FILE}.tmp"
    staged.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    os.replace(staged, run_dir / RUN_PLAN_FILE)
    return run_dir


def _register(catalog: Catalog, plan: RunPlan, yaml_text: str) -> None:
    catalog.add_priors_spec(plan.priors, plan.priors_hash, yaml_text, plan.created_at)
    record = RunRecord(
        run_id=plan.run_id,
        dataset_id=plan.dataset_id,
        priors_hash=plan.priors_hash,
        engine=plan.engine,
        engine_version=plan.engine_version,
        config_hash=config_hash(plan),
        git_sha=None,
        backend=plan.backend,
        profile=plan.profile,
        seed=plan.seed,
        created_at=plan.created_at,
    )
    catalog.create_run(record, plan.tasks)


def read_plan(run_dir: Path) -> RunPlan:
    """Return the plan in ``run_dir/run_plan.json``, checked as on construction.

    Raises RunPlanNotFoundError when there is no plan file, and InvalidRunPlanError when it is
    not a valid plan or belongs to another run.
    """
    path = run_dir / RUN_PLAN_FILE
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise RunPlanNotFoundError(path) from None
    try:
        plan = RunPlan.model_validate_json(text)
    except ValidationError as exc:
        raise InvalidRunPlanError(path, str(exc)) from exc
    if plan.run_id != run_dir.name:
        raise InvalidRunPlanError(path, f"it is the plan of run {plan.run_id}")
    return plan
