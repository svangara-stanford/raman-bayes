"""``run_manifest.json`` (DESIGN.md §8.3): what a collected run is and the code that made it.

The plan gives the run's identity, recipe and engine configuration. Provenance comes from the
tasks' done markers: tasks must agree on the code they ran (package version, git commit and
state, pixi.lock, engine version), since results from different code are not one analysis.
Hosts and Slurm array ids may differ between tasks; the manifest lists each distinct one.
``n_failed`` and ``n_skipped_qc`` count rows of the merged fits table.
"""

from collections import Counter
from collections.abc import Mapping
from typing import Annotated, Literal

import pyarrow as pa
from pydantic import AwareDatetime, Field, JsonValue

from raman_bayes.domain._base import DomainModel
from raman_bayes.domain.errors import MixedCodeError
from raman_bayes.domain.markers import DoneMarker
from raman_bayes.domain.priors import PreprocessingSpec
from raman_bayes.domain.run import RunPlan

__all__ = ["CODE_FIELDS", "RunManifest", "build_manifest"]

CODE_FIELDS = ("package_version", "git_sha", "git_dirty", "pixi_lock_hash", "engine_version")
Count = Annotated[int, Field(ge=0)]


class RunManifest(DomainModel):
    """The §8.3 keys, exactly.

    ``host`` and ``slurm_array_job_id`` list the distinct values of the tasks collected,
    sorted and separated by ``", "`` (the array id is None off Slurm). ``started_at`` is the
    earliest task start and ``completed_at`` the latest finish. ``partial`` is true when some
    task had not finished; ``shard_hashes`` has an entry per collected task.
    """

    run_id: str
    dataset_id: str
    dataset_hash: str
    priors_name: str
    priors_version: str
    priors_hash: str
    engine: str
    engine_version: str
    engine_config: dict[str, JsonValue]
    preprocessing: PreprocessingSpec
    run_seed: Count
    backend: Literal["local", "slurm_array"]
    profile: str
    package_version: str
    git_sha: str | None
    git_dirty: bool | None
    pixi_lock_hash: str | None
    host: str
    slurm_array_job_id: str | None
    started_at: AwareDatetime
    completed_at: AwareDatetime
    n_spectra: Count
    n_tasks: Count
    n_failed: Count
    n_skipped_qc: Count
    partial: bool
    shard_hashes: dict[int, str]


def build_manifest(plan: RunPlan, markers: Mapping[int, DoneMarker], fits: pa.Table) -> RunManifest:
    """Return the manifest of ``plan`` collected from the tasks in ``markers``.

    ``markers`` maps task index to done marker and must not be empty; ``fits`` is the merged
    fits table. Raises MixedCodeError naming every field on which the tasks disagree.
    """
    differences = [_difference(markers, name) for name in CODE_FIELDS]
    if any(differences):
        raise MixedCodeError(plan.run_id, [d for d in differences if d])
    done = list(markers.values())
    first = done[0]
    statuses = Counter(fits.column("status").to_pylist())
    arrays = sorted({m.slurm_array_job_id for m in done if m.slurm_array_job_id is not None})
    return RunManifest(
        run_id=plan.run_id,
        dataset_id=plan.dataset_id,
        dataset_hash=plan.dataset_hash,
        priors_name=plan.priors.spec.name,
        priors_version=plan.priors.spec.version,
        priors_hash=plan.priors_hash,
        engine=plan.engine,
        engine_version=plan.engine_version,
        engine_config=plan.engine_config,
        preprocessing=plan.priors.preprocessing,
        run_seed=plan.seed,
        backend=plan.backend,
        profile=plan.profile,
        package_version=first.package_version,
        git_sha=first.git_sha,
        git_dirty=first.git_dirty,
        pixi_lock_hash=first.pixi_lock_hash,
        host=", ".join(sorted({m.host for m in done})),
        slurm_array_job_id=", ".join(arrays) or None,
        started_at=min(m.started_at for m in done),
        completed_at=max(m.completed_at for m in done),
        n_spectra=plan.n_spectra,
        n_tasks=len(plan.tasks),
        n_failed=statuses["failed"],
        n_skipped_qc=statuses["skipped_qc"],
        partial=len(markers) < len(plan.tasks),
        shard_hashes={i: m.shard_hash for i, m in sorted(markers.items())},
    )


def _difference(markers: Mapping[int, DoneMarker], name: str) -> str | None:
    tasks: dict[object, list[int]] = {}
    for index, marker in sorted(markers.items()):
        tasks.setdefault(getattr(marker, name), []).append(index)
    if len(tasks) == 1:
        return None
    seen = ", ".join(f"{value} (tasks {', '.join(map(str, ids))})" for value, ids in tasks.items())
    return f"{name} {seen}"
