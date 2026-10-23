# Development log

Short, dated entries: what was done, what was decided, anything overridden. Newest at the bottom.

## 2026-10-23 — Phase 0 scaffold
- Started Phase 0 with Claude Code.
- Import contract: cli/ may also import config/ and raman_bayes.logging (both docs updated).

## 2026-10-22 — Phase 0 sub-step 2 merged: priors models
- `domain/priors.py` (16 models, frozen, extra=forbid, §5 key names). No validation rules yet.
- Plan review decisions for later steps: `raw_stage` stored as verbatim text (`str`), not a dict; `TaskState` is a model with a `TaskStatus` enum field, mirroring the §8.5 task table; `SmoothingSpec` is a provisional placeholder.
- Said no to: nothing. Approved extra fixture-vs-DESIGN.md drift test.

## 2026-10-23 — Phase 0 sub-step 3 merged: dataset models
- Branch `phase-0-03-dataset` (41aefec, bbe0c2f). Entry completed retroactively in sub-step 4.
- `domain/dataset.py`: QCFlag bits 1/2/4/8, FILENAME_RE verbatim from §3, grid/instrument/acquisition models, provisional ScanInfo, DatasetManifest.
- `acquired_at` is naive by design (Info.txt records local time with no timezone; plain `datetime`, so an aware value is not rejected).
- `created_at` is a pydantic `AwareDatetime` (pipeline-generated, written in UTC): naive values are rejected; other offsets are accepted as the same instant, so writing UTC is the caller's job.
- Manifest carries dataset_id/sample/content_hash/created_at itself, duplicating the §4 zarr attrs; this placement is provisional, like ScanInfo.

## 2026-10-23 — Phase 0 sub-step 4: run models
- Branch `phase-0-04-run`. Built `domain/run.py`: RunStatus (7 §7.4 states, no transition table), TaskSpec, RunPlan (§7.1 fields plus run_id), and provisional RunConfig, TaskStatus, TaskState, SubmissionRecord.
- TaskState is a model (`status: TaskStatus`, attempts, slurm_job_id, has_done_marker, n_ok, n_failed, error) mirroring the §8.5 task table, per plan review.
- `SubmissionRecord.submitted_at` is `AwareDatetime`, following the `created_at` decision.
- `RunPlan.engine_config` and `resolved_config` are JSON-valued dicts: domain cannot import `ports.EngineConfig`, and config models are Phase 1.
- Slurm ids are strings (`sbatch --parsable` can return `id;cluster`; array tasks are `id_N`).
