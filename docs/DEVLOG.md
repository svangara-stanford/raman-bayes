# Development log

Short, dated entries: what was done, what was decided, anything overridden. Newest at the bottom.

## 2026-10-22 — Phase 0 scaffold
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

## 2026-10-23 — Phase 0 sub-step 5: result models and Parquet schemas
- Branch `phase-0-05-results-ports` (sub-steps 5 and 6 batched into one PR, approved). Built `domain/results.py`: frozen dataclasses PosteriorSummary, PeakPosterior, WindowFitResult (field for field as §6), and PEAKS_SCHEMA (41 columns, §8.2 order) and FITS_SCHEMA (fixed columns only; merge appends `diag_*`).
- String identifiers are non-nullable `dictionary<int32, string>`; `end_member_id` is nullable (a peak may belong to no end member).
- `presence_prob`, the 25 summary columns and both `*_at_bound_frac` are nullable float32.
- FITS column types are my choice, since §8.2 lists names only: `spectrum_id` uint32, `runtime_s` float64, `noise_sd` nullable float32, `error` nullable plain string, other string columns dictionary.
- The schema constants carry no metadata; `run_manifest.json` and `schema_version` are attached per file at write time.
- mypy: `ignore_missing_imports` override scoped to `pyarrow` / `pyarrow.*` (pyarrow 25 still ships no `py.typed`).

## 2026-10-23 — Phase 0 sub-step 6: port Protocols
- Branch `phase-0-05-results-ports`. Built `ports/engine.py` (frozen PreparedSpectrum and EngineConfig, `@runtime_checkable` PeakEngine) and `ports/backend.py` (ExecutionBackend per §7.3; `poll` returns `Mapping[int, TaskState]` using the provisional run models).
- PeakEngine's `fit_many` has a body: the §6 default ("loops over `fit`"), one `default_rng(seed)` per SeedSequence, `zip(strict=True)`. Only explicit subclasses inherit it; a structural engine must still define `fit_many` itself, so `isinstance` rejects one without it.
- PreparedSpectrum arrays are typed `NDArray[np.float64]` (mypy strict rejects bare `np.ndarray`); §6 does not fix the dtype.
- ExecutionBackend is not `@runtime_checkable`, as in §7.3; conformance is checked by mypy (typed assignment of a dummy backend in the tests).

## 2026-10-23 — Phase 0 sub-step 6 fix: backend runtime check and Protocol signature guard
- Branch `phase-0-06-backend-typing-guard`. PR #5 was merged before this review fix was pushed, so the commit was cherry-picked here from the deleted `phase-0-05-results-ports` branch.
- ExecutionBackend is now `@runtime_checkable`, matching PeakEngine; this supersedes the sub-step 6 bullet. Added isinstance tests: a full backend passes, one without `resubmit` fails.
- `tests/unit/typing_checks_ports.py` is a signature guard that mypy checks and pytest does not collect. Typed assignments of a backend with a wrong `poll` return and an engine with a wrong `fit` return carry `# type: ignore[assignment]`. Under strict `warn_unused_ignores`, `pixi run lint` fails if either Protocol stops catching the mismatch. `isinstance` on runtime-checkable Protocols checks member names only, so both checks are kept. No subprocess-mypy test.
- Accepted (PR #5 review): FITS column types as in sub-step 5 (`spectrum_id` uint32, `runtime_s` float64, `noise_sd` nullable float32, `error` nullable string, other string columns dictionary).
- Accepted (PR #5 review): `presence_prob` is nullable float32 in PEAKS_SCHEMA.
- Accepted (PR #5 review): PreparedSpectrum `wavenumber` and `intensity` are `NDArray[np.float64]`.

## 2026-10-23 — Phase 0 sub-step 7: logging setup and import-linter contracts
- Branch `phase-0-07-logging-imports`. Built `raman_bayes/logging.py` (`setup_logging(level="WARNING", *, log_file=None)`) and the four §2 contracts as `forbidden` contracts in `[tool.importlinter]`; `pixi run lint` now runs `lint-imports`.
- Logging: the `raman_bayes` logger has `propagate=False`. The stderr handler looks up `sys.stderr` at emit time, so it works when tests or the CLI swap stderr. The optional file handler is for Phase 1's per-task log. Repeated calls replace and close only the handlers `setup_logging` added.
- Dropped `captureWarnings(True)` (plan review): it routes warnings to the `py.warnings` logger, which never reaches the `raman_bayes` handlers.
- Contract 1 lists what cli may not import (ports, io, storage, preprocessing, engine, orchestration), with `allow_indirect_imports = true`; the package root stays importable for `__version__`. Contracts 2 and 3 also cover `raman_bayes.logging`.
- Tests check that each contract's module list covers every top-level module (`pkgutil.iter_modules`), so a new top-level module must be explicitly allowed or forbidden. The plan required this for contract 1 only; I added it for contracts 2 and 3. A planted violation per contract, in a tmp copy of the tree, breaks exactly that contract.
- `lint-imports --no-cache`, so no `.import_linter_cache/` appears and `.gitignore` is unchanged. pixi.toml changes tasks only; `pixi.lock` is unchanged. DESIGN.md and CLAUDE.md already state the contracts (v0.4), so neither is edited.

## 2026-10-23 — Phase 0 sub-step 8: CLI stubs and the --json envelope
- Branch `phase-0-08-cli-stubs`. Built `cli/output.py` (ExitCode 0/2/3/4, CliEnvelope, `make_console`, `emit`), `cli/main.py` (all 18 §10 commands as stubs that exit 3) and `scripts/gen_schemas.py`, which writes `schemas/cli/envelope.schema.json`; a test fails if the committed file is stale (`pixi run gen-schemas`).
- Every command has `--json` and `--profile` (plain str, default `local`, ignored by the stubs; added in plan review so `--help` shows the full §9/§10 surface). `run-task` has `--workers` (int ≥ 1); `priors export-csv` requires `--window-id`.
- Arguments beyond the §10 table: `ingest --allow-axis-mismatch` (§3), `priors import-csv --window-id --range LO HI` (§5), and the `export` options with §8.4 defaults. `submit --dry-run --confirm` together is not rejected yet.
- Plain Click help via `rich_markup_mode=None`: Typer forces a Rich terminal when GITHUB_ACTIONS, FORCE_COLOR or PY_COLORS is set, which would put ANSI into piped `--help` on CI.
- Strict no-ANSI: when stdout (or stderr) is not a TTY, `make_console` emits no ANSI even if FORCE_COLOR is set.
- `pretty_exceptions_enable=False`: tracebacks are plain stdlib ones, no Rich.
- CliEnvelope is provisional (§10 only says output validates against `schemas/cli/`): schema_version, command, exit_code, ok (must equal `exit_code == 0`), message, data, warnings, errors.
- Accepted Phase 0 gap: Click usage errors (exit 2) print plain text, not an envelope, even under `--json`.
- Accepted Phase 0 gap: unhandled exceptions exit 1, which is outside the §10 codes.
- Human-mode output goes to stdout for exit 0/3 and stderr for 2/4. Typer 0.27 vendors Click (`typer._click`), so the test walks the command tree by its `commands` mapping instead of using `isinstance(click.Group)`.
- Config: mypy `files` now includes `scripts/`; new pixi task `gen-schemas`. No dependency changes; `pixi.lock` is unchanged.
- Fix (PR #8 review): `--help` text no longer cites DESIGN.md (root help and `--profile`), since installed users don't have it; comments and docstrings keep the references. The help tests now assert that no `--help` output mentions DESIGN.md.
- Fix (PR #8 review): rule from now on: help text, error messages and other user-visible output cite no repo paths or internal docs; those references go in comments and docstrings. Applied to the `--json` help (drops `schemas/cli/envelope.schema.json`), the stub message (drops "(Phase 1)"), and `gen_schemas.py --help` (no longer prints the module docstring). The CLI tests now check every `--help`, stub message and envelope message for DESIGN.md, §, schemas/, docs/, tests/ and "Phase".

## 2026-10-23 — Phase 0 sub-step 9: pre-commit, CI, packaged profile, project docs
- Branch `phase-0-09-ci-docs`. Built `.pre-commit-config.yaml`, `.github/workflows/ci.yml`, `src/raman_bayes/config/profiles/local.yaml` (the §9 YAML), README, CHANGELOG, CITATION.cff, and `.gitkeep` in `tests/integration/`, `tests/regression/`, `docs/adr/`.
- CI runs on pull_request and on push to main only (plan review), so PR branches don't run twice. Every job has `timeout-minutes: 30` (draft review). Matrix {ubuntu, macos} × {py311, py312, py313}, pixi v0.83.0 with `locked: true`; a pre-commit job; a `wheel-install` job (§11) that builds with `python -m build` on Python 3.11, installs into a plain venv, and runs `raman --help`, `import raman_bayes`, an `importlib.resources` check for `local.yaml`, and `raman doctor --json` (exit 3). osx-64 is locked but not run.
- `ci.yml` was drafted by me and copied into place by Sreya: `.claude/settings.json` denies me edits under `.github/`.
- Added `hatchling>=1.24` to the `dev` feature (approved) so a unit test can build the wheel and check its contents: `local.yaml` byte-identical, `py.typed`, the `raman` entry point, nothing outside `raman_bayes/`. The lock only adds hatchling, editables, tomlkit and trove-classifiers.
- `tests/unit/test_project_files.py` pins the approved CI and pre-commit choices (triggers, matrix vs pixi envs, pixi version, pinned action and hook versions, local hooks through pixi) and checks CITATION.cff against pyproject.
- pre-commit: `pixi.lock` (~550 KB) is also excluded from check-added-large-files, not only from the whitespace fixers as planned. The legacy priors CSVs are excluded from end-of-file-fixer and trailing-whitespace: the fixer added a final newline to both, and they must stay verbatim (§5 round-trip, BOM/line-ending handling).
- README doc links are absolute GitHub URLs, since the README is the PyPI long description.
- `configs/` added to the CLI tests' forbidden user-visible fragments, following the new CLAUDE.md rule.
- Fixed the date of the first entry (Phase 0 scaffold): 2026-10-23 → 2026-10-22.
- Short entry by agreement: Phase 0 decisions and gaps are already in each sub-step's entry.
