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

## 2026-10-28 — Phase 0 complete
- §13 "done when" met: CI green on all six matrix cells plus pre-commit and wheel-install jobs (PR #9); local test/lint green; `raman --help` and `raman ingest x --json` (exit 3) verified by hand.
- Branch protection enabled on main; CI required for merge from here on.
- Phase 1 plan from Claude Code saved to docs/plans/phase-1-plan.md pending approval.

## 2026-10-29 — Phase 1 batch A, chore: ruff D and TRY003
- Branch `phase-1-01-foundation`. Fixed the 28 pydocstyle violations (contract docstrings on the flagged classes, Protocol methods and `gen_schemas.py` functions; `make_console` in imperative mood) and the TRY003 in `CliEnvelope`, now raised as `EnvelopeStatusError(ValueError)` so pydantic still reports it as a ValidationError. No ignores added.
- Protocol method bodies are now the docstring alone; the `...` placeholders went with them.

## 2026-10-29 — Phase 1 sub-step 01: fixtures and make_fixture script
- Branch `phase-1-01-foundation`. Built `scripts/make_fixture.py` and committed `tests/fixtures/nanofinder_mini/NMC811_v0/` (real Info.txt + 25 corner spectra), `tests/fixtures/info_txt/` (the five Info.txt files) and `tests/fixtures/spectra/` (two NMC111_v1 spectra, 19920 and 19922 bytes); `.gitattributes` marks fixtures and legacy CSVs `-text`; pre-commit whitespace fixers exclude `tests/fixtures/`.
- Deviation (plan §4.1, recorded as planned): `tests/fixtures/spectra/` is beyond the §2 fixture layout, because §11 wants NMC111 reader tests and the mini fixture is NMC811 only.
- The two NMC111 files are named constants (`z0y0x0`, `z0y11x0`) rather than picked by size at run time, so the fixture is reviewable and stable.
- The script checks every source and every destination before writing, so a missing source or an unforced conflict writes nothing. Identical destinations are skipped (no mtime change).
- Data findings from the plan stand: `n_accum` 1/5/5/5/10; NMC111 Info.txt is 102 lines (no `Center Row`/`Track Height`), NMC811 is 104; 7 single-space blank lines per file.

## 2026-11-02 — Phase 1 sub-step 02: Nanofinder I/O and ScanInfo
- Branch `phase-1-01-foundation`. Built `io/filename.py` (`compile_pattern`, `parse_filename` → `SpectrumName`), `io/nanofinder.py` (section-aware `parse_info` → ScanInfo, `read_spectrum`, `axis_matches` with `AXIS_TOLERANCE_CM1 = 0.05`) and `domain/errors.py` (user-facing errors, so cli/ can catch what io/ raises). ScanInfo, GridSpec, AcquisitionInfo and DatasetManifest now enforce the v0.6 §3/§4 rules; the "provisional" wording is gone.
- Validation errors are `PydanticCustomError`s with stable types (`single_point_axis`, `multi_point_axis_zero_step`, `axis_stop_mismatch`, `time_series_not_supported`, `n_spectra_grid_mismatch`, `n_spectra_exceeds_grid`) that the tests assert on. Axis checks are written so NaN fails.
- The parser refuses `n_series ≠ 1` and negative durations with their own errors before building ScanInfo; any other model violation is re-raised as `InvalidScanInfoError` naming the file. Keys are found anywhere in their line, after a word boundary, with `:` or `=` and any spacing; numbers are the first token after the separator. Channel keys are read only from the `Channel : 0` subsection. Per-axis Frame Dimentions keys are read from that axis's block, and a missing one names `Frame Dimentions/Scanned points <axis>`.
- `parse_filename` uses `fullmatch`: FILENAME_RE ends in `$`, which would also accept a trailing newline under `re.match`. A non-matching name returns None (ingest decides what that means); a custom pattern without `scan`/`iz`/`iy`/`ix` is refused.
- `read_spectrum` parses lines in Python rather than with `np.loadtxt`, because loadtxt numbers rows from 0 for a bad value and from 1 for a column-count change, and the error must name the line. That's about 0.3 ms per file, about 4 s for the full 11,250-spectrum smoke.
- Phase 0 test changes (table): `test_scan_info_is_marked_provisional` replaced by `test_scan_info_is_not_provisional`; the `axes`/`grids`/`acqs`/`manifests` strategies now generate consistent axes, exposure > 0, duration ≥ 0 and n_spectra = grid size (≤ when partial).
- Approved change outside the original table (2026-11-02, row added to `phase-1-tests.md`): `test_manifest_rejects_unknown_nested_key` fixture grid `ny, nz` 1 → 2. The old fixture had 1-point axes with step 1, which patch 01 rejects; the assertion is unchanged.
- Added test: `test_scan_n_wn_and_n_spectra_lower_bounds` (`scan` non-empty, `n_wn ≥ 1`, `n_spectra ≥ 1`).
- DESIGN clarification for the batch A checkpoint: with `grid.allow_partial`, `n_spectra ≤ nx·ny·nz` (implied by duplicates and out-of-grid files always being errors; §4 states only `>= 1` and `== nx·ny·nz unless allow_partial`).

## 2026-11-02 — Phase 1 sub-step 03: priors rules and validate-priors
- Branch `phase-1-01-foundation`. Built `domain/priors_rules.py` (rules 1–7 as `Finding(rule, severity, message)`s in a `RuleReport`; `canonical_json` and `priors_hash` for rule 9), the named rule in `domain/priors.py` (`quarter_range`, `SD_RULES`, `center_sd`; module now has `__all__`), `app/validate.py` (`validate_priors_file` → `PriorsValidation`) and the real `raman validate-priors`.
- Every rule reports every violation; structure errors (pydantic, reported as `windows[0].peaks[0].center.mean: … (got 'auto')`) stop before the rules. `spec` and `priors_hash` are set only when there are no errors. YAML is loaded with `safe_load`; an unsafe tag is reported with its line and never constructed.
- Rule interpretations: overlap (rules 4, 5) is of open intervals, so touching ranges don't overlap; "inside" (rules 2, 3) is inclusive; the rule 2 edge warning fires when a center bound is less than `fwhm.bounds[0]` from the window edge.
- `canonical_json`: `model_dump(mode="json")`, sorted keys, `(",", ":")` separators, `ensure_ascii=False`, UTF-8. Defaults are included and numbers are parsed floats, so `465`/`465.0` and explicit defaults hash alike.
- DESIGN clarification for the batch A checkpoint (rule 1): ids are unique within their kind (windows, peaks across all windows, end members), not across kinds. The §5 example names both a window and an end member `carbon`, so a single namespace would reject it.
- Warnings go to stderr in text mode via a new opt-in `emit(..., warnings_to_stderr=True)`. The default is unchanged because Phase 0's `test_emit_human_success_goes_to_stdout` expects warnings on stdout for exit 0. Proposal for the checkpoint: make stderr the default for every command and update that test.
- Phase 0 test changes (table): `test_stub_exits_nothing_to_do` and `test_stub_json_envelope` now run over `STUBS` (every command but `validate-priors`).
- Phase 0 test change applied early: `test_profile_is_accepted` takes its approved sub-step 05 form now ("--profile parses": under `--json`, stdout is one envelope for that command), because `validate-priors` really exits 2 on a missing file from this sub-step on.
- Added tests: `test_rule1_same_id_for_window_and_end_member_allowed`, `test_rule4_touching_windows_do_not_overlap`, `test_valid_file_returns_spec_and_hash`, `test_auto_rejected_as_sd_rule_through_validate_file`, `test_warnings_do_not_block_a_valid_result`, `test_unreadable_and_non_mapping_files`.
- Unsure: `validate-priors` keeps `--profile` (every command has it, Phase 0 decision) but doesn't use it; that will trip ARG001 when the temporary `cli/main.py` ignore is removed at sub-step 14.

## 2026-11-02 — Phase 1 sub-step 04: legacy CSV and nmc_commercial.yaml
- Branch `phase-1-01-foundation`. Built `io/legacy_csv.py` (`read_legacy_csv` → Window, `format_legacy_csv`, `HEADER`), `app/priors_csv.py` (`import_window`, `export_window`), the real `raman priors import-csv` / `export-csv`, and `configs/priors/nmc_commercial.yaml`.
- A9 as decided: the first column is lowercased into the id and kept as `label` when that changes it (`D` → id `d`, label `D`); names equal up to case are a repeated-row error naming the line. Export writes ids and `repr(float)`, LF, no BOM, with a final newline (the originals have none; rows compare equal as values modulo case). The CSV has no label column, so round trips preserve everything but labels.
- The header must match `HEADER` exactly after the BOM is stripped; otherwise `LegacyCsvHeaderError`.
- `domain/priors_rules.py` now exposes `check_window` (rule 1 id pattern, rules 2, 3, 5) and `check_rules` uses it; findings are sorted by rule. Import runs `check_window`, so `--range 1750 1150` or a range that misses a peak exits 2 with the same messages `validate-priors` gives.
- Import prints the fragment as a one-item YAML list for `windows:` (`exclude_defaults`); `--json` carries `window` (full dump) and `fragment`. Export validates the whole recipe first; an unknown window id names the windows that exist.
- `nmc_commercial.yaml`: windows come from the two imports (ids, centers and bounds match the CSVs, as tested); labels, defaults, preprocessing and end members come from §5. The file parses equal to the §5 example.
- `cli/main.py`: shared `_invalid` / `_finish` helpers for result-based commands; `validate-priors` now uses them (no behaviour change).
- Phase 0 test changes (table): `priors import-csv` and `priors export-csv` moved out of the stub tests.
- Added tests: `test_import_rejects_unexpected_header`, `test_cli_import_csv_missing_file_exits_2`, `test_cli_export_csv_invalid_priors_exits_2`. The list's `test_cli_import_csv_prints_fragment / --json` and `test_cli_export_csv_prints_csv / --json` are each two tests (`…_json`).

## 2026-11-02 — Phase 1 sub-step 05: config
- Branch `phase-1-01-foundation`. Built `config/expansion.py` (`expand_variables`), `config/profiles.py` (`resolve_profile` → `ProfileSource`), `config/settings.py` (`Settings` and its section models, `load_settings` → `LoadedSettings` with `resolved_config()`), config errors in `domain/errors.py`, and `configs/profiles/{labpc,sherlock}.yaml` (equal to the §9 blocks).
- Dependency: `pydantic-settings >=2.16.2,<3` (approved), in `pixi.toml` and `pyproject.toml`. Lock diff accepted 2026-11-02: pydantic-settings 2.16.1 → 2.16.2, its `requires_dist` line on the editable package, and purl-mapping metadata churn (wrapt, markupsafe, scipy, fastapi, ruff) with no other version changes. `pixi install --locked` passes for all four environments.
- Layers are built explicitly (defaults < expanded profile < `RAMAN_*` < CLI overrides) and merged key by key; pydantic-settings' `EnvSettingsSource` parses the environment (nested `__`, JSON for lists, case-insensitive). `Settings.settings_customise_sources` keeps only the init source, so constructing it never reads the environment or a `.env` file a second time. `RAMAN_PROFILE` selects the file and is dropped from the env layer. A scoped `ARG003` per-file ignore covers that hook, whose keyword signature pydantic-settings fixes.
- `${NAME}` only; `$NAME` is literal; a `${` that doesn't open a valid name is `MalformedVariableError`; unset → `UnsetVariableError` naming the variable, the dotted key (`slurm.partitions[1]`) and the profile; expanded text is not rescanned.
- Profile resolution: a value with a path separator or a `.yaml`/`.yml` suffix is a path relative to `cwd`; otherwise packaged `raman_bayes/config/profiles/<name>.yaml`, then `cwd/configs/profiles/<name>.yaml`. The unknown-profile error names both (A15). An explicit profile beats `RAMAN_PROFILE`, which beats `local`.
- Settings shape follows the §9 YAMLs. Defaults where §9 is silent: `backend: local`, `local.workers: 1`, `engine.name: mock`, `slurm`/`limits` absent, `slurm.lock_stale_minutes: 30` (§7.2). `backend: slurm_array` requires a `slurm` section. Relative `paths.*` values become absolute (normalised, symlinks not resolved) after merging, so env and CLI values are anchored too.
- Approved this session: warnings go to stderr for every command (text mode); `emit` lost its opt-in keyword. Phase 0 test change, row added to the table: `test_emit_human_success_goes_to_stdout` now expects the message alone on stdout and the warning on stderr.
- Added tests: `test_expanded_values_are_not_expanded_again`, `test_slurm_backend_requires_slurm_section`.
- DESIGN clarifications for the checkpoint: the profile-vs-path rule above (§9 says only "a name or a path"); the defaults listed above; `lock_stale_minutes` lives under `slurm` as in the §9 YAML, though §7.2 applies it to local runs too (to settle at sub-step 11).
