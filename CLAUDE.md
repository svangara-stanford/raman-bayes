# CLAUDE.md
Read docs/DESIGN.md before any work. It is authoritative; Section 15 decisions require my approval to change.

## Commands
- pixi run test        # full suite; must be green before any commit
- pixi run lint        # ruff + mypy + import-linter
- pixi run raman ...   # the CLI

## Rules
- All Python runs via pixi. Never pip install globally.
- Import contracts (import-linter, DESIGN.md §2): cli/ imports only app/, domain/, config/ and raman_bayes.logging; nothing outside cli/ imports cli/; domain/ imports no other subpackage; app/ never imports rich or typer and never prints.
- Packaging (DESIGN.md §2): runtime deps must have PyPI wheels. Any non-.py file read at runtime lives under src/raman_bayes/ and is loaded via importlib.resources, never by repo-relative path.
- Tasks write only to their own shards/task=N/. Only the CLI driver writes catalog.sqlite.
- No zero-fill: absent peaks are null in Parquet.
- Every new module gets tests in the same PR. No test, no merge.
- Small commits with Conventional Commit messages (feat:, fix:, test:, refactor:, docs:, chore:, ci:).
- Do not start the next phase until the current phase's "done when" criteria in DESIGN.md §13 pass.
- Ask before adding a dependency, changing a schema, or touching anything in §15. When a provisional model is finalized, propose the DESIGN.md patch first.
- For each sub-step: failing tests first, then implementation. One commit per sub-step.
- Branching: one branch per sub-step, or per batch of sub-steps I have explicitly approved as one PR, named phase-N-NN-topic (NN = first sub-step), always from main after I say "merged". Never branch from an unmerged branch.
- When a branch is done: `pixi run test && pixi run lint` green, then push the branch and open a PR against main with `gh pr create`, using the five-line summary (built, tested, unsure, deviations, next) as the PR body. Print the same summary in chat and stop. Never merge; never push to main.
- Each sub-step's commit includes an entry in docs/DEVLOG.md: date, branch, what was built in one line, and any decisions or deviations made in plan review or during the sub-step. I may append my own lines afterwards.
- User-facing text (--help, error messages, log lines a user sees) never references repository paths or internal docs (DESIGN.md, schemas/, configs/, docs/). Installed users don't have the repo. Such references go in comments and docstrings.

## Data
- data/raw/ holds the test archives (gitignored). Fixture: tests/fixtures/nanofinder_mini/. Unit tests use tests/fixtures/, never data/raw/.
- Legacy priors CSVs: configs/priors/legacy/. Generate configs/priors/nmc_commercial.yaml from them (DESIGN.md §5).
- Packaged default profile: src/raman_bayes/config/profiles/local.yaml. Site profiles (labpc, sherlock) live in configs/profiles/.

## Batch mode (Phase 1)
- Phase 1 runs in three batches on one branch each: A = sub-steps 01–06 (`phase-1-01-foundation`), B = 07–10 (`phase-1-07-pipeline`), C = 11–15 (`phase-1-11-execution`). One commit per sub-step with its DEVLOG entry; push after every commit so CI runs on each; open one PR at the end of the batch with all five-line summaries in the body; stop.
- Hard stops. Stop immediately, print the summaries so far, and wait, if any of these occur: a dependency not already approved (approved: blake3, pydantic-settings, jsonschema); any change to a Parquet schema, the CLI envelope, FILENAME_RE, or anything in DESIGN.md §15; a DESIGN.md patch that differs from the one I applied; CI red after two fix attempts on one sub-step; any test deleted, skipped, xfailed, or weakened to make it pass; a pixi.lock change beyond the approved dependencies.
- Resume rule: after compaction or a new session, read CLAUDE.md, docs/DESIGN.md, docs/plans/phase-1-plan.md and the last DEVLOG entry, then continue from the next sub-step of the current batch.

## Code standards
This is a package for release. Write it as a senior engineer would review it.
- A module does one thing and is named for it. No utils.py, helpers.py, common.py or misc.py.
- Functions fit on a screen; more than five parameters means the arguments want to be a model. Prefer flat code to nested conditionals; return early.
- Comments say why, never what. No docstring that restates the signature; public functions get a docstring that states the contract (inputs, output, errors raised). No "attempts to", "helper for", "this function".
- Catch an exception only where you handle it, and name the exception. Never `except Exception`. Errors raised to the user are specific exception classes in domain/ or app/, with messages a user can act on.
- Use what exists before writing it: numpy, scipy, pyarrow, pydantic, pathlib, itertools, functools. No hand-rolled versions of library functions.
- No dead code, commented-out code, TODO without an issue reference, or unused parameters.
- Public API is explicit: each public module lists `__all__`; anything not listed starts with `_`.
- Before committing each sub-step, reread the diff once as a reviewer and remove what would not survive code review at a serious open-source project.
