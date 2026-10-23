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
- Small commits with Conventional Commit messages (feat:, fix:, test:, refactor:, docs:, chore:).
- Do not start the next phase until the current phase's "done when" criteria in DESIGN.md §13 pass.
- Ask before adding a dependency, changing a schema, or touching anything in §15.
- For each sub-step: failing tests first, then implementation. Stop with a five-line summary (built, tested, unsure, deviations, next)
- Each sub-step's commit includes an entry in docs/DEVLOG.md: date, branch, what was built in one line, and any decisions or deviations made in plan review or during the sub-step. I may append my own lines afterwards.

## Data
- data/raw/ holds the test archives (gitignored). Fixture: tests/fixtures/nanofinder_mini/.
- Legacy priors CSVs: configs/priors/legacy/. Generate configs/priors/nmc_commercial.yaml from them (DESIGN.md §5).
- Packaged default profile: src/raman_bayes/config/profiles/local.yaml. Site profiles (labpc, sherlock) live in configs/profiles/.
