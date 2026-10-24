# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Phase 0 scaffold: hatchling packaging, pixi environments (Python 3.11–3.13), ruff, mypy
  (strict) and import-linter with the four layering contracts.
- Domain models for priors, datasets, runs and results, with explicit Parquet schemas for
  `peaks.parquet` and `fits.parquet`.
- `PeakEngine` and `ExecutionBackend` protocols.
- Logging setup for the `raman_bayes` logger (stderr plus an optional per-task file).
- `raman` command line: all commands registered as stubs, exit codes 0/2/3/4, `--json`
  envelope with a generated JSON schema.
- Packaged default profile `local.yaml`.
- Pre-commit hooks and a CI workflow (Linux and macOS, Python 3.11–3.13, plus a plain-pip
  wheel install check).
