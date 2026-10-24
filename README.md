# raman-bayes

Bayesian Raman peak-detection pipeline for hyperspectral maps. It packs Nanofinder exports
into a Zarr dataset, fits every spectrum window by window, and writes long-format Parquet
results, running the same way on a laptop, a lab workstation and Stanford Sherlock (Slurm).
The design is in [docs/DESIGN.md](https://github.com/svangara-stanford/raman-bayes/blob/main/docs/DESIGN.md).

## Status

**Phase 0 (scaffold).** The package layout, domain models, port protocols, logging, CI and
the `raman` command line are in place. Every command is still a stub that exits with code 3.
Ingest, fitting (with a mock engine) and export arrive in Phase 1; the real Bayesian engine
comes later.

## Quickstart (development)

Requires [pixi](https://pixi.sh) 0.83 or later.

```bash
git clone https://github.com/svangara-stanford/raman-bayes.git
cd raman-bayes
pixi install --locked
pixi run test          # full test suite
pixi run lint          # ruff, ruff format --check, mypy, import-linter
pixi run raman --help  # the CLI
```

Other environments: `pixi run -e py311 test` (also `py312`, `py313`).
Pre-commit hooks: `pixi run pre-commit install`, then `pixi run pre-commit run --all-files`.

## Installing without pixi

The package is a plain wheel with PyPI dependencies only:

```bash
pip install .
raman --help
```

## Test data

The full sample archives are not in the repository (`data/` is gitignored).

- Unzip `NMC811-Exported-Data.zip` and `NMC111-Exported-Data.zip` into `data/raw/` on a
  laptop, `/data/raman/raw/` on the lab PC, or `$SCRATCH/raman/raw/` on Sherlock.
- The legacy priors CSVs live in `configs/priors/legacy/` (`nmc_lithiation.csv`,
  `nmc_carbon.csv`).
- Unit tests use only the small fixtures under `tests/fixtures/`.

## Citation

See [CITATION.cff](https://github.com/svangara-stanford/raman-bayes/blob/main/CITATION.cff).

## License

Pending: BSD-3-Clause is awaiting clearance from Stanford OTL. Until then no license is
granted.
