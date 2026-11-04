# raman-bayes

Bayesian Raman peak-detection pipeline for hyperspectral maps. It packs Nanofinder exports
into a Zarr dataset, fits every spectrum window by window, and writes long-format Parquet
results, running the same way on a laptop, a lab workstation and Stanford Sherlock (Slurm).
The design is in [docs/DESIGN.md](https://github.com/svangara-stanford/raman-bayes/blob/main/docs/DESIGN.md).

## Status

**Phase 1 (core pipeline, local).** The whole pipeline runs on one machine with a mock
engine: `raman ingest` packs Nanofinder scans into Zarr, `raman plan` freezes a run,
`raman submit` (or `raman run`) fits every spectrum in parallel processes, `raman collect`
merges the results into Parquet with a run manifest, and `raman export` writes the legacy
per-window CSVs or a flat CSV. `raman status`, `cancel`, `runs import` and `doctor` complete
the command line; every command takes `--json`. The mock engine is a deterministic stand-in
with the real engine's interface, so its presence probabilities say nothing about the
sample. The Slurm backend (Sherlock) is the next phase; the real Bayesian engine comes after.

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

## Full-data smoke

A check of throughput and plumbing on the full sample archives, run by hand (not in CI):
both archives, 5 datasets, 11,250 spectra, each fitted with the mock engine in the two
windows of `configs/priors/nmc_commercial.yaml`. It should take under 3 minutes on a laptop.
The scans carry no detectable carbon, so this shows the pipeline works, not carbon detection.

With `NMC811-Exported-Data` and `NMC111-Exported-Data` unzipped into `data/raw/`, and the
state kept apart from your own under `data/smoke/` (gitignored):

```bash
export RAMAN_PATHS__DATA_ROOT="$PWD/data/smoke/datasets"
export RAMAN_PATHS__RUNS_ROOT="$PWD/data/smoke/runs"
export RAMAN_PATHS__CATALOG="$PWD/data/smoke/catalog.sqlite"
pixi run raman ingest data/raw/NMC811-Exported-Data   # NMC811_v0, v1, v2
pixi run raman ingest data/raw/NMC111-Exported-Data   # NMC111_v1, v2
pixi run raman datasets list                          # 5 datasets, 11,250 spectra in all
datasets=$(pixi run -q raman datasets list --json | pixi run -q python -c \
  "import json, sys; print(*(d['dataset_id'] for d in json.load(sys.stdin)['data']['datasets']))")
for dataset in $datasets; do
  run=$(pixi run -q raman run --dataset "$dataset" --priors configs/priors/nmc_commercial.yaml \
    --json | pixi run -q python -c "import json, sys; print(json.load(sys.stdin)['data']['run_id'])")
  pixi run raman collect "$run"
done
```

Every run should end complete and collected: 22,500 fits and 45,000 peak rows, with
NMC811_v1's one saturated spectrum `skipped_qc` in both windows. Then compare the
presence-probability distributions of NMC111 and NMC811, which should be alike despite the
roughly threefold difference in intensity (DuckDB is in the development environment):

```bash
pixi run python -c "import duckdb; duckdb.sql(\"SELECT substr(dataset_id, 1, 6) AS sample, \
  peak_id, median(presence_prob) AS median_presence \
  FROM 'data/smoke/runs/*/results/peaks.parquet' WHERE fit_status = 'ok' \
  GROUP BY ALL ORDER BY peak_id, sample\").show()"
```

## Citation

See [CITATION.cff](https://github.com/svangara-stanford/raman-bayes/blob/main/CITATION.cff).

## License

Pending: BSD-3-Clause is awaiting clearance from Stanford OTL. Until then no license is
granted.
