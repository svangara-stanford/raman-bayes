# raman-bayes — Design Brief (v0.2, trimmed)

**Status:** approved for Phase 0–2 implementation. Engine, GUI, MCP, LLM helpers and Globus commands are out of scope.

## 1. Purpose

Rebuild the Bayesian Raman peak-detection pipeline as a clean, testable Python package that runs identically on a laptop, the lab workstation and Stanford Sherlock (Slurm). The pipeline has five stations:

```
 raw .txt files ──► 1 PACK ──► dataset.zarr
 priors.yaml  ───► 2 RECIPE (validate)
 dataset + recipe ► 3 SPLIT & FIT ──► per-task Parquet shards   (parallel; Slurm array or local pool)
 shards ─────────► 4 MERGE ──► results Parquet + run manifest
 results ────────► 5 LOOK ──► CLI status / export (CSV)
 catalog.sqlite records every dataset and run.
```

Only station 3 runs in parallel and only station 3 writes to `shards/`. Everything else is a single process that owns `catalog.sqlite`. This rule is what keeps Lustre and SQLite from fighting.

The Bayesian engine itself is **not** built now. A `MockEngine` with the same interface stands in so the whole pipeline can be exercised end to end. The real engine is a separate, later phase.

## 2. Repository layout

```
raman-bayes/
├── pyproject.toml            # PEP 621, hatchling; console script `raman`
├── pixi.toml / pixi.lock     # env + lockfile (linux-64, osx-arm64, osx-64)
├── README.md  LICENSE(BSD-3, pending OTL)  CITATION.cff  CHANGELOG.md
├── .pre-commit-config.yaml   # ruff, ruff-format, mypy, check-yaml, gitleaks
├── .github/workflows/ci.yml  # lint, type, test (py3.11–3.13, ubuntu+macos)
├── src/raman_bayes/
│   ├── domain/        # Pydantic models + validation. No I/O.
│   │   ├── priors.py         PriorsSpec, Window, Peak, EndMember
│   │   ├── dataset.py        DatasetManifest, ScanInfo
│   │   ├── run.py            RunConfig, RunPlan, TaskSpec
│   │   └── results.py        PeakPosterior, WindowFitResult, PosteriorSummary
│   ├── ports/         # typing.Protocol interfaces: PeakEngine, ExecutionBackend
│   ├── io/
│   │   ├── nanofinder.py     Info.txt parser + two-column spectrum reader
│   │   ├── filename.py       index regex parser
│   │   └── legacy_csv.py     import/export of the old priors CSV format
│   ├── storage/
│   │   ├── zarr_dataset.py   write/read dataset.zarr
│   │   ├── shards.py         atomic Parquet shard writer, _DONE markers
│   │   ├── merge.py          shards -> results/*.parquet
│   │   └── catalog.py        SQLite catalog (single-writer)
│   ├── preprocessing/        clip, despike (median/MAD), ALS baseline, savgol
│   ├── engine/
│   │   ├── registry.py       entry-point discovery
│   │   └── mock.py           MockEngine (deterministic, microseconds)
│   ├── orchestration/
│   │   ├── chunking.py       spectra -> task ranges
│   │   ├── local.py          LocalBackend (ProcessPoolExecutor, spawn)
│   │   ├── slurm.py          SlurmArrayBackend (render sbatch, submit, poll)
│   │   └── templates/array.sbatch.j2
│   ├── app/           # use-cases; take/return domain objects; never print
│   │   ingest.py validate.py plan.py submit.py run_task.py status.py collect.py export.py
│   ├── config/        # Pydantic Settings; profiles; env override RAMAN_*
│   └── cli/           # Typer + Rich; every command has --json; calls app/ only
├── configs/
│   ├── profiles/{local.yaml, labpc.yaml, sherlock.yaml}
│   └── priors/
│       ├── nmc_commercial.yaml            # converted from the two CSVs below
│       └── legacy/{nmc_lithiation.csv, nmc_carbon.csv}
├── tests/
│   ├── unit/  integration/  regression/
│   └── fixtures/nanofinder_mini/NMC811_v0/   # Info.txt + 5x5 corner (26 files)
├── docs/              # MkDocs Material; this file lives at docs/DESIGN.md
└── data/              # GITIGNORED. data/raw/ holds the full test archives locally
```

**Import rule (enforced by import-linter in CI):** `cli/` imports only `app/` and `domain/`; `app/` never imports `cli/`. This is what lets a GUI be added later without re-architecture.

## 3. Input data: the Nanofinder export format (concrete)

Verified against `NMC811-Exported-Data.zip` and `NMC111-Exported-Data.zip`.

**Folder structure.** One folder per scan; each folder is one Dataset.
```
NMC811-Exported-Data/
├── NMC811_v0/  NMC811_v0-Info.txt  NMC811_v0-z0y0x0.txt … NMC811_v0-z0y24x49.txt   (50×25 = 1250)
├── NMC811_v1/  …                                                                 (50×50 = 2500)
└── NMC811_v2/  …                                                                 (50×50 = 2500)
```

**Spectrum file.** 1024 lines, tab-separated `shift_cm1 \t intensity`, CRLF line endings, no header, no NaN column. Intensities are raw CCD counts (float with `.00`). Axis is identical across all scans: −41.55539 … 1945.85360 cm⁻¹. The reader must accept LF or CRLF and must not assume the axis; it reads it from each file and checks it against the dataset axis (tolerance 0.05 cm⁻¹). A mismatch raises at ingest.

**Filename grammar.** `<scan>-z<iz>y<iy>x<ix>.txt`, unpadded integer **grid indices** (not microns).
```python
FILENAME_RE = r"^(?P<scan>.+?)-z(?P<iz>\d+)y(?P<iy>\d+)x(?P<ix>\d+)\.txt$"
```
Make the regex a manifest field with this as the default so other instruments can override it.

**`<scan>-Info.txt`.** Key/value text, CRLF. Parse at least:
| Field in file | Manifest key |
|---|---|
| `Date:`, `Time Start:`, `Time End:` | `acquired_at`, `duration_s` |
| `System name:`, `Detector name:` | `instrument.system`, `instrument.detector` |
| `Spectral size (number of pixels) :` | `n_wn` (assert 1024) |
| `Scanned points X/Y/Z :` | `grid.nx, ny, nz` |
| per-axis `Axis scan start/stop/step, [um]:` | `grid.{x,y,z}.{start_um, stop_um, step_um}` |
| `Axis position, [um]:` (when Z has 1 point) | `grid.z.start_um` |
| `Exposure time, [sec] =`, `Number in Accumulation =` | `acq.exposure_s`, `acq.n_accum` |
| `Temperature [grad C] =`, `Central Pixel =` | `acq.ccd_temp_c`, `acq.central_pixel` |

Physical coordinates: `x_um = x.start_um + ix * x.step_um` (same for y, z). Store both indices and µm.

**Validation at ingest.** File count == nx·ny·nz, no missing indices, no duplicates, axis consistent, no NaN/inf, all values ≥ 0. Flag saturated spectra (any value ≥ 65535 for this 16-bit CCD) in `qc_flags`.

## 4. Dataset store: `dataset.zarr`

Zarr v3, sharded. Flat layout (spectra as rows); the grid is reconstructed from `ix, iy, iz`.

```
<dataset_id>.zarr/
  zarr.json                  attrs: schema_version="1.0", manifest (full), content_hash, created_at
  intensity    float32  [n_spectra, 1024]   chunks (1024,1024)  shards (65536,1024)  zstd-3 + shuffle
  wavenumber   float64  [1024]
  spectrum_id  uint32   [n_spectra]          0..n-1 in raster order (iz, iy, ix)
  ix, iy, iz   int32    [n_spectra]
  x_um,y_um,z_um float64[n_spectra]
  source_file  str      [n_spectra]
  qc_flags     uint16   [n_spectra]          bit0 nan, bit1 saturated, bit2 axis_mismatch
```

For the test data (≤ 2500 spectra) a dataset is 1–3 chunks; that is fine. `dataset_id = "<scan>-<first 10 hex of BLAKE3 of sorted file digests + axis>"`. Ingest is idempotent: same input → same id → skip.

Raw files are not copied; on Sherlock, `ingest` optionally tars them to `archive_root` (`--archive`).

## 5. Priors recipe: `priors.yaml`

Replaces the per-window CSVs. One file describes all windows for a sample type.

```yaml
# configs/priors/nmc_commercial.yaml
schema_version: "1.0"
spec:
  name: nmc_commercial
  version: 1.0.0
  status: approved            # draft | approved ; production runs require approved
  source: "converted from [sample descriptor]_lithiation.csv and _carbon.csv (2026-07-06)"

defaults:
  presence: {prior: bernoulli, p: 0.35}
  lineshape: pseudo_voigt
  center:   {prior: normal_truncated, mean: explicit, sd: auto}   # mean from each peak
  fwhm:     {prior: uniform}                                      # bounds from each peak
  likelihood: {model: student_t, nu: 5}

preprocessing:                 # applied per window before the engine sees data
  despike:  {method: median_mad, kernel: 3, thresh: 3.0}
  baseline: {method: als, lam: 1.0e5, p: 0.01}
  smoothing: null
  clamp_nonnegative: true

end_members:
  - {id: lithiated_oxide, label: "NMC (lithiation-sensitive modes)", peaks: [u1, u2]}
  - {id: carbon,          label: "Carbon black / binder",          peaks: [d, g]}

windows:
  - id: lithiation
    range_cm1: [350, 700]
    peaks:
      - {id: u1, label: "ν1", center: {mean: 465, bounds: [445, 485]}, fwhm: {bounds: [35, 70]}}
      - {id: u2, label: "ν2", center: {mean: 568, bounds: [550, 585]}, fwhm: {bounds: [30, 60]}}
  - id: carbon
    range_cm1: [1150, 1750]
    peaks:
      - {id: d, label: "D", center: {mean: 1335, bounds: [1310, 1370]}, fwhm: {bounds: [90, 220]}}
      - {id: g, label: "G", center: {mean: 1584, bounds: [1548, 1620]}, fwhm: {bounds: [35, 125]}}

derived:
  - {id: u2_u1_area_ratio, expr: "area[u2] / area[u1]", require_presence: 0.5}
  - {id: id_ig,            expr: "area[d] / area[g]",   require_presence: 0.5}
```

**Validation rules** (`raman validate-priors`, exit 2 on error):
1. `schema_version` known; ids match `^[a-z][a-z0-9_]{0,31}$`, unique across the file.
2. Every peak's `center.bounds` lies inside its window's `range_cm1`; warn if within `fwhm.bounds[0]` of an edge.
3. `lower < upper` for all bounds; `center.mean` inside `center.bounds`; FWHM bounds > 0.
4. Windows must not overlap unless both set `allow_overlap: true`.
5. Overlapping center bounds inside one window → warning (identifiability) unless `ordered: true`.
6. `end_members[*].peaks` and `derived[*].expr` reference existing peak ids; every peak belongs to at most one end member.
7. At plan time: each window covers ≥ 32 channels on the dataset axis.
8. `priors_hash = sha256(canonical_json(parsed_model))`.

**Legacy CSV I/O.** `raman priors import-csv FILE --window-id W --range LO HI` → YAML fragment. Must strip UTF-8 BOM (the lithiation file has one), accept LF/CRLF, map `x (center)`→`center.mean`, limits→bounds. `export-csv` writes the legacy header `,x (center),x (lower limit),x (upper limit),FWHM (lower limit),FWHM (upper limit)`. Round-trip test on both committed CSVs.

## 6. Engine interface (built now; real engine later)

```python
@dataclass(frozen=True)
class PreparedSpectrum:
    spectrum_id: int
    wavenumber: np.ndarray      # already clipped to the window
    intensity: np.ndarray       # already preprocessed
    qc_flags: int

@dataclass(frozen=True)
class EngineConfig:
    n_iter: int = 6000; burn_in: int = 3000; thin: int = 5
    options: Mapping[str, Any] = field(default_factory=dict)

@runtime_checkable
class PeakEngine(Protocol):
    name: str
    version: str
    def validate(self, window: Window, cfg: EngineConfig) -> list[str]: ...
    def fit(self, s: PreparedSpectrum, window: Window, cfg: EngineConfig,
            rng: np.random.Generator) -> WindowFitResult: ...
    def fit_many(self, ss: Sequence[PreparedSpectrum], window: Window, cfg: EngineConfig,
                 seeds: Sequence[np.random.SeedSequence]) -> list[WindowFitResult]: ...
```

`fit_many` is what the orchestrator calls; default implementation loops over `fit`. Engines do no I/O, read no global config, never call `np.random.seed`, spawn no threads.

```python
@dataclass(frozen=True)
class PosteriorSummary:
    mean: float; median: float; sd: float; q025: float; q975: float
    at_bound_frac: float | None = None

@dataclass(frozen=True)
class PeakPosterior:
    peak_id: str
    presence_prob: float
    center: PosteriorSummary | None     # None when peak judged absent
    fwhm:   PosteriorSummary | None
    area:   PosteriorSummary | None
    height: PosteriorSummary | None
    eta:    PosteriorSummary | None

@dataclass(frozen=True)
class WindowFitResult:
    spectrum_id: int; window_id: str
    status: Literal["ok", "failed", "skipped_qc"]
    peaks: tuple[PeakPosterior, ...]
    diagnostics: Mapping[str, float]    # engine-defined; e.g. rmse, accept_rate, n_iter_run
    runtime_s: float
    engine: str; engine_version: str
    error: str | None = None
```

**Seeding.** `SeedSequence(run_seed, spawn_key=(window_index, spectrum_id))`. Independent of chunking and retries.

**MockEngine (`engine: mock`).** For each peak: center = argmax of intensity within `center.bounds`; fwhm = midpoint of bounds; area = sum of intensity within bounds; height = max; eta = 0.5; presence_prob = sigmoid of (height / robust noise) with a small seeded jitter; sd = 5% of value; q025/q975 = ±1.96 sd. Deterministic for a given seed. Runs in microseconds.

Engines are discovered via the `raman_bayes.engines` entry-point group so the real engine can later live in this repo or a sibling package.

## 7. Orchestration

### 7.1 Chunking and plan
`raman plan` produces `runs/<run_id>/run_plan.json` (frozen, hashed): dataset_id, priors_hash, engine + config, preprocessing, profile, seed, and a list of tasks `[{task_index, spec_start, spec_stop}]`. Task boundaries are multiples of 1024. Default target: `spectra_per_task` such that a task runs ≈ `target_task_minutes` (benchmark 20 fits at plan time; with the mock engine this is irrelevant, so also accept `--spectra-per-task`). Cap at `max_array_tasks` (500 on Sherlock; raise `spectra_per_task` if needed). `run_id` = UUIDv7.

### 7.2 `raman run-task --run RUN --task N`
The single entry point every backend executes. Reads its range from `run_plan.json`, loads that slice of the Zarr, loops over windows → preprocess → `fit_many` in sub-batches of 512, and writes `shards/task=N/part-k.parquet` via write-tmp → fsync → `os.replace`. On start it scans existing parts and resumes. Writes `_DONE.json` (counts, shard hash, host, slurm ids, versions) when finished. Per-spectrum errors become `status="failed"` rows; a non-zero exit means infrastructure failure. Never touches the catalog.

### 7.3 Backends
```python
class ExecutionBackend(Protocol):
    name: str
    def submit(self, plan: RunPlan, *, dry_run: bool = False) -> SubmissionRecord: ...
    def poll(self, run_id: str) -> Mapping[int, TaskState]: ...
    def cancel(self, run_id: str, tasks: Sequence[int] | None = None) -> None: ...
    def resubmit(self, run_id: str, tasks: Sequence[int]) -> SubmissionRecord: ...
```
- **LocalBackend:** `ProcessPoolExecutor(mp_context=spawn, max_workers=profile.local.workers)`, each worker runs `run_task` for one task. Ctrl-C keeps finished shards; rerun resumes.
- **SlurmArrayBackend:** renders the template, `sbatch --parsable`, records the array job id. `poll` = `_DONE.json` markers (source of truth) merged with `sacct -P -j <id> --format=JobID,State,Elapsed,MaxRSS`. `resubmit` submits `--array=3,17,42` for missing tasks only; TIMEOUT → `--time ×1.5`, OOM → `--mem ×1.5`, max 3 attempts.

```jinja
#!/bin/bash
#SBATCH --job-name=rb-{{ run_id[:8] }}
#SBATCH --partition={{ partitions|join(",") }}
#SBATCH --array=0-{{ n_tasks-1 }}%{{ max_concurrent }}
#SBATCH --cpus-per-task={{ cpus }}
#SBATCH --mem-per-cpu={{ mem_per_cpu }}
#SBATCH --time={{ walltime }}
#SBATCH --signal=B:TERM@120
#SBATCH --output={{ run_dir }}/logs/%A_%a.out
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export RAMAN_PROFILE={{ profile }}
cd {{ repo_dir }}
exec pixi run raman run-task --run {{ run_id }} --task "$SLURM_ARRAY_TASK_ID" --workers {{ cpus }}
```
Note: `--partition=normal,owners` with **no space** after the comma.

### 7.4 Sherlock facts baked into `sherlock.yaml`
Partitions `normal,owners`; `normal` max 2 days; `owners` jobs can be preempted and requeued (hence sub-batch checkpointing). `plan` queries `scontrol show config | grep MaxArraySize` and `sacctmgr show qos normal format=MaxSubmitJobsPerUser` live and refuses plans that exceed them. `$SCRATCH` purges files untouched for 90 days → `collect` copies results to `archive_root` on `$OAK` and `raman doctor` warns about datasets older than 60 days.

## 8. Results

### 8.1 Layout
```
runs/<run_id>/
  run_plan.json      submission.json      logs/
  shards/task=00000/part-000.parquet … _DONE.json
  results/
    peaks.parquet    fits.parquet    derived.parquet
    run_manifest.json
```

### 8.2 `peaks.parquet` (one row per spectrum × window × peak)
`run_id, dataset_id, spectrum_id:u32, ix, iy, iz:i32, x_um, y_um, z_um:f64, window_id, end_member_id, peak_id` (dictionary strings), `presence_prob:f32`, then for each of `center, fwhm, area, height, eta`: `_mean, _median, _sd, _q025, _q975` (f32, **null when absent** — never zero-filled), `center_at_bound_frac, fwhm_at_bound_frac:f32`, `fit_status`. Sorted by `spectrum_id`. Row-group size 64k. Parquet key-value metadata carries `run_manifest.json` and `schema_version`.

### 8.3 `fits.parquet` (one row per spectrum × window)
`run_id, spectrum_id, window_id, status, runtime_s, engine, engine_version, error` + flattened `diagnostics.*`.

### 8.4 `run_manifest.json`
`run_id, dataset_id, dataset_hash, priors_name, priors_version, priors_hash, engine, engine_version, engine_config, preprocessing, run_seed, backend, profile, package_version, git_sha, git_dirty, pixi_lock_hash, host, slurm_array_job_id, started_at, completed_at, n_spectra, n_tasks, n_failed, shard_hashes`.

### 8.5 Exports
- `raman export RUN --format legacy-csv --window lithiation --min-presence 0.90` → one CSV per window with header `x,y,<peak> center,<peak> height,<peak> width,<peak> confidence,…` reproducing the old layout. `--height-source area|height` (old files put area median under "height"), `--absent-fill 0`. x,y are grid indices to match old behaviour; `--coords um` switches.
- `raman export RUN --format csv` → `peaks.parquet` flattened to CSV.
- Direct access needs no CLI: `duckdb.sql("SELECT … FROM 'runs/*/results/peaks.parquet'")`.

### 8.6 Catalog (`catalog.sqlite`, single writer = the CLI driver)
```sql
dataset(dataset_id PK, scan, sample, content_hash, n_spectra, n_wn, store_uri, manifest_json, created_at)
priors_spec(priors_hash PK, name, version, status, yaml_text, created_at)
run(run_id PK, dataset_id, priors_hash, engine, engine_version, config_hash, git_sha, backend, profile,
    seed, status, created_at, completed_at, result_uri)
task(run_id, task_index, spec_start, spec_stop, status, attempts, slurm_job_id, n_ok, n_failed, error,
     PRIMARY KEY(run_id, task_index))
```
Rollback journal (not WAL) on Sherlock because the file lives on a network filesystem. `raman runs import DIR` registers a run folder copied from another machine (folder is self-describing via `run_manifest.json`).

## 9. Configuration

Pydantic Settings. Precedence: defaults < profile YAML < `RAMAN_*` env < CLI flags. Profile chosen by `--profile` or `RAMAN_PROFILE`. The resolved config is frozen into `run_plan.json`.

```yaml
# configs/profiles/local.yaml  (laptop development)
profile: local
paths: {data_root: ./data/datasets, runs_root: ./data/runs, catalog: ./data/catalog.sqlite}
backend: local
local: {workers: 8}
engine: {name: mock}
```
```yaml
# configs/profiles/labpc.yaml
profile: labpc
paths: {data_root: /data/raman/datasets, runs_root: /data/raman/runs, archive_root: /archive/raman,
        catalog: /data/raman/catalog.sqlite}
backend: local
local: {workers: 48}
engine: {name: mock}
```
```yaml
# configs/profiles/sherlock.yaml
profile: sherlock
paths: {data_root: "${SCRATCH}/raman/datasets", runs_root: "${SCRATCH}/raman/runs",
        archive_root: "${OAK}/raman", catalog: "${GROUP_HOME}/svangara/raman/catalog.sqlite",
        repo_dir: "${GROUP_HOME}/svangara/raman-bayes"}
backend: slurm_array
slurm: {partitions: [normal, owners], cpus_per_task: 16, mem_per_cpu: 4G, target_task_minutes: 60,
        max_array_tasks: 500, max_concurrent: 200}
engine: {name: mock}
limits: {max_core_hours_without_confirm: 2000}
```

## 10. CLI (Typer + Rich; `--json` everywhere; Rich disabled when stdout is not a TTY)

| Command | Service | Notes |
|---|---|---|
| `raman ingest PATH [--sample S] [--dry-run] [--archive] [--as-job]` | `ingest` | PATH may be a scan folder or a parent containing several; `--dry-run` prints parsed Info.txt, grid, first 10 filenames |
| `raman datasets list/show` | catalog | |
| `raman validate-priors FILE` | `validate` | exit 2 on error |
| `raman priors import-csv / export-csv` | `legacy_csv` | |
| `raman plan --dataset D --priors P [--engine mock] [--spectra-per-task N]` | `plan` | prints tasks, core-hour estimate, sbatch preview |
| `raman submit RUN [--dry-run | --confirm] [--resubmit-failed]` | `submit` | |
| `raman run --dataset D --priors P` | plan+submit+wait (local only) | one-shot for laptop/lab PC |
| `raman run-task --run RUN --task N` | `run_task` | invoked by backends, not humans |
| `raman status RUN [--watch] [--failures]` | `status` | markers + sacct |
| `raman cancel RUN --confirm` | backend | |
| `raman collect RUN [--as-job]` | `collect` | merge shards → results, archive, update catalog |
| `raman export RUN --format legacy-csv\|csv [--window W] [--min-presence P]` | `export` | |
| `raman runs list/show/import` | catalog | |
| `raman doctor` | — | env, paths, quotas (`sh_quota`), live Slurm limits, scratch age |

Exit codes: 0 ok · 2 validation · 3 nothing to do · 4 backend/infrastructure.

## 11. Testing

- **Unit:** PriorsSpec rules (Hypothesis for bounds), filename regex, Info.txt parser against the committed `NMC811_v0-Info.txt`, chunking, seed derivation, CSV round-trip, BOM/CRLF handling.
- **Fixture:** `tests/fixtures/nanofinder_mini/NMC811_v0/` = Info.txt + files for iz=0, iy∈0..4, ix∈0..4 (25 spectra). Carved from `data/raw/` by `scripts/make_fixture.py`. The Info.txt is the real one; tests assert the parser reads nx=50, ny=25 and the fixture loader tolerates the subset (`--allow-partial-grid`).
- **Integration:** ingest fixture → validate `nmc_commercial.yaml` → plan (2 tasks) → LocalBackend run with `mock` → collect → `export legacy-csv`. Assert row counts (25×2 windows×2 peaks), nulls where presence < threshold, manifest hashes present. Kill-and-resume test: SIGTERM mid-run, rerun, byte-identical results.
- **Full-data smoke (not in CI):** `raman ingest data/raw/NMC811-Exported-Data --sample NMC811` → 3 datasets, 6250 spectra; `raman run` on each with mock; total < 2 min on a laptop.
- **Slurm in CI:** shim scripts `sbatch`/`sacct`/`squeue` on PATH that run tasks locally and return scripted states (COMPLETED, PREEMPTED, TIMEOUT). Real Sherlock smoke test is manual, once per release.

## 12. Development workflow

Develop on a laptop/lab PC with `profile: local`; push to GitHub. On Sherlock: `git clone` into `$GROUP_HOME/svangara/raman-bayes`, `export PIXI_CACHE_DIR=$GROUP_HOME/svangara/.pixi-cache`, `pixi install`. Use Claude Code on Sherlock (inside `sh_dev`) only for Phase 2 work that needs real Slurm. Compute nodes have no internet: all installs happen on login/dev nodes.

## 13. Phases and definition of done

| Phase | Builds | Done when |
|---|---|---|
| **0 Scaffold** | layout, pyproject, pixi, CI, pre-commit, stub domain models, import-linter, `raman --help` | fresh clone → `pixi run test` green on Linux + macOS |
| **1 Core pipeline (local)** | Nanofinder reader + Info.txt, ingest → Zarr, PriorsSpec + validator + CSV I/O, preprocessing, MockEngine, chunking, LocalBackend, shards, collect, exports, catalog, CLI | integration test passes; full NMC811/NMC111 smoke runs end-to-end on a laptop; legacy-csv export has the old column layout |
| **2 Slurm** | SlurmArrayBackend, sbatch template, live-limit checks, status/cancel/resubmit, fake-Slurm CI shims, `doctor` | 6250-spectrum mock run on `normal,owners` completes; simulated preemption loses ≤ 1 sub-batch; results identical to LocalBackend |
| **3 (later) Real engine** | your redesigned sampler as a `PeakEngine` plugin | golden tests; `engine: mock` → `engine: <yours>` in profile is the only change |
| **4+ (deferred)** | GUI, Apptainer image, Globus sync, result cube, MCP, open-source release | — |

## 14. Where to put the test data

- Unzip `NMC111-Exported-Data.zip` and `NMC811-Exported-Data.zip` into `data/raw/` (laptop), `/data/raman/raw/` (lab PC) or `$SCRATCH/raman/raw/` (Sherlock). `data/` is gitignored.
- Copy `[sample descriptor]_lithiation.csv` → `configs/priors/legacy/nmc_lithiation.csv` and `_carbon.csv` → `configs/priors/legacy/nmc_carbon.csv` (committed, ~300 bytes).
- Claude Code generates `configs/priors/nmc_commercial.yaml` from them with windows `[350,700]` and `[1150,1750]`, and carves `tests/fixtures/nanofinder_mini/` from `data/raw/`.

## 15. Decision log

| # | Decision |
|---|---|
| 1 | Ingest once to sharded Zarr v3; raw `.txt` is archive-only. |
| 2 | PriorsSpec YAML with explicit windows replaces per-window CSVs; CSV kept as import/export. |
| 3 | Engine behind `PeakEngine` protocol; MockEngine now, real engine later as a plugin. |
| 4 | Own thin Slurm array backend (no Snakemake/Nextflow/Parsl); same `run-task` entry point for local and Slurm. |
| 5 | Tasks write only their own shard; the CLI driver is the only catalog writer. |
| 6 | Long Parquet is the system of record; nulls for absent peaks; legacy CSV via export only. |
| 7 | Pydantic Settings + YAML profiles (no Hydra). |
| 8 | pixi for environments; plain `pixi run` in sbatch (Apptainer deferred). |
| 9 | Coordinates: store grid indices and µm (from Info.txt); legacy export uses indices. |
| 10 | Deferred: GUI, MCP, LLM helpers, Globus commands, result cube, containers, PostgreSQL. |
