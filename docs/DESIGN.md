# raman-bayes — Design Brief (v0.4, adversarially reviewed)

**Status:** approved for Phase 0–2 implementation. The package is a plain PyPI wheel (§2 Packaging). Engine, GUI, MCP, LLM helpers, Globus commands and derived quantities are out of scope. §16 records what was reviewed and why.

## 1. Purpose

Rebuild the Bayesian Raman peak-detection pipeline as a clean, testable Python package that runs identically on a laptop, the lab workstation and Stanford Sherlock (Slurm). Five stations:

```
 raw .txt files ──► 1 PACK ──► dataset.zarr
 priors.yaml  ───► 2 RECIPE (validate)
 dataset + recipe ► 3 SPLIT & FIT ──► per-task Parquet shards   (parallel; Slurm array or local pool)
 shards ─────────► 4 MERGE ──► results Parquet + run manifest
 results ────────► 5 LOOK ──► CLI status / export (CSV)
 catalog.sqlite records every dataset and run.
```

Only station 3 runs in parallel, and only station 3 writes to `shards/`. Everything else is a single process that owns `catalog.sqlite`.

The Bayesian engine is **not** built now. A `MockEngine` with the same interface stands in so the pipeline can be exercised end to end. The real engine is a later phase.

## 2. Repository layout

```
raman-bayes/
├── pyproject.toml            # PEP 621, hatchling; console script `raman`
├── pixi.toml / pixi.lock     # env + lockfile (linux-64, osx-arm64, osx-64)
├── README.md  LICENSE(BSD-3, pending OTL)  CITATION.cff  CHANGELOG.md  CLAUDE.md
├── .pre-commit-config.yaml   # ruff, ruff-format, mypy, check-yaml, gitleaks
├── .github/workflows/ci.yml  # lint, type, import-linter, test (py3.11–3.13, ubuntu+macos)
├── src/raman_bayes/
│   ├── domain/        # Pydantic models + validation. No I/O.
│   │   ├── priors.py         PriorsSpec, Window, Peak, EndMember
│   │   ├── dataset.py        DatasetManifest, ScanInfo, GridSpec
│   │   ├── run.py            RunConfig, RunPlan, TaskSpec, RunStatus
│   │   └── results.py        PeakPosterior, WindowFitResult, PosteriorSummary, PEAKS_SCHEMA, FITS_SCHEMA
│   ├── ports/         # typing.Protocol: PeakEngine, ExecutionBackend
│   ├── io/
│   │   ├── nanofinder.py     section-aware Info.txt parser + two-column spectrum reader
│   │   ├── filename.py       index regex parser
│   │   └── legacy_csv.py     import/export of the old priors CSV format
│   ├── storage/
│   │   ├── zarr_dataset.py   write/read dataset.zarr
│   │   ├── shards.py         atomic Parquet shard writer, _LOCK/_DONE markers
│   │   ├── merge.py          shards -> results/*.parquet
│   │   └── catalog.py        SQLite catalog (single writer) + RunStatus transitions
│   ├── preprocessing/        clip(with margin), despike (median/MAD), ALS baseline, savgol, trim
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
│   ├── config/        # Pydantic Settings; ${VAR} expansion; RAMAN_* env override
│   │   └── profiles/local.yaml   # packaged default profile (importlib.resources)
│   ├── logging.py     # stdlib logging setup (console + per-task file)
│   └── cli/           # Typer + Rich; every command has --json; calls app/ only
├── schemas/           # JSON schemas generated from Pydantic (priors, manifest, cli outputs)
├── configs/
│   ├── profiles/{labpc.yaml, sherlock.yaml}   # site profiles, repo-level; local.yaml is packaged
│   └── priors/
│       ├── nmc_commercial.yaml
│       └── legacy/{nmc_lithiation.csv, nmc_carbon.csv}
├── scripts/make_fixture.py
├── tests/
│   ├── unit/  integration/  regression/
│   └── fixtures/
│       ├── nanofinder_mini/NMC811_v0/      # real Info.txt + 5x5 corner (26 files)
│       ├── info_txt/                       # the five real *-Info.txt files
│       └── legacy_output/                  # optional: an old counts CSV, if one exists
├── docs/              # MkDocs Material; DESIGN.md, DEVLOG.md, adr/
└── data/              # GITIGNORED. data/raw/ holds the full test archives locally
```

**Import rules (import-linter, CI).** Four contracts:
1. `cli/` imports only `app/`, `domain/`, `config/` and `raman_bayes.logging` (indirect imports via `app/` are fine).
2. Nothing outside `cli/` imports `raman_bayes.cli`.
3. `domain/` imports no other `raman_bayes` subpackage ("domain: no I/O").
4. `app/` does not import `rich` or `typer` ("app never prints"; ruff T20 enforces no `print` across `src/`).

**Packaging.** `pip install raman-bayes` must work without pixi on Linux and macOS (Windows untested). Runtime dependencies must be available as PyPI wheels; conda-only packages are not allowed. Any non-`.py` file the code reads at runtime (profile defaults, the sbatch template) lives under `src/raman_bayes/` and is loaded with `importlib.resources`, never by a path relative to the repo. `configs/` and `schemas/` are repo-level: user overrides and generated artefacts, never read by installed code. pixi is the developer environment; `pixi.lock` pins what developers and CI use, `pyproject.toml` declares what users install.

## 3. Input data: the Nanofinder export format

Verified against `NMC811-Exported-Data.zip` and `NMC111-Exported-Data.zip`.

**Folder structure.** One folder per scan; each folder is one Dataset.
```
NMC811-Exported-Data/
├── NMC811_v0/  NMC811_v0-Info.txt  NMC811_v0-z0y0x0.txt … z0y24x49.txt   (50×25 = 1250)
├── NMC811_v1/  …                                                      (50×50 = 2500)
└── NMC811_v2/  …                                                      (50×50 = 2500)
NMC111-Exported-Data/{NMC111_v1, NMC111_v2}                             (50×50 each)
```

**Spectrum file.** 1024 lines of `shift_cm1 \t intensity`, no header, no NaN column. Line endings are CRLF in the samples; the reader accepts LF or CRLF. Number formatting varies between scans (NMC111 files are 19920–19922 or 20944 bytes), so parse with `np.loadtxt`/`np.genfromtxt`, never by fixed width. Intensities are raw accumulated CCD counts. The axis in the samples is −41.55539 … 1945.85360 cm⁻¹, identical across scans, but the reader reads the axis from every file and checks it against the dataset axis (tolerance 0.05 cm⁻¹); a mismatch sets `qc_flags.axis_mismatch` and ingest fails unless `--allow-axis-mismatch`.

**Filename grammar.** `<scan>-z<iz>y<iy>x<ix>.txt`, unpadded integer **grid indices**.
```python
FILENAME_RE = r"^(?P<scan>.+?)-z(?P<iz>\d+)y(?P<iy>\d+)x(?P<ix>\d+)\.txt$"
```
Stored in the manifest as `filename_pattern.regex` with this default.

**`<scan>-Info.txt` is section-structured and internally inconsistent.** Keys like `Axis scan start` repeat once per axis, and in `NMC811_v0-Info.txt` the `Frame Dimentions` block says Z has 1 point at 42.68 µm while the `Frame Stage / Axis : 3` block says Z has 6 points from 75.06 to 79.05 µm. Only z0 files exist. Rules:
1. Parse by section (`Frame Header`, `Frame Options`, `Frame Dimentions`, `Frame Stage/Axis : N`, `Scan detection channel information/Channel : N`).
2. **`Frame Dimentions` is authoritative** for `nx, ny, nz` and the per-axis `start/stop/step [um]`. Its order is X, Y, Z. When an axis has 1 point, `Axis position, [um]` is its coordinate and step is 0.
3. Cross-check `nx·ny·nz` against the number of spectrum files found. Mismatch is an error unless `grid.allow_partial: true`.
4. Store the whole `Frame Stage` block raw under `manifest.instrument.raw_stage` for provenance; do not use it for coordinates.
5. `Spectral series number (maximum)` must be 1. Otherwise refuse ingest with "time series not supported yet".

Fields parsed into the manifest:

| Section / key | Manifest key |
|---|---|
| Header `Date:`, `Time Start:`, `Time End:` | `acquired_at`, `duration_s` |
| Header `System name:`, `Detector name:`, `Positioner name:` | `instrument.system/detector/positioner` |
| Dimentions `Spectral size (number of pixels) :` | `n_wn` |
| Dimentions `Scanned points X/Y/Z :` + `Axis scan start/stop/step, [um]:` / `Axis position, [um]:` | `grid.{nx,ny,nz}`, `grid.{x,y,z}.{start_um,stop_um,step_um}` |
| Dimentions `Spectral series number (maximum) :` | `n_series` (must be 1) |
| Channel `Exposure time, [sec] =`, `Number in Accumulation =` | `acq.exposure_s`, `acq.n_accum` |
| Channel `Temperature [grad C] =`, `Central Pixel =`, `Head model =` | `acq.ccd_temp_c`, `acq.central_pixel`, `acq.head_model` |

Coordinates: `x_um = grid.x.start_um + ix * grid.x.step_um` (same for y, z). Store both indices and µm.

**QC flags (uint16 bits):** 0 `nan_or_inf`, 1 `saturated`, 2 `axis_mismatch`, 3 `negative`. Saturation threshold = `acq.n_accum × 65535` (the Andor DU920P is 16-bit and counts are accumulated), **or** ≥ 3 consecutive identical samples at the spectrum maximum (flat top). Either condition sets the bit.

**Partial grids.** `grid.allow_partial` (manifest, default false; CLI `--allow-partial-grid`). When true, missing positions are recorded in `grid_mask` (bool `[nz, ny, nx]`) and `n_spectra` is the number of files found. Duplicates and files outside the grid are always errors.

## 4. Dataset store: `dataset.zarr`

Zarr v3, sharded, flat layout (spectra as rows). Pin `zarr>=3.0,<4`.

```
<dataset_id>.zarr/
  zarr.json                  attrs: schema_version="1.0", manifest (full), content_hash, created_at
  intensity    float32  [n_spectra, n_wn]   chunks (1024, n_wn)  shards (65536, n_wn)  zstd-3 + shuffle
  wavenumber   float64  [n_wn]
  spectrum_id  uint32   [n_spectra]          0..n-1 in raster order (iz, iy, ix)
  ix, iy, iz   int32    [n_spectra]
  x_um,y_um,z_um float64[n_spectra]
  source_file  str      [n_spectra]
  qc_flags     uint16   [n_spectra]
  grid_mask    bool     [nz, ny, nx]
```

Float32 is exact for integer counts < 2²⁴ (16.7 M); ingest asserts no count exceeds this.

**`dataset_id`** = `<scan>-<first 10 hex of BLAKE3(intensity bytes ‖ wavenumber bytes ‖ sorted source_file list)>`, computed **from the written arrays**, not from the raw files. Ingest is idempotent: an existing id → log and skip (exit 3). Raw files are not copied; `--archive` tars them to `archive_root` on Sherlock.

## 5. Priors recipe: `priors.yaml`

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
  center:   {prior: normal_truncated, sd_rule: quarter_range}   # sd = (upper-lower)/4 unless peak sets sd
  fwhm:     {prior: uniform}                                     # over the peak's bounds
  likelihood: {model: student_t, nu: 5}

preprocessing:                 # applied per window; see §7.2 for order
  margin_cm1: 25
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
```

Every hyperparameter is either a literal or a named rule defined in `domain/priors.py` (`quarter_range` is the only rule in v1). No value may be the string `auto`. `derived:` is reserved for Phase 3 and rejected by the v1 validator.

**Validation rules** (`raman validate-priors`, exit 2 on error, warnings to stderr):
1. `schema_version` known; ids match `^[a-z][a-z0-9_]{0,31}$`, unique across the file.
2. Each peak's `center.bounds` inside its window's `range_cm1`; warn if within `fwhm.bounds[0]` of an edge.
3. `lower < upper` for all bounds; `center.mean` inside `center.bounds`; FWHM bounds > 0; `fwhm.bounds[1]` ≤ window width.
4. Windows must not overlap unless both set `allow_overlap: true`.
5. Overlapping center bounds inside one window → warning unless `ordered: true`.
6. `end_members[*].peaks` reference existing peak ids; each peak in at most one end member.
7. `preprocessing.margin_cm1 ≥ 0`; `despike.kernel` odd.
8. At plan time: each window (without margin) covers ≥ 32 channels on the dataset axis; window ∪ margin lies within the axis.
9. `priors_hash = sha256(canonical_json(parsed_model))`.

**Legacy CSV I/O.** `raman priors import-csv FILE --window-id W --range LO HI` → YAML fragment. Strip UTF-8 BOM (`nmc_lithiation.csv` has one), accept LF/CRLF, map `x (center)`→`center.mean`, limits→bounds. `export-csv` writes header `,x (center),x (lower limit),x (upper limit),FWHM (lower limit),FWHM (upper limit)` with LF, no BOM. Round-trip test on both committed CSVs.

## 6. Engine interface (built now; real engine later)

```python
@dataclass(frozen=True)
class PreparedSpectrum:
    spectrum_id: int
    wavenumber: np.ndarray      # trimmed to the window (margin removed)
    intensity: np.ndarray       # preprocessed
    noise_sd: float             # robust noise estimate from preprocessing (MAD of residual)
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

`fit_many` is what the orchestrator calls; default loops over `fit`. Engines do no I/O, read no global config, never call `np.random.seed`, spawn no threads.

```python
@dataclass(frozen=True)
class PosteriorSummary:
    mean: float; median: float; sd: float; q025: float; q975: float
    at_bound_frac: float | None = None

@dataclass(frozen=True)
class PeakPosterior:
    peak_id: str
    presence_prob: float
    center: PosteriorSummary | None     # None when absent
    fwhm:   PosteriorSummary | None
    area:   PosteriorSummary | None
    height: PosteriorSummary | None
    eta:    PosteriorSummary | None

@dataclass(frozen=True)
class WindowFitResult:
    spectrum_id: int; window_id: str
    status: Literal["ok", "failed", "skipped_qc"]
    peaks: tuple[PeakPosterior, ...]
    diagnostics: Mapping[str, float]    # engine-defined
    runtime_s: float
    engine: str; engine_version: str
    error: str | None = None
```

**Seeding.** `SeedSequence(run_seed, spawn_key=(window_index, spectrum_id))`. Independent of chunking and retries.

**MockEngine (`mock`, version `0.1.0`).** Per peak: center = argmax within `center.bounds`; fwhm = midpoint of bounds; area = sum within bounds × channel width; height = max within bounds; eta = 0.5; `presence_prob = sigmoid((height / noise_sd − 3) / 2)` plus seeded N(0, 0.02) jitter clipped to [0, 1] — scale-free so NMC111 and NMC811 behave alike; sd = 5% of value; q025/q975 = ±1.96 sd. Peaks with `presence_prob < 0.5` return `None` summaries. Deterministic for a given seed.

Engines are discovered via the `raman_bayes.engines` entry-point group.

## 7. Orchestration

### 7.1 Chunking and plan
`raman plan` writes `runs/<run_id>/run_plan.json` (frozen, hashed): dataset_id, dataset_hash, priors_hash, engine + config, preprocessing, profile, resolved config, seed, tasks `[{task_index, spec_start, spec_stop}]`. Boundaries are multiples of 1024 except the last. `spectra_per_task` defaults from a 20-fit benchmark targeting `target_task_minutes`; `--spectra-per-task` overrides (needed for the mock). Cap at `max_array_tasks`. `run_id` = UUIDv7. Catalog row created with status `planned`.

### 7.2 `raman run-task --run RUN --task N`
The single entry point every backend executes.
1. Acquire `shards/task=N/_LOCK` (JSON: host, pid, started_at; `O_EXCL` create). If present and younger than `lock_stale_minutes` (default 30), exit 3 "already running". If stale, overwrite.
2. Load the task's slice of the Zarr.
3. For each window: **clip to `range ± margin_cm1` → despike → baseline → smoothing → clamp → trim margin → noise_sd = MAD(residual of a 9-point median filter)**. Spectra with `nan_or_inf` or `saturated` flags get `status="skipped_qc"` without fitting.
4. `fit_many` in sub-batches of 512; each sub-batch written as `part-<k>.parquet` via write `.tmp` → `fsync` → `os.replace`. Existing valid parts are skipped on restart (resume).
5. Write `_DONE.json` (counts, shard hash, host, Slurm ids, versions), remove `_LOCK`.
Per-spectrum exceptions become `status="failed"` rows with the message. Non-zero exit means infrastructure failure. Never touches the catalog. Logs to `runs/<id>/logs/task-N.log`.

### 7.3 Backends
```python
class ExecutionBackend(Protocol):
    name: str
    def submit(self, plan: RunPlan, *, dry_run: bool = False) -> SubmissionRecord: ...
    def poll(self, run_id: str) -> Mapping[int, TaskState]: ...
    def cancel(self, run_id: str, tasks: Sequence[int] | None = None) -> None: ...
    def resubmit(self, run_id: str, tasks: Sequence[int]) -> SubmissionRecord: ...
```
- **LocalBackend:** `ProcessPoolExecutor(mp_context=spawn, max_workers=profile.local.workers)`; each worker runs `run_task`. Ctrl-C keeps finished parts; rerun resumes.
- **SlurmArrayBackend:** renders the template, `sbatch --parsable`, records the array job id. `poll` = `_DONE.json` markers (source of truth) merged with `sacct -P -j <id> --format=JobID,State,Elapsed,MaxRSS`. `resubmit` targets only tasks with **no `_DONE.json` and no PENDING/RUNNING state in sacct** (Slurm's own requeue after preemption must not be duplicated); TIMEOUT → `--time ×1.5`, OUT_OF_MEMORY → `--mem ×1.5`; max 3 attempts.

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
exec {{ pixi_bin }} run raman run-task --run {{ run_id }} --task "$SLURM_ARRAY_TASK_ID" --workers {{ cpus }}
```
`pixi_bin` is an **absolute path** (default `$HOME/.pixi/bin/pixi`, set in the profile) because batch shells do not source `~/.bashrc`. `--partition=normal,owners` has **no space** after the comma. The `--signal` handler must be verified on Sherlock in Phase 2 (with `exec`, the signal reaches Python directly).

### 7.4 Run status state machine (enforced in `catalog.py`)
```
planned ──submit──► submitted ──first task starts──► running ──all _DONE──► complete ──collect──► collected
   │                    │                                 │
   └──cancel──► cancelled ◄───────────────────────────────┘        any ──► failed (max attempts exhausted)
```
`collect` refuses unless status is `complete`, or `--partial` is given (then results carry `partial: true` in the manifest). `submit` refuses unless `planned` or (with `--resubmit-failed`) `running`/`failed`.

### 7.5 Sherlock facts baked into `sherlock.yaml`
Partitions `normal,owners`; `normal` max 2 days; `owners` jobs can be preempted and requeued. `plan` queries `scontrol show config | grep MaxArraySize` and `sacctmgr show qos normal format=MaxSubmitJobsPerUser -P` live and refuses plans exceeding them. `$SCRATCH` purges files untouched for 90 days → `collect` copies results to `archive_root`, `doctor` warns about datasets older than 60 days.

## 8. Results

### 8.1 Layout
```
runs/<run_id>/
  run_plan.json  submission.json  logs/
  shards/task=00000/_LOCK | part-000.parquet … _DONE.json
  results/peaks.parquet  fits.parquet  run_manifest.json
```

### 8.2 Schemas are explicit constants
`PEAKS_SCHEMA` and `FITS_SCHEMA` are `pyarrow.Schema` objects in `domain/results.py`. **Every shard and every merged file is written with the explicit schema**, so an all-null column is still `float32`, never pyarrow's inferred `null` type. Merge asserts schema equality before concatenating.

**`peaks.parquet`** (one row per spectrum × window × peak): `run_id, dataset_id` (dictionary str), `spectrum_id:u32, ix, iy, iz:i32, x_um, y_um, z_um:f64, window_id, end_member_id, peak_id` (dictionary str), `presence_prob:f32`, and for each of `center, fwhm, area, height, eta`: `_mean, _median, _sd, _q025, _q975` (f32, **null when absent**), `center_at_bound_frac, fwhm_at_bound_frac:f32`, `fit_status` (dictionary str). Sorted by `spectrum_id`; row groups of 64k. Key-value metadata: `run_manifest.json`, `schema_version`.

**`fits.parquet`** (one row per spectrum × window): `run_id, spectrum_id, window_id, status, runtime_s, noise_sd, engine, engine_version, error` + `diag_*` columns (float32; unknown diagnostics become new `diag_*` columns at merge).

### 8.3 `run_manifest.json`
`run_id, dataset_id, dataset_hash, priors_name, priors_version, priors_hash, engine, engine_version, engine_config, preprocessing, run_seed, backend, profile, package_version, git_sha, git_dirty, pixi_lock_hash, host, slurm_array_job_id, started_at, completed_at, n_spectra, n_tasks, n_failed, n_skipped_qc, partial, shard_hashes`.

### 8.4 Exports
- `raman export RUN --format legacy-csv --window W --min-presence 0.90 [--height-source area|height] [--absent-fill 0] [--coords index|um]`: one CSV per window with header `x,y,<peak> center,<peak> height,<peak> width,<peak> confidence,…` in peak order of the YAML. Defaults reproduce old behaviour (area median under "height", zero-fill, grid indices).
- `raman export RUN --format csv`: `peaks.parquet` flattened.
- Direct: `duckdb.sql("SELECT … FROM 'runs/*/results/peaks.parquet'")`.

### 8.5 Catalog (`catalog.sqlite`, single writer = CLI driver)
```sql
dataset(dataset_id PK, scan, sample, content_hash, n_spectra, n_wn, store_uri, manifest_json, created_at)
priors_spec(priors_hash PK, name, version, status, yaml_text, created_at)
run(run_id PK, dataset_id, priors_hash, engine, engine_version, config_hash, git_sha, backend, profile,
    seed, status, created_at, completed_at, result_uri)
task(run_id, task_index, spec_start, spec_stop, status, attempts, slurm_job_id, n_ok, n_failed, error,
     PRIMARY KEY(run_id, task_index))
```
Rollback journal (not WAL) on Sherlock. `raman runs import DIR` registers a run folder copied from elsewhere.

## 9. Configuration

Pydantic Settings. Precedence: defaults < profile YAML < `RAMAN_*` env < CLI flags. `--profile` (or `RAMAN_PROFILE`) is a **name** or a **path**. A name resolves first to the packaged `raman_bayes/config/profiles/<name>.yaml`, then to `./configs/profiles/<name>.yaml`; a path is used as given. Default is `local`. Resolved config is frozen into `run_plan.json`.

**`${VAR}` expansion:** only `${NAME}` syntax, expanded at load time; an unset variable is a hard error naming the variable and the profile key. Sherlock sets `SCRATCH` and `GROUP_HOME`; **Oak is not an environment variable**, so `archive_root` is a literal path you fill in.

```yaml
# raman_bayes/config/profiles/local.yaml  (packaged default; laptop)
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
        archive_root: "/oak/stanford/groups/<PI_GROUP>/raman",      # fill in; not an env var
        catalog: "${GROUP_HOME}/svangara/raman/catalog.sqlite",
        repo_dir: "${GROUP_HOME}/svangara/raman-bayes"}
backend: slurm_array
slurm: {partitions: [normal, owners], cpus_per_task: 16, mem_per_cpu: 4G, target_task_minutes: 60,
        max_array_tasks: 500, max_concurrent: 200, pixi_bin: "${HOME}/.pixi/bin/pixi",
        lock_stale_minutes: 30}
engine: {name: mock}
limits: {max_core_hours_without_confirm: 2000}
```

## 10. CLI (Typer + Rich; `--json` everywhere; Rich off when stdout is not a TTY)

| Command | Notes |
|---|---|
| `raman ingest PATH [--sample S] [--dry-run] [--archive] [--allow-partial-grid] [--as-job]` | PATH = scan folder or parent of several; `--dry-run` prints parsed sections, grid, first 10 filenames, file-count check |
| `raman datasets list/show` | |
| `raman validate-priors FILE` | exit 2 on error |
| `raman priors import-csv / export-csv` | |
| `raman plan --dataset D --priors P [--engine mock] [--spectra-per-task N]` | prints tasks, core-hour estimate, sbatch preview |
| `raman submit RUN [--dry-run \| --confirm] [--resubmit-failed]` | |
| `raman run --dataset D --priors P` | plan + submit + wait (local backend only) |
| `raman run-task --run RUN --task N` | invoked by backends |
| `raman status RUN [--watch] [--failures]` | markers + sacct |
| `raman cancel RUN --confirm` | |
| `raman collect RUN [--partial] [--as-job]` | refuses unless `complete` or `--partial` |
| `raman export RUN --format legacy-csv\|csv …` | |
| `raman runs list/show/import` | |
| `raman doctor` | env, paths, `${VAR}` resolution, pixi_bin exists, quotas, live Slurm limits, scratch age |

Exit codes: 0 ok · 2 validation · 3 nothing to do / already running · 4 backend or infrastructure. `--json` outputs validate against schemas in `schemas/cli/`.

## 11. Testing

**Fixtures.** `tests/fixtures/nanofinder_mini/NMC811_v0/` = the real `NMC811_v0-Info.txt` + files for iz=0, iy∈0..4, ix∈0..4 (25 spectra), carved by `scripts/make_fixture.py`; ingested with `allow_partial`. `tests/fixtures/info_txt/` = all five real Info.txt files. `tests/fixtures/legacy_output/` = an old counts CSV from the previous pipeline **if one can be found**; otherwise the legacy header is compared to a committed string.

**Unit.**
- Info.txt parser on all five files: correct `nx, ny, nz` (50×25×1 and 50×50×1), `Frame Dimentions` wins over `Frame Stage` for Z, `n_accum=5`, `n_series=1`; a modified copy with `n_series=2` is refused.
- Spectrum reader on NMC811 **and** NMC111 files (differing byte sizes), LF and CRLF, BOM-free; axis check tolerance.
- Filename regex: valid, malformed, and a different-instrument override.
- Grid completeness: one missing file, one duplicate, one out-of-grid file → three distinct errors; `allow_partial` turns the first into a mask.
- QC: synthetic saturated (≥ n_accum·65535) and flat-topped spectra set the bit; a tall legitimate peak does not.
- PriorsSpec rules 1–8 (Hypothesis for bounds); `auto` rejected; `derived:` rejected in v1.
- CSV import/export round-trip on both legacy files; BOM and CRLF handling.
- Zarr round-trip bit-identical; `dataset_id` deterministic; second ingest is a no-op (exit 3).
- Chunking: random n → ranges partition `0..n` exactly, boundaries multiples of 1024, respects cap.
- Seed derivation independent of task boundaries.
- Preprocessing: injected single-channel spike removed, adjacent real peak preserved; synthetic sloped baseline flattened within tolerance; margin trim returns exactly the window channels.
- Shard writer: `.tmp` never visible as final; interrupted write leaves no partial final file; all-null column keeps `float32`.
- Lock: second `run-task` on a held lock exits 3; stale lock is taken over.
- Config: `${VAR}` expansion; unset var → error naming var and key; precedence order.
- Catalog state machine: every illegal transition raises.

**Integration.**
- Fixture end-to-end: ingest → validate → plan (2 tasks) → LocalBackend → collect → both exports. Assert 25×2×2 peak rows, nulls where absent, explicit schema, manifest hashes present.
- **Seed invariance (key test):** same fixture with `--spectra-per-task 1024` vs `--spectra-per-task 8` → byte-identical `peaks.parquet` after sorting.
- Kill-and-resume: SIGTERM mid-run, rerun, identical results; no duplicate rows.
- Concurrent duplicate: two `run-task` on the same task; one exits 3; results intact.
- Merge with one all-null shard succeeds.
- `collect` on an incomplete run refuses; `--partial` succeeds and marks the manifest.
- CLI `--json` outputs validate against schemas; exit codes per §10.

**Full-data smoke (not in CI, documented in README).** Ingest **both** `NMC811-Exported-Data` and `NMC111-Exported-Data` (5 datasets, 11,250 spectra); run each with `mock`; < 3 min on a laptop. Presence-probability distributions should be comparable between NMC111 and NMC811 despite the ~3× intensity difference.

**Wheel install in CI.** One ubuntu job builds the wheel (`python -m build`), installs it with `pip install dist/*.whl` into a plain venv (no pixi), and runs `raman --help` and `python -c "import raman_bayes"`. This catches files or dependencies missing from the wheel.

**Slurm in CI.** Shim `sbatch`/`sacct`/`squeue` on PATH that run tasks locally and return scripted states (COMPLETED, PREEMPTED, TIMEOUT, OUT_OF_MEMORY). Real Sherlock smoke test is manual, once per release.

## 12. Development workflow

Develop on a laptop/lab PC with `profile: local`; push to GitHub. On Sherlock: `git clone` into `$GROUP_HOME/svangara/raman-bayes`, `export PIXI_CACHE_DIR=$GROUP_HOME/svangara/.pixi-cache`, `pixi install`. Claude Code on Sherlock (inside `sh_dev`) only for Phase 2. Compute nodes have no internet; installs happen on login/dev nodes.

For each sub-step: failing tests first, then implementation; stop with a five-line summary (built, tested, unsure, deviations, next). Record decisions in `docs/DEVLOG.md`; design changes get an ADR in `docs/adr/`.

## 13. Phases and definition of done

| Phase | Builds | Done when |
|---|---|---|
| **0 Scaffold** | layout, pyproject, pixi, CI incl. wheel-install job, pre-commit, stub models, import-linter (4 contracts), `raman --help`, logging setup, packaged `local.yaml` (file only; loading is Phase 1) | fresh clone → `pixi run test` green on Linux + macOS; wheel installs and `raman --help` runs in a plain venv |
| **1 Core pipeline (local)** | §3 reader/parser, fixture script, §5 PriorsSpec + CSV I/O + `nmc_commercial.yaml`, §4 ingest, preprocessing, MockEngine, chunking, run-task with lock/resume, LocalBackend, collect, exports, catalog with state machine, CLI | all §11 unit + integration tests pass including seed invariance; full-data smoke on both samples; legacy-csv header matches |
| **2 Slurm** | SlurmArrayBackend, template with absolute `pixi_bin`, live-limit checks, status/cancel/resubmit, shims, `doctor` | 6250-spectrum mock run on `normal,owners` completes; simulated preemption loses ≤ 1 sub-batch; no duplicate rows; results identical to LocalBackend; `--signal` handler verified |
| **3 (later) Real engine** | your sampler as a `PeakEngine` plugin; `derived:` quantities | golden tests; profile change only |
| **4+ (deferred)** | GUI, Apptainer, Globus, result cube, MCP, time series, release | — |

## 14. Where to put the test data

- Unzip both archives into `data/raw/` (laptop), `/data/raman/raw/` (lab PC) or `$SCRATCH/raman/raw/` (Sherlock). `data/` is gitignored.
- `[sample descriptor]_lithiation.csv` → `configs/priors/legacy/nmc_lithiation.csv`; `_carbon.csv` → `configs/priors/legacy/nmc_carbon.csv`.
- Claude Code generates `configs/priors/nmc_commercial.yaml` (windows `[350,700]`, `[1150,1750]`), carves `tests/fixtures/nanofinder_mini/`, and copies the five Info.txt files to `tests/fixtures/info_txt/`.

## 15. Decision log

| # | Decision |
|---|---|
| 1 | Ingest once to sharded Zarr v3; raw `.txt` is archive-only. |
| 2 | PriorsSpec YAML with explicit windows replaces per-window CSVs; CSV kept as import/export. |
| 3 | Engine behind `PeakEngine` protocol; MockEngine now, real engine later as a plugin. |
| 4 | Own thin Slurm array backend; same `run-task` entry point for local and Slurm. |
| 5 | Tasks write only their own shard; the CLI driver is the only catalog writer. |
| 6 | Long Parquet with explicit schema is the system of record; nulls for absent peaks; legacy CSV via export only. |
| 7 | Pydantic Settings + YAML profiles with `${VAR}` expansion (no Hydra). |
| 8 | pixi for environments; absolute `pixi_bin` in sbatch (Apptainer deferred). |
| 9 | Coordinates: grid indices and µm from `Frame Dimentions`; legacy export uses indices. |
| 10 | `dataset_id` hashed from written arrays, not raw files. |
| 11 | Run lifecycle is an explicit state machine; `collect` requires `complete` or `--partial`. |
| 12 | Deferred: GUI, MCP, LLM helpers, Globus, result cube, containers, PostgreSQL, derived quantities, time series. |
| 13 | Plain-pip installable: PyPI-wheel deps only; runtime files loaded via `importlib.resources`; `local.yaml` packaged, site profiles repo-level; wheel-install CI job. |

## 16. Review log

### v0.3 → v0.4 (packaging)
- Decision 13. §2 Packaging paragraph and four import contracts written out; §9 profile-name resolution; §11 wheel-install job; §13 Phase 0 row. No change to architecture or data formats.

### v0.2 → v0.3

Adversarial review against the real sample data. Changes made:
- **Info.txt**: discovered `Frame Dimentions` vs `Frame Stage` Z contradiction in `NMC811_v0-Info.txt`; parser is now section-aware with `Frame Dimentions` authoritative (§3).
- **Saturation**: threshold scaled by `n_accum` (=5 in samples) plus flat-top detection (§3).
- **Parquet**: explicit `pyarrow.Schema` everywhere to prevent null-type inference on all-absent shards (§8.2).
- **Slurm template**: absolute `pixi_bin` because batch shells skip `~/.bashrc` (§7.3).
- **Config**: strict `${VAR}` expansion; Oak path is a literal (§9).
- **Double execution**: `_LOCK` file + resubmit excludes PENDING/RUNNING tasks (§7.2, §7.3).
- **dataset_id**: hashed from arrays (§4).
- **State machine** and `collect --partial` (§7.4).
- **Preprocessing order** with `margin_cm1` (§7.2); `noise_sd` passed to engine; mock presence made scale-free (§6).
- **Schema hygiene**: `auto` replaced by named rules; `derived:` deferred (§5).
- **Partial grids** first-class with `grid_mask`; `n_series` guard (§3, §4).
- **Logging** restored minimally (§7.2).
- **Tests** added: five Info files, NMC111 reader, grid errors, QC, all-null shard, lock, config expansion, state machine, **seed invariance**, concurrent duplicate, partial collect, CLI schemas (§11).

Considered and deliberately **not** changed: distributed locks or a task-lease DB (file lock suffices); structlog (stdlib is enough for now); a safe expression evaluator for `derived` (deferred instead); resampling spectra at ingest (axes are identical in samples; mismatch is a flagged error); Zarr float32 (exact for counts < 2²⁴, asserted).
