# Phase 1 plan

Status: superseded by CLAUDE.md (Batch mode) for branching, PRs and order of work, and by `docs/plans/phase-1-tests.md` for tests. Kept as the record of the Phase 1 data findings.
Date: 2026-10-29. Base: `main` at `ee0980c` (Phase 0 complete; CI green on all 8 jobs).

Phase 1 is the end-to-end pipeline with the mock engine: ingest, priors, plan, run-task,
local backend, collect, export. It is done when the DESIGN.md §13 Phase 1 criteria pass.

## 1. Data findings

Surveyed in `data/raw/` (gitignored; never read by unit tests).

### 1.1 `n_accum` is not 5 in every file

§11 says the Info.txt parser test asserts `n_accum=5` on all five files, and the v0.2 → v0.3
entry in §16 says "(=5 in samples)". Only three of the five files have 5:

| Scan | `Number in Accumulation` | `Exposure time, [sec]` | `Cycle time [sec]` |
|---|---|---|---|
| NMC111_v1 | 1 | 1.0000 | 1.0042 |
| NMC111_v2 | 5 | 0.2000 | 0.0000 |
| NMC811_v0 | 5 | 0.2000 | 0.2862 |
| NMC811_v1 | 5 | 0.2000 | 0.2978 |
| NMC811_v2 | 10 | 0.2000 | 0.2862 |

This matters because the saturation threshold is `acq.n_accum × 65535` (§3). Tests will assert
the real value per file. The §11 and §16 patches are in section 6.

### 1.2 Info.txt layout differs from §3 in three places

All five files share one layout (CRLF, ASCII; same line numbers for every section). NMC811 files
are 104 lines long; NMC111 files are 102. The NMC111 files lack the last two lines of the
`Channel : 0` subsection, `Center Row = 136` and `Track Height = 15`. This is a readout-mode
difference: NMC111 is `Readout Mode: Full Vertical Binning` and NMC811 is
`Readout Mode: Single Track`, and only Single Track readout has a center row and track height.
None of the three keys (`Readout Mode`, `Center Row`, `Track Height`) is stored, and the parser
must not require `Center Row` or `Track Height` (decided 2026-10-29). An earlier draft of this
plan said all five files were 104 lines.
The parser must handle these three differences from §3:

1. **Section structure.**
   - Line 1 is the export timestamp (`6/9/2026 1:04:44 PM`) and line 2 is the title
     `Scan common information`. Neither is a section.
   - The `Frame Header:`, `Frame Options:`, `Frame Dimentions:` and `Frame Stage:` headings end
     in `:`, but `Scan detection channel information` has no colon.
   - That section has a preamble before its `Channel : 0` subsection: `Device Name : Andor CCD`
     and `Channels :1`. The plural `Channels :1` must not be read as a subsection.
   - `Frame Stage:` subsections are `Axis : 1`, `Axis : 2`, `Axis : 3`. They contain flag lines
     with no separator (`Axis is present`, `Axis in use`, `Axis is fast`).
   - Blank lines contain a single space (`" \r\n"`), 7 per file.
   - Separators vary: `Date: `, `Spectral size (number of pixels) : `,
     `Axis scan start, [um]: `, `Head model = `, `Channels :1`.
2. **Exposure shares a line with cycle time:**
   `Exposure time, [sec] = 0.2000 Cycle time [sec] = 0.2862`.
3. **Accumulation count shares a line with acquisition mode:**
   `Acquisition mode: Accomulate. Number in Accumulation = 5`.

### 1.3 Other format facts

- Header `Date:` is `M/D/YYYY` (`5/28/2026`). `Time Start:` and `Time End:` are 12-hour
  `h:mm:ss AM/PM` (`3:47:23 PM`). No scan crosses midnight, so `duration_s` is
  `Time End − Time Start` on `Date`. A negative result is an error, not a wrap.
- In `Frame Dimentions:`, the keys `Axis scan start/stop/step, [um]:` repeat once per axis in
  the order X, Y, Z. An axis with 1 point uses `Axis position, [um]:` instead (Z in every file).
- NMC811_v0's `Frame Stage` Z disagrees with `Frame Dimentions`: 6 points from 75.06 to 79.05 µm,
  against 1 point at 42.68 µm. This is the contradiction §3 already describes.
- Spectra are CRLF, tab-separated `shift\tintensity`, 1024 lines. NMC111_v1 files are
  19920–19922 bytes; NMC111_v2 and all NMC811 files are 20944 bytes. This matches §3.
- Legacy CSVs: `nmc_lithiation.csv` has a UTF-8 BOM, CRLF line endings and integer values.
  `nmc_carbon.csv` is ASCII with float values. Neither ends with a newline.

## 2. Roadmap

There is one branch and one PR per sub-step. Each sub-step starts with failing tests, and each
CLI command is wired in the same sub-step as the feature behind it.

| NN | Branch | Builds |
|---|---|---|
| 01 | `phase-1-01-fixtures` | `scripts/make_fixture.py`, `nanofinder_mini/NMC811_v0` (Info + 25 spectra), `info_txt/` (5 files), NMC111 spectra, `.gitattributes` |
| 02 | `phase-1-02-nanofinder-io` | `io/filename.py`, `io/nanofinder.py` (section parser that produces ScanInfo, spectrum reader, axis check). ScanInfo finalized (DESIGN patch first) |
| 03 | `phase-1-03-priors-rules` | §5 rules 1–7 and 9 (`quarter_range`, `priors_hash`), with Hypothesis tests; `app/validate.py` and `raman validate-priors` |
| 04 | `phase-1-04-legacy-csv` | `io/legacy_csv.py`, `priors import-csv`/`export-csv`, and `configs/priors/nmc_commercial.yaml` generated from the legacy CSVs |
| 05 | `phase-1-05-config` | `config/` (profile name/path resolution, strict `${VAR}`, `RAMAN_*`, precedence); `configs/profiles/{labpc,sherlock}.yaml` |
| 06 | `phase-1-06-catalog` | `storage/catalog.py` (§8.5 schema, rollback journal, §7.4 state machine); `datasets list/show`, `runs list/show` |
| 07 | `phase-1-07-zarr-ingest` | `storage/zarr_dataset.py`, `dataset_id`, QC flags, grid completeness and mask; `app/ingest.py` and `raman ingest` |
| 08 | `phase-1-08-preprocessing` | Clip with margin, despike, ALS baseline, savgol, clamp, trim, `noise_sd`. SmoothingSpec finalized (patch first) |
| 09 | `phase-1-09-mock-engine` | `engine/registry.py` (entry points), `engine/mock.py`, seed derivation |
| 10 | `phase-1-10-plan` | `orchestration/chunking.py`, `app/plan.py`, §5 rule 8 (plan-time window coverage), `run_plan.json`, UUIDv7, `raman plan`. RunConfig and RunPlan finalized (patch first) |
| 11 | `phase-1-11-run-task` | `storage/shards.py` (atomic parts, `_LOCK`/`_DONE`), `app/run_task.py`, resume, `raman run-task` |
| 12 | `phase-1-12-local-backend` | `orchestration/local.py`; `submit`, `status`, `cancel`, `run`. TaskState and SubmissionRecord finalized (patch first) |
| 13 | `phase-1-13-collect` | `storage/merge.py`, `run_manifest.json`, `--partial`, `raman collect` |
| 14 | `phase-1-14-export` | legacy-csv and csv export, `runs import`, local `doctor` |
| 15 | `phase-1-15-integration` | §11 integration suite (end to end, seed invariance, kill-and-resume, duplicate task, all-null merge, partial collect, CLI schemas); README full-data smoke; Phase 1 "done when" check |

Why this order:

- The catalog (06) comes before ingest (07) because ingest registers datasets in it.
- §5 rule 8 checks windows against a dataset axis, so it belongs with `plan` (10), not with the
  file-only validator (03).
- Config (05) comes before anything that resolves paths from a profile.

## 3. Decisions needed

### 3.1 Two runtime dependencies (CLAUDE.md: ask first)

| Dependency | Needed at | For | Why this one |
|---|---|---|---|
| `blake3` | 07 | `dataset_id` (§4) | §4 specifies BLAKE3. The standard library has no BLAKE3, and writing our own is not reasonable. PyPI wheels exist for Linux, macOS and Windows on 3.11–3.13; conda-forge has 1.0.11. |
| `pydantic-settings` | 05 | §9 settings and `RAMAN_*` precedence | It is already installed (2.16.1) as a transitive dependency, but importing it means it must be declared directly. It comes from the Pydantic project and ships pure-Python wheels. |

I'll add each one with `pixi add` in the sub-step that needs it, after you approve.

### 3.2 UUIDv7: own implementation, not a dependency

`uuid.uuid7` exists only from Python 3.14, and we support 3.11–3.13. RFC 9562 UUIDv7 is about
15 lines: a 48-bit Unix-millisecond timestamp, the version and variant bits, and random bits from
`os.urandom`. A small, tested function in sub-step 10 avoids adding `uuid-utils`, which is a
Rust extension, for one function. Tests will cover the version and variant bits, time ordering
across calls, and uniqueness. When the minimum Python version reaches 3.14, the function can be
swapped for `uuid.uuid7`.

### 3.3 Closing the two Phase 0 `--json` gaps (sub-step 15)

§10 and §11 want every `--json` output to validate against the schema. The two accepted Phase 0
gaps break that: Click usage errors print plain text, and unhandled exceptions exit 1. In
sub-step 15 I'd:

- emit a `CliEnvelope` with exit 2 for usage errors under `--json`;
- map infrastructure errors to exit 4.

Validating against the JSON schema properly needs `jsonschema` as a dev-only dependency. I'll ask
at that sub-step. The fallback is validating with `CliEnvelope.model_validate_json`, which checks
the model but not the committed schema file.

### 3.4 Provisional models: DESIGN.md patch before implementation

CLAUDE.md requires the DESIGN.md patch to be proposed before a provisional model is finalized. In
each of these sub-steps I'll send the patch first and wait for your approval before
implementing:

- 02: ScanInfo
- 08: SmoothingSpec
- 10: RunConfig, RunPlan
- 12: TaskState, SubmissionRecord

I can't edit DESIGN.md, so you apply each approved patch, and I branch after you say "merged".

## 4. Sub-step 01 in detail: `phase-1-01-fixtures`

Branch from `main` after `git pull --ff-only`. Commit message:
`test: add Nanofinder fixtures and make_fixture script`.

### 4.1 `scripts/make_fixture.py`

This is a developer-only script that reads `data/raw/`. It:

- writes `tests/fixtures/nanofinder_mini/NMC811_v0/`, containing the real `NMC811_v0-Info.txt`
  and the 25 spectra for iz=0, iy 0–4, ix 0–4;
- copies the five real Info.txt files to `tests/fixtures/info_txt/`;
- copies two NMC111_v1 spectra of different byte sizes (for example 19920 and 19922 bytes) to
  `tests/fixtures/spectra/`. **This goes beyond §2's fixture layout:** §11 asks for NMC111 reader
  tests, and the planned fixture contains no NMC111 spectra.

Every copy is byte-for-byte, keeping CRLF and any BOM. The script takes `--raw` and `--out`
options with repo defaults. It refuses to overwrite an existing fixture that differs unless
given `--force`, and it fails with a clear error if a source file is missing.

### 4.2 Protecting the fixtures

- **`.gitattributes`:** `tests/fixtures/** -text` and `configs/priors/legacy/*.csv -text`, so
  git never converts CRLF line endings.
- **Pre-commit:** exclude `tests/fixtures/` from `end-of-file-fixer` and `trailing-whitespace`,
  as is already done for the legacy CSVs. Without this, the fixers would strip the `" \r\n"`
  lines from Info.txt.

### 4.3 Tests

These are unit tests only, and none of them reads `data/raw/`.

`tests/unit/test_make_fixture.py` runs the script on a synthetic raw tree in `tmp_path` and checks
that:

- exactly the corner files are selected;
- the copies are byte-identical to the sources;
- the Info files are collected;
- the script refuses to overwrite without `--force`;
- a missing source file gives a clear error.

`tests/unit/test_fixtures.py` checks the committed fixtures:

- 25 spectra plus one Info file, all names matching `FILENAME_RE` and within the corner;
- each spectrum is 1024 CRLF lines of two numbers;
- five Info files are present;
- the NMC111 spectra have different byte sizes;
- `.gitattributes` marks the fixtures `-text`;
- the pre-commit excludes are in place.

`scripts/` is already covered by mypy.

### 4.4 Wrap-up

- Add a DEVLOG entry recording the `n_accum` finding, the Info.txt layout differences and the
  `tests/fixtures/spectra/` deviation.
- Confirm `pixi run test`, `pixi run lint` and `pixi run pre-commit run --all-files` pass.
- Push, open the PR with the five-line summary as its body, print the summary, and stop.

## 5. Patch summary

| # | Section | Change |
|---|---|---|
| P1 | §11 Unit, Info.txt bullet | `n_accum=5` becomes the real per-file values |
| P2 | §3 rule 1 | Describe the real section structure |
| P3 | §3, after the fields table | Exposure time shares a line with cycle time |
| P4 | §3, after the fields table | Accumulation count shares a line with acquisition mode |
| P5 | §16 v0.2 → v0.3 (optional) | Correct "(=5 in samples)" |

The exact text is in section 6. Each "Replace" block is copied from `docs/DESIGN.md` at `ee0980c`.

## 6. Exact DESIGN.md patch text

### P1. §11 `n_accum` (line 428)

Replace:

```
- Info.txt parser on all five files: correct `nx, ny, nz` (50×25×1 and 50×50×1), `Frame Dimentions` wins over `Frame Stage` for Z, `n_accum=5`, `n_series=1`; a modified copy with `n_series=2` is refused.
```

With:

```
- Info.txt parser on all five files: correct `nx, ny, nz` (50×25×1 and 50×50×1), `Frame Dimentions` wins over `Frame Stage` for Z, `n_accum` per file (1, 5, 5, 5, 10 for NMC111_v1, NMC111_v2, NMC811_v0, NMC811_v1, NMC811_v2), `n_series=1`; a modified copy with `n_series=2` is refused.
```

### P2. §3 section structure (rule 1, line 109)

Replace:

```
1. Parse by section (`Frame Header`, `Frame Options`, `Frame Dimentions`, `Frame Stage/Axis : N`, `Scan detection channel information/Channel : N`).
```

With:

```
1. Parse by section (`Frame Header`, `Frame Options`, `Frame Dimentions`, `Frame Stage/Axis : N`, `Scan detection channel information/Channel : N`). Line 1 is the export timestamp and line 2 the title `Scan common information`; neither is a section. The `Frame …` headings end in `:`; `Scan detection channel information` does not, and its preamble (`Device Name : Andor CCD`, `Channels :1`) precedes the `Channel : 0` subsection, so `Channels :1` must not be read as a subsection. `Frame Stage` axes contain flag lines with no separator (`Axis is present`). Blank lines contain a single space. Separators vary (`Date: `, `Scanned points X : `, `Axis scan start, [um]: `, `Head model = `, `Channels :1`).
```

### P3. §3 exposure and cycle time on one line

Insert after the fields table, after the row that begins `| Channel `Temperature [grad C] =``
and before the line that begins `Coordinates:`:

```
`Exposure time, [sec] =` shares its line with `Cycle time [sec] =` (`Exposure time, [sec] = 0.2000 Cycle time [sec] = 0.2862`); the parser reads the number immediately after `Exposure time, [sec] =`, not the remainder of the line. Cycle time is not stored.
```

### P4. §3 accumulation count and acquisition mode on one line

Insert directly after the P3 paragraph:

```
`Number in Accumulation =` follows `Acquisition mode: Accomulate.` on the same line (`Acquisition mode: Accomulate. Number in Accumulation = 5`); the parser finds the key within the line rather than splitting on the first separator. `n_accum` differs per scan (1, 5 or 10 in the samples) and must be read from each file, never assumed.
```

### P5 (optional). §16 consistency (line 508)

P1 makes this entry inaccurate. It is a historical review log, so you may prefer to leave it.

Replace:

```
- **Saturation**: threshold scaled by `n_accum` (=5 in samples) plus flat-top detection (§3).
```

With:

```
- **Saturation**: threshold scaled by `n_accum` (1, 5 or 10 in samples) plus flat-top detection (§3).
```
