> **v0.6.1 update:** See [INTERACTIVE_VISUALIZATIONS_GUIDE.md](INTERACTIVE_VISUALIZATIONS_GUIDE.md) for the new Baseline vs. Test Heatmap and Interactive Sleep Replay.

# Neuro-buddy V2 v0.6.0: Single EEG + Athlete/Team Comparative Analysis

**New:** Baseline (3+ EEGs) → longitudinal baseline mean/sample SD → testing EEG → per-statistic z-score table, all using the *unchanged* four-path EEG pipeline. See **[COMPARATIVE_ANALYSIS_GUIDE.md](COMPARATIVE_ANALYSIS_GUIDE.md)** for commands, frontend manifest schemas and outputs. Original single EEG operation and large sleep `.fdt` support remain available.

---

# Neuro-buddy V2 v0.5.0. Sleep Microstructure and Automatic Staging

The Sleep pathway now supports YASA-based sleep spindles, slow-wave event detection, and optional automatic 30-second staging, with full report tables and CSV/JSON packaging. See **[SLEEP_ADVANCED_GUIDE.md](SLEEP_ADVANCED_GUIDE.md)** for installation, commands, algorithm limits and references.

**Install advanced sleep features:** `python -m pip install -r requirements-sleep.txt`

**Run:** `python -m neurobuddy_v2 /path/to/PSG_eeg.set --output reports/sleep --sleep-staging if_missing`

---

# Neuro-buddy V2 | Runnable research implementation

This repository is the first working vertical slice of the Neuro-buddy V2 PRD. It is **not clinically validated**. It incorporates the previous combined EEG + IONM pipeline by importing its analysis functions unchanged from `neurobuddy_v2/legacy_combined.py`.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Analyze one real OpenNeuro recording

```bash
python -m neurobuddy_v2 /path/to/sub-001_task-P300_run-1_eeg.set --output reports/oddball
python -m neurobuddy_v2 /path/to/sub-001_task-motion_run-4_eeg.set --output reports/motor
# Or scan a dataset folder for multiple runs:
python -m neurobuddy_v2 /path/to/bids/sub-001/eeg --output reports/batch
```

All matching TSV/CSV/JSON sidecars in the same folder are discovered automatically. An EEGLAB `.fdt` is optional only if the `.set` contains its own samples. If the `.set` requires an external `.fdt`, it must be located beside the `.set`.

## What works in this initial implementation

- BIDS-inspired recording grouping by subject, session, task and run; matching sidecars, `.fdt`, and reporting SHA256 manifests.
- Detect identical duplicate TSV filenames like `events(1).tsv`, and flag conflicting variants.
- Normalize BIDS TSV and EEGLAB embedded annotations, preserving original labels and timestamps. Never treat arbitrary numeric codes as meaningful motor or sleep events.
- Explainable routing to GENERAL, ODDBALL, MOTOR, SLEEP. NEEDS_REVIEW/FAILED state is supported without falsely running a specialized analysis.
- Reuse original combined pipeline descriptive EEG statistics, PSD integrals, quality, artifacts and event screening. **Only the applicable specialized route runs after the shared analysis.**
- Oddball ERP, target-standard difference, response-event-derived reaction times, charts.
- Motor cue-specific mu/beta ERD/ERS at C3/C4 with known run mapping and a chart.
- Sleep hypnogram, stage durations, and representative stage-specific PSD bandpowers when externally supplied stages are aligned. Supports OpenNeuro ds005207 numeric `staging` codes **only when its BIDS JSON Levels dictionary is supplied**; zero-duration stage markers are converted to intervals from verified event onsets. No automatic sleep-stage guessing.
- Self-contained `report.html` with a **2D scalp activity player** (or channel fallback), seek bar, frequency modes and event jumps.  `report.json`, `player_data.json`, PNG graphs, `manifest.json` with hashes, and portable `report.zip`.

## New in V2 v0.3.1: Readable specialized pathway results

The Specialized pathway results section shows regular statistics tables, not raw JSON. Event counts, ERP results, behavioral accuracy and reaction times, motor trial measurements, and sleep architecture values are grouped and labeled. Long trial tables can be paged and exported as CSV. No additional tabs. The underlying report.json remains unchanged.

## New in V2 v0.3.0: Full numerical tables beneath every graph

Every generated static plot in `report.html` now has the numerical statistics
and plotted measurements immediately **under that graph**, not in a separate tab.

- A summary of applicable measures and statistical test results appears first.
- The complete underlying data table is embedded directly into the offline HTML.
- Use **Next / Previous** to browse large tables without freezing the browser.
- Select **Download every value as CSV** to get the full original-precision table.
- Plot CSV files are included in `report.zip`, and `report.json` lists each
  figure's `data_table` file, column names, row count and supplementary tables.
- For long spectra or waveforms the HTML initially shows 75 rows, but **all**
  rows remain accessible using pagination. The CSV has every numerical row.
- The shared spectral and IONM quality overview sections also expose all
  available fields, not merely a subset.

The original combined EEG/IONM scientific calculation functions are unchanged.

## Important current limitations

1. Small/medium inputs use the original in-memory combined analysis. Large inputs use the v0.4.0 streaming branch with documented scientific and performance limitations.
2. `ds005207` sleep was **not included in the uploaded test files**, so real PSG + PSGScoring timing/mapping remains unvalidated. The sleep route is implemented but needs dataset validation.
3. The motor task adapter uses the published PhysioNet EEGMMI run assignment. `TASK2T1` only maps to left-hand imagery within documented task contexts.
4. The oddball dataset contains conflicting units metadata for its `response_time` column. The original values remain in provenance. Mean RT is calculated only from independent response event onsets.
5. The scalp-map interpolation is an **illustrative sensor-level interpolation**, not 3D source imaging. Expert validation and spatial quality gates are required before it is used for scientific inference.
6. No EEG-only athlete fatigue/recovery or stress diagnosis is provided. Relevant descriptive EEG metrics are retained.
7. Initial file discovery supports EEGLAB and MNE common recording formats, plus TSV/CSV/JSON and safe ZIP input. It supports BIDS-style entity matching and inherited event dictionaries, but does not yet implement the complete MNE-BIDS metadata inheritance model or HED parsing.
8. `report.html` contains its own scripts and derived player data for local offline playback. Downstream applications embedding HTML must sandbox it and verify asset hashes. `report.json` is the authoritative machine-readable interface.

## Output files

`report.json` | `report.schema.json` | `report.html` | `player_data.json` | `manifest.json` | `report.zip` | PNG static graphs.

The HTML embeds playback data and figures so it does not need the original EEG or a CDN. The ZIP contains the same HTML, the JSON schema, normalized events, report JSON, playback data and hashed images. For shorter recordings the default playback budget is up to **1500 derived frames**; use `--player-frames` to change it. Large overnight recordings are sampled more sparsely, not played back sample-for-sample.

For directly uploaded ds005207 PSG files, include the corresponding `PSGScoring_events.tsv` **and** its documented `task-sleep_acq-psgScoring_events.json` dictionary. The actual OpenNeuro TSV has columns `onset`, `duration`, `staging`; `duration` is zero and numeric `staging` codes are described by the JSON. Never assume code 1 means N1 or code 2 means N2.

Safe ZIP intake also works: `python -m neurobuddy_v2 /path/to/bids.zip --output reports/`.

## Regression tests

```bash
pytest -q tests
```

Real-world reference sources:
- https://openneuro.org/datasets/ds003061
- https://openneuro.org/datasets/ds004362
- https://openneuro.org/datasets/ds005207
- https://physionet.org/content/eegmmidb/1.0.0/
- https://github.com/OpenNeuroDatasets/ds005207/blob/main/task-sleep_acq-psgScoring_events.json
- https://mne.tools/stable/generated/mne.io.read_raw_eeglab.html

This code is an implementation starter that **must be validated with external reference outputs** before research or performance-intelligence deployment.

## New in v0.4.0: Files larger than 800 MB / overnight sleep EEG

Large files are **not** analyzed by loading the entire .fdt into memory. For source files or estimated float64 EEG arrays >=256 MiB, the pipeline switches to a chunked, lazy reading mode. This is especially important for ds005207 full-night PSG recordings. The `.set` and its **actual matching `.fdt`** must still be in the same folder. Do not rename `.fdt` to a different name than the `.set` expects.

```bash
# macOS: choose any EEG .set file in Finder
python -m neurobuddy_v2 \
  "$(osascript -e 'POSIX path of (choose file with prompt "Select your EEG .set file")')" \
  --output reports/sleep --chunk-seconds 60 --player-frames 1200

# Reduce memory / choose shorter chunks for very high-density EEG:
python -m neurobuddy_v2 /path/to/sleep_eeg.set --output reports/sleep \
  --chunk-seconds 20 --large-threshold-mb 128

# Force streaming on a smaller recording to verify low-memory behavior:
python -m neurobuddy_v2 /path/to/sample_eeg.set --output reports/test \
  --large-threshold-mb 0
```

**In large mode**: channel statistics, band powers and quality indicators are derived chunk-wise, not by materializing the whole EEG recording. The raw power spectrum uses a segment-weighted Welch estimate with approximately 0.25 Hz resolution at 256 Hz (4 s segments). Approximate median/MAD use a deterministic subsample. The 1–45 Hz filter uses 8-second-context, zero-phase SOS processing per chunk, so numerical output differs from the original small-file full-record FIR branch. The report documents these differences. The legacy IONM event-locked comparisons and across-channel outlier tests are **not** run for large recordings; the report explicitly flags those omissions. Existing sleep scoring and stage-wise EEG spectra run on short read windows. The original `.fdt` is **not** included in the portable ZIP; the JSON, HTML, graphs, tables, and compact derived player payload are included.

**Verification status**: automated tests include the real EEGLAB external-file reader on an 811.8 MiB sparse `.fdt`, plus successful complete report generation in forced streaming mode on small Sleep/Oddball examples. A complete end-to-end run on a real >800 MB ds005207 recording is **not yet verified**, and a complete synthetic >800 MiB stress run exceeded the execution window in this environment. Please do not treat the large-file code as clinically validated or performance-certified until that full test completes on the target machine.

### Output contents by path

All complete runs generate: `report.html` (self-contained interactive viewer, player and figure tables), `report.json` (machine-readable data), `report.schema.json`, `player_data.json` (derived EEG playback), PNG figures and their CSV source tables, `manifest.json` (asset checksums), and `report.zip` (portable report without original raw EEG).

| Path | Additional outputs (when prerequisites are present) | Not currently included |
|---|---|---|
| GENERAL | General EEG channel stats, quality, PSD/bandpowers, Hjorth parameters, raw-vs-filtered waveform, PSD, relative bands, channel RMS/quality | Validated stress, overtraining, or recovery diagnosis |
| ODDBALL | Condition event counts, ERP target/standard/difference, P300/N200 window measures, Welch trial comparison, stimulus-response RT and accuracy estimates, RT histogram | Clinically validated P300 diagnosis; automatic response timestamp reconciliation for arbitrary protocols |
| MOTOR | Cue counts, C3/C4 mu and beta ERD/ERS per condition and per trial, condition bar plot | Real physical-fatigue/recovery measurement without external performance sensors; full time-frequency ERSP map |
| SLEEP | Scored hypnogram, stage durations, total sleep time, transitions, stage-specific raw-EEG Welch bandpowers and bar plot | Spindle/slow-wave event counts; automatic sleep staging; sleep efficiency unless time-in-bed is verified |

For Sleep: `*_PSG_events.tsv` containing Lights On/Off is *not* interchangeable with a scored `*_PSGScoring_events.tsv` and its JSON stage-code dictionary. With only lights events, the sleep route can identify the task but will not fabricate an N1/N2/N3/REM hypnogram.
