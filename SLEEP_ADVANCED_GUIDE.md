# Neuro-buddy V2.5.0. Advanced Sleep Pipeline

## What's new

The Sleep route now adds three scientific research features on top of the existing EEG+IONM analysis, scored hypnogram, stage duration statistics and stage-specific band powers:

1. **Sleep spindle candidates.** Uses YASA's published spindle detector, with default sigma band (12–15 Hz) and 0.5–2.0 s event duration. Reports event counts by selected channel and sleep stage, mean duration/amplitude, density per N2+N3 hour, event onset/peak/end/frequency/amplitude, a counts-per-hour plot and accompanying complete tables and CSV data.
2. **Slow-wave candidates.** Uses YASA's slow-wave detector, defaults of 0.3–1.5 Hz and published amplitude/duration criteria, reporting count, amplitude, duration, N2/N3 stage distribution, temporal density, event locations, graph and full event table.
3. **Optional automated sleep staging.** Uses YASA's pretrained LightGBM sleep-stage classifier to predict Wake, N1, N2, N3 and REM for 30-second epochs, where no verified scores are supplied (default). Predictions are explicitly marked **automated/unreviewed**. Optionally predict in parallel with human-scored stages for comparison, *without replacing* the human labels.

**Non-clinical, research use.** Neither EEG-only fatigue/recovery indices nor the automatic stage labels are medical diagnoses. Detected spindle and slow-wave counts must be reviewed on representative PSG excerpts. The automatic staging method has known limitations, especially N1, abnormal PSG and mismatched montage/references.

## macOS installation

Unzip the project, change into the extracted folder, then:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-sleep.txt
python -m pytest -q tests
```

The base `requirements.txt` still installs the existing EEG pipeline without new optional sleep packages. To enable sleep staging and the YASA microstructure detectors, install `requirements-sleep.txt`. It installs YASA, LightGBM and antropy in addition to the base dependencies. Internet access is needed for the installation.

## Run a sleep dataset from Finder

Keep the PSG `.set`, its exact referenced `.fdt`, sleep stage TSV, and any JSON stage dictionary together under the original BIDS directory hierarchy. The EEG+stage sidecars must be associated with the **same recording and clock**; do not mix PSG with cEEGrid timing blindly.

```bash
python -m neurobuddy_v2 \
  "$(osascript -e 'POSIX path of (choose file with prompt "Select PSG EEG .set")')" \
  --output reports/sleep \
  --chunk-seconds 60 \
  --sleep-staging if_missing
```

If the recording lacks a human-scored stage file, YASA will attempt automatic sleep staging. If verified scores exist, the scored hypnogram is authoritative, but spindle and slow-wave candidates still run within N2/N3. To run an **additional** model comparison without overriding the human stages:

```bash
python -m neurobuddy_v2 /path/to/recording.set --output reports/sleep --sleep-staging always
```

To disable automatic staging (e.g. offline with scored sleep file):

```bash
python -m neurobuddy_v2 /path/to/recording.set --output reports/sleep --sleep-staging off
```

## Processing and memory behavior

- The **original >800 MB `.fdt` stays lazy** under MNE for the shared out-of-core route.
- Microstructure detectors load at most **two EEG channels** in short overlapping windows, resample to 100 Hz, and operate only on N2/N3. Events whose peaks fall inside the nonoverlapping core are retained. Overlap prevents double-counting near chunk boundaries, but boundary sensitivity remains a research limitation.
- Automated staging takes **one best available central EEG channel**, plus optional confidently identified EOG/EMG, and builds a **100 Hz reduced-channel representation** in RAM. This is much smaller than the original full multi-channel PSG, but automated staging is **not strictly constant memory**; it uses time proportional to recording duration and retains the reduced-channel recording in memory. Automatic staging is capped at 18 hours by default. No full-night >800 MB real-data benchmark has been completed yet.
- Provided stages remain authoritative. Unknown numeric stage codes **cannot** be guessed without a documented JSON dictionary; if necessary, the automatic pathway may be used and labeled separately.
- Candidate event counts are **per channel**. A spindle visible on two channels is counted twice (not a claim of two separate sources). N2/N3 density uses stage exposure time, and is biased if many windows were rejected; review QC/warnings.
- Gross amplitude and flatline windows are excluded as provisional QC, not clinical artifact removal. Only two prioritized EEG channels are scanned; results are not a whole-head inventory.

## New HTML / JSON / ZIP outputs

All prior report sections remain. Added inside the existing **Specialized Pathway Results** tables:

- `stage_origin` = `provided_scoring`, `automated_yasa_unreviewed`, or `unscored`.
- `automatic_staging`: algorithm, channel selection, YASA version, model probability summaries, and low-confidence epoch counts.
- `automatic_stage_epochs`: each 30-s epoch with label, optional probability vector, and review flag.
- `automatic_vs_scored_comparison` when `--sleep-staging always` and scored stages exist (agreement fraction and Cohen's kappa, exploratory).
- `spindle_summary`, `spindle_events`, `slow_wave_summary`, `slow_wave_events`, `sleep_event_detection` and any skipped-window QC.

Figures (when corresponding data exist):

- `sleep_hypnogram.png` (scored, or automated explicitly labeled as unreviewed)
- `sleep_automatic_stage_confidence.png` (with per-epoch probability data)
- `sleep_spindle_counts.png` (candidate counts per elapsed hour)
- `sleep_slow_wave_counts.png` (candidate counts per elapsed hour)
- Existing `sleep_stage_bandpowers.png` remains.

**Each graph's exact values** appear in the HTML table beneath it, with downloadable full CSV tables included in `report.zip`. The entire report is machine-readable as `report.json`, which contains the provenance and individual event records. The Brain Activity Player and all original processing steps remain.

## Current test status and limits

- Existing V2 routing / report / streaming tests continue to run.
- New tests cover automatic stage data models, simulated YASA detections, core-window deduplication, scored-stage priority, JSON and HTML rendering, ZIP assets, and dependency-absent graceful degradation.
- CI uses a deterministic **YASA test double** to exercise integration because the YASA package cannot be installed in this execution environment. This is **not** empirical validation of YASA's scientific accuracy or an actual pretrained model run. The full real ds005207 overnight PSG `.fdt` is not available here; an end-to-end test on the user's Mac is still mandatory.

## Research references

- Vallat, R. & Walker, M.P. (2021). *An open-source, high-performance tool for automated sleep staging.* eLife 10, e70092. https://doi.org/10.7554/eLife.70092
- YASA SleepStaging API: https://yasa-sleep.org/generated/yasa.SleepStaging.html
- YASA spindle detector: https://yasa-sleep.org/generated/yasa.spindles_detect.html
- YASA slow-wave detector: https://yasa-sleep.org/generated/yasa.sw_detect.html
- Lacourse, K. et al. (2018). *A sleep spindle detection algorithm that emulates human expert spindle scoring.* J Neurosci Methods. See YASA spindle API references.
- Massimini, M. et al. (2004). *The sleep slow oscillation as a traveling wave.* J Neurosci 24, 6862–6870. See YASA slow-wave API references.
