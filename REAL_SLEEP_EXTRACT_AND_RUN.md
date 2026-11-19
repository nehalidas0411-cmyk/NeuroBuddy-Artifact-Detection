# Neuro-buddy v0.5.1. Real sleep EEG excerpt and analysis

This utility takes a **real** local `.set` + `.fdt` recording and generates a **self-contained EEGLAB `.set`** holding 10, 15 or 20 minutes of the *original EEG samples* (no companion FDT required for the excerpt). It also crops accompanying event and sleep-scoring TSV files to the same window, preserving original stage codes, and runs the Sleep pipeline on the excerpt.

It was developed against the **real ds005207 subject 003 `.set` header** and its 15-channel, 256 Hz structure. A temporary **synthetic FDT** was used only to test the extraction and report-writing mechanics, since the actual 600+ MB FDT has not been uploaded. **No real ds005207 sleep EEG has been analyzed in this environment.**

## Run on your Mac

1. Extract `Neurobuddy_v0.5.1_Real_Sleep_Test_Kit.zip` and open Terminal in its extracted folder.
2. The *original* `sub-003_ses-001_task-sleep_acq-PSG_eeg.set` and its large `.fdt` must be in the **same local folder** on your Mac. The `.set` can also be named with `(1)` if its original referenced `.fdt` is present.
3. If available, place `sub-003_ses-001_task-sleep_acq-PSGScoring_events.tsv` in that folder. The official `task-sleep_acq-psgScoring_events.json` dictionary is bundled in this kit and is used automatically for this specific ds005207 scoring format.
4. From the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-sleep.txt
python extract_and_run_sleep.py
```

A macOS Finder picker opens to select the original `.set` file. The tool chooses a 20-minute window rich in documented N2/N3 stages if the scoring TSV exists. Otherwise it chooses a segment away from the beginning of the recording. It does **not assume** that the fallback segment contains sleep.

### Exact segment control

```bash
python extract_and_run_sleep.py --minutes 10 --start-minutes 60
```

Or with explicitly supplied scorer:

```bash
python extract_and_run_sleep.py --set "/path/to/sub-003_ses-001_task-sleep_acq-PSG_eeg.set" \
  --scoring "/path/to/sub-003_ses-001_task-sleep_acq-PSGScoring_events.tsv" \
  --minutes 20 --sleep-staging always
```

To only create a shareable excerpt for uploading back to ChatGPT, without running the pipeline:

```bash
python extract_and_run_sleep.py --no-run --minutes 10
```

## Files produced

```text
real_sleep_excerpt_and_report/
    sample/
        sub-003_ses-001_task-sleep_acq-PSG_eeg.set     # SELF-CONTAINED, no FDT
        sub-003_ses-001_task-sleep_acq-PSG_events.tsv   # if supplied
        sub-003_ses-001_task-sleep_acq-PSGScoring_events.tsv # if supplied
        task-sleep_acq-psgScoring_events.json            # if found or bundled
    segment_manifest.json  # original-relative offset and provenance
    report/
        report.html
        report.json
        report.zip
        *_data.csv
        *.png
    run_result.json
```

The segment starts at time 0 in the excerpt, with its original start offset preserved in `segment_manifest.json`. Event onsets and durations are clipped and shifted to match the excerpt. The data is re-encoded using standard EEGLAB microvolt units, retaining channel positions.

## Verification / caveats

- The exported `.set` is re-opened and signal samples checked for readability. The developer also compared a synthetic test fixture against its cropped export and found identical readback values (maximum absolute difference 0 for the tested samples).
- The sleep report was executed with the synthetic fixture and returned `PARTIAL`, as expected without scored stages/YASA. This is an **engineering smoke test**, **not** a real physiological test.
- The actual ds005207 `.fdt` and optional supplied scoring TSV are necessary to produce and verify the *real* analysis. Download them directly to your Mac. The large FDT never needs to be uploaded.
- Automated staging, spindle and slow-wave features need the YASA dependency. They are estimates, not clinical diagnoses, and should be checked against expert PSG scoring.
- **Do not use a 10–20-minute excerpt to estimate full-night total sleep time, sleep efficiency or REM latency.** Those require the full scored night.
- Inspect the exported EEG and score alignment, and never interpret the results as medical advice.

Reference scoring dictionary: https://raw.githubusercontent.com/OpenNeuroDatasets/ds005207/main/task-sleep_acq-psgScoring_events.json
Dataset: https://openneuro.org/datasets/ds005207
