# Neuro-buddy V2 v0.4.0: What is generated for each EEG recording?

## Shared analysis for ALL four paths

As soon as valid EEG samples load, the report includes:
- Recording metadata: channel names, sample rate, channel count and duration.
- Decision intelligence evidence: selected GENERAL, ODDBALL, MOTOR, or SLEEP route, subtype, annotation and readiness flags.
- Shared EEG statistics: mean/median, standard deviation, variance, RMS, peak-to-peak, min/max, MAD, skew/kurtosis, zero crossings, crest factor, delta/theta/alpha/beta absolute+relative bandpower, band ratios, dominant frequency, peak-alpha frequency, spectral entropy and 95% spectral edge.
- Heuristic IONM signal quality and potential artifacts: flatline, unusual amplitudes, RMS differences, clipping proxy, line noise (when observable), transient screening, and legacy event-response comparisons where applicable and permitted by memory mode. This is screening, not artifact removal or a validated diagnosis.
- Static plots, each followed by a numerical data table and downloadable CSV: waveform raw-versus-filtered, power spectrum, relative bandpower by channel, RMS by channel, quality grading. Extra plots are generated if their source evidence is available.
- Interactive 2D scalp EEG player, seek bar, frequency-band modes, event markers and channel fallback when electrode coordinates are unavailable.

## Extra pipeline outputs

### GENERAL EEG
- Exploratory Hjorth activity, mobility, complexity by EEG channel.
- No additional dedicated general-only graph in this version. The shared waveform, PSD, bandpower and QC graphs remain.
- Does **not** claim stress, fatigue, overtraining, or athlete readiness are clinically measurable from EEG alone.

### ODDBALL
- Target, standard, distractor stimulus counts from verified event labels.
- Event-related potential (ERP) traces at a selected Pz/Cz/evaluable EEG channel, target/standard mean traces, target-minus-standard difference.
- P300 difference amplitude (mean and peak in 250–500 ms), peak latency, exploratory N200 difference mean (150–300 ms).
- Exploratory independent-trial Welch t-test, Cohen's d and uncorrected p, if sufficient trials.
- Behavioral stimulus-response matching: hits, misses, false alarms, correct rejections, log-linear d-prime, mean and SD of reaction times if valid response timestamps exist.
- Graphs: condition ERP overlay and reaction-time histogram when enough usable events exist. Each graph has summary and full numerical CSV tables.
- Not all raw response-time columns are trusted; incompatible units are flagged.

### MOTOR IMAGERY / MOTOR EXECUTION
- Motor task subtype from explicit context/documented run semantics.
- Left/right/feet/hands cue counts.
- Per-trial and per-condition mu (8–12 Hz) and beta (13–30 Hz) ERD/ERS over C3 and C4, when those channels and properly aligned cues are present.
- Graph: mu-band ERD/ERS by task condition and channel; an accompanying table contains beta measurements and trial-level values.
- This version does **not** generate an entire time-frequency ERSP heatmap or infer actual exercise fatigue from motor imagery alone.

### SLEEP
- Verified sleep-stage intervals W, N1, N2, N3 and REM, constructed from externally scored annotations.
- Sleep stage durations, total sleep time, transitions, scored interval count; sleep efficiency explicitly unavailable when time in bed is unverified.
- Graph: scored hypnogram, plus grouped stage-specific EEG delta/theta/sigma/beta PSD bandpowers on a selected channel. Complete per-stage interval and spectral data tables.
- If valid stages are absent, the pathway may still be identified as SLEEP but the output becomes PARTIAL and cannot show a scored hypnogram.
- This version does **not** automatically score sleep stages, detect spindles/slow waves individually, or estimate validated recovery metrics.

## Generated report files

```
reports/<recording>/
    report.html                 interactive standalone report with embedded player and graph tables
    report.json                 structured machine-readable results, figures+CSV references
    report.schema.json          JSON validation schema
    player_data.json            derived sampled EEG playback data, not the raw continuous recording
    waveform_raw_vs_filtered.png
    waveform_raw_vs_filtered_data.csv
    power_spectrum.png
    power_spectrum_data.csv
    relative_band_power.png
    relative_band_power_data.csv
    channel_rms.png
    channel_rms_data.csv
    channel_quality.png
    channel_quality_data.csv
    [specialized figures and their *_data.csv, if applicable]
    manifest.json               SHA-256 hashes for portable report assets
    report.zip                  transfer package containing report and derived figures/tables
```

The `.set`/`.fdt` raw signal files are deliberately **not included** in `report.zip`.

## Large EEG mode

When the disk signal or estimated float64 EEG signal exceeds 256 MiB, automatic out-of-core processing uses bounded lazy windows, cumulative Welch bandpower and incremental channel summaries. This avoids full signal preload and supports large `.fdt` input without needing RAM proportional to file size. Large-mode PSD, median/MAD and filtering differ from the small-mode algorithms and are labeled in the report. Legacy global IONM event statistics and the cross-channel outlier audit are skipped in the streaming path and flagged. A true >800 MiB sparse external `.fdt` was tested for lazy open and edge-window reads; a complete real overnight ds005207 run remains to be tested.

## Advanced sleep results in v0.5.0

When `requirements-sleep.txt` is installed, the Sleep pathway additionally exports YASA-based spindle and slow-wave candidate event lists, count/density summaries, counts-per-hour graphs, and tables. If no verified sleep-stage scores are present, the `--sleep-staging if_missing` default attempts automated 30-second sleep staging and produces an explicitly **unreviewed** hypnogram. With `--sleep-staging always`, scored stages remain authoritative and model predictions plus agreement statistics are shown separately.

New assets: `sleep_spindle_counts.png`, `sleep_slow_wave_counts.png`, `sleep_automatic_stage_confidence.png` where applicable, each with associated `*_data.csv` and individual event CSV tables. Full structured results are in `report.json`. See `SLEEP_ADVANCED_GUIDE.md` for scientific limitations and memory details.
