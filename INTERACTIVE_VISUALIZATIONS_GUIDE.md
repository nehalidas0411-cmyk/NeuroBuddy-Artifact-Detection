# Neuro-buddy v0.6.1. Interactive Comparison Heatmap and Sleep Replay

This is an **additive update** to the existing v0.6.0 two-path architecture. All original single-recording statistics, four task routes, longitudinal mean/SD/CV, per-metric z-scores, threshold logic, HTML/JSON/CSV exports, and portable report ZIPs remain in place.

## Baseline vs. Test Brain Heatmap

In comparative mode, open `athlete_report.html` and expand **COMPARATIVE ANALYSIS**, or open `comparative_analysis.html` directly. The interactive heatmap uses **only** the existing `comparative_analysis.json` metric rows. It does not recalculate z-scores.

- Rows: electrode/channel labels (including any channel-level quality statistics).
- Columns: individual channel statistics. Use the metric-group dropdown to narrow the view.
- Blue: negative z-score; red: positive z-score; gray: missing or not comparable.
- Gold outline: **Abnormal**, using the original strict `abs(z) > 3` rule. Exactly ±3 is still **Within range**.
- Click any cell to see baseline mean, baseline sample SD, test value, z-score, units, status and explanation.
- Non-channel scalar metrics continue to appear in the unchanged full comparison table below the heatmap. No fictitious electrode location is inferred from channel names; this is a **channel-by-metric matrix**, not a spatially interpolated anatomical scalp map.

## Interactive Sleep Replay

In single EEG mode, a **SLEEP** recording's `report.html` now contains an interactive sleep hypnogram replay next to the existing Brain Activity Player. It includes:

- Play/pause, seek, 1×/10×/60×/300× playback, clickable timeline and jump-to-stage.
- Stage labels W/N1/N2/N3/REM, stage intervals and unscored gaps.
- Source label: provided scoring vs automated YASA/unreviewed.
- Candidate spindle and slow-wave counts within ±15 seconds of the cursor when those detectors provide onset timestamps.
- Seeking in the sleep replay also moves the existing Brain Activity Player to the corresponding time in the **single-recording report**. Starting playback from the original 2D player does not control the sleep replay in the reverse direction.

In comparative mode, **BASELINE** includes a separate sleep replay for each baseline recording, and **TEST EEG** includes the test recording's replay. Their independent timelines are not combined or treated as identical recordings. Each original full EEG report also contains its own replay.

Sleep stage intervals are preserved in the `specialized_metrics.sleep.stage_intervals` field of each single-recording `report.json` (derived from verified stage annotations or separately labeled automated YASA estimates). If no usable stages are available, the UI explicitly reports that replay is unavailable; it does not invent sleep stages.

## Scientific and data boundaries

- Playback is of stage labels and derived candidate event timestamps, **not a full raw EEG signal replay**. The original Brain Activity Player continues to display its own downsampled EEG sensor data.
- Automated YASA sleep stages remain **unreviewed** and never replace verified expert scores.
- The comparative heatmap is a visualization of the existing statistical comparisons, **not** a diagnosis of fatigue, recovery, or overtraining.
- Reports remain self-contained HTML with inline JavaScript and work without a CDN or internet connection. Original EEG data are not included in the ZIP packages.
- Original comparative threshold remains strictly `|z| > 3`, not `>= 3`.

## Tests

Run `python -m pytest -q tests`. Added tests verify sleep-stage serialization, absence of fabricated stage data, comparative z-score reuse, missing-metric behavior, and JavaScript syntax. The synthetic comparison demo is in `tests/demo_comparison_run.py`. Real overnight PSG replay and cross-browser performance still need validation on real sleep datasets.
