## 0.5.1 (ds005207 real-excerpt testing kit)

- Keep raw TSV/CSV columns as strings so numeric sleep-stage codes remain exactly as encoded (for example, `4`, not `4.0`). This fixes lookup against documented BIDS JSON `staging.Levels`.
- Add local one-command real .set/.fdt 10–20 minute extractor, preservation of cropped scoring/events, and automated Neuro-buddy report generation.
- Add regression test for ds005207 numeric sleep stages.
- The genuine ds005207 PSG FDT remains to be tested by an end-to-end local run.

## 0.5.0. Sleep microstructure and model-based staging (research implementation)

- Introduced optional YASA sleep staging with 30-second W/N1/N2/N3/REM labels and probability reporting when YASA exposes scores. Default stage source is supplied scorer; automatic staging runs only if scored stages are missing. `--sleep-staging always` runs a separate comparison, never overwriting scored stages.
- Added per-channel spindle and slow-wave candidate detectors constrained to N2/N3 with short overlap windows for large lazy PSG inputs.
- Added spindle/slow-wave event counts, mean durations/amplitudes, density, event lists, skipped QC windows, plots, HTML tables and CSV/ZIP assets.
- Added `requirements-sleep.txt`, optional `sleep` project extra, and automated tests using a deterministic YASA test double (real pretrained YASA/PSG validation pending).

# Neuro-buddy V2 change log

## v0.3.1: Readable specialized pathway tables

- The Specialized pathway results section now displays grouped statistical tables instead of raw JSON.
- Oddball event counts, ERP and behavioral test values are shown as numerical tables with units.
- Motor task metadata, cue counts, per-condition measures and all trial-level results are in readable tables.
- Sleep and General EEG pathway statistics use the same generic hierarchical table renderer.
- Long trial-level tables have paging controls and a transferable CSV asset.
- The report.json data contract and EEG/IONM analysis calculations are unchanged.
- Added renderer tests for Oddball, Motor and Sleep results.

## v0.3.0: Graph-aligned statistics tables

- No new report tab. Added scientific summaries and numerical tables directly beneath every static graph.
- Charts now export full plotted data and test statistics as individual CSV files included in each report ZIP.
- All graph rows are also embedded in the offline HTML and rendered in pages of 75 rows.
- Each chart in `report.json` has a `visualizations.figures[*].data_table` manifest with complete filenames, row counts, columns and supplementary tables.
- Added complete legacy raw EEG metrics and complete quality fields to existing overview tables.
- Added regression coverage confirming complete CSVs, under-graph table presence, HTML transfer and ZIP integrity.
- Original EEG/IONM numerical calculation functions remain unchanged.

## v0.4.0: Large EEG / overnight PSG bounded-memory processing

- Adds `large_recordings.py`, an automatic streaming branch for recordings whose source `.fdt`/EEG signal or estimated float64 in-memory size is at least 256 MiB.
- Opens external EEGLAB `.set`/`.fdt` lazily and reads EEG in bounded chunks (target 60 s, capped at about 32 MiB of float64 raw EEG samples per chunk); no full `raw.load_data()` or full-length filtered copy on this path.
- Incremental raw/filtered channel moments, Welch PSDs, canonical band powers, artifact screening, and a second bounded QC pass. Median and MAD are explicitly approximate. Streaming uses context-padded Butterworth SOS filtering rather than the previous entire-record MNE FIR filter.
- Existing static figures and graph-associated CSV/table exports retained in streaming mode. Specialized Sleep stage PSD/hypnogram and Motor ERD/ERS can pull event windows lazily. Bounded Oddball event epoch extraction is included for larger event-driven datasets.
- `--large-threshold-mb` and `--chunk-seconds` configuration options (a threshold of zero forces streaming for regression testing), progress logs, and processing-method provenance.
- On missing required `.fdt`, classification identity derived from filename/task metadata is still reported when possible, but analysis status remains INCOMPLETE.
- Tested via true >800 MiB **sparse simulated external EEGLAB `.fdt`** for header loading and beginning/end window reads, plus successful end-to-end forced-streaming General/Sleep/Oddball tests on smaller data. **Not a claim of complete production validation on an 800 MiB real overnight ds005207 PSG.** A full-duration synthetic stress run exceeded the available execution window; that remains a performance test.
- Resource safety note: maximum 32 MiB of first-stage float64 chunk data does not equal a 32 MiB total process budget; processing arrays, plots, and Python/MNE overhead also consume memory.

## 0.6.0 | Athlete and Team comparative layer
- Preserved the existing four-task single EEG analysis routines and interactive reporting.
- Added `neurobuddy_v2.comparative` for ≥3 distinct baseline recordings and one testing recording per athlete.
- Reports: baseline individuals, longitudinal means/sample SD/CV, test individual, z-score comparison with `Abnormal` / `Within range` / `Not comparable` status, exact three collapsible HTML sections.
- Added frontend JSON input-manifest and batch Team manifest interfaces. Team baseline separation is enforced.
- Full report package with HTML, JSON, CSV and retained individual graphical reports, excluding original raw EEG.
- Blocks task/subtype/protocol mismatches, duplicate baseline signal hashes, mixed preprocessing, partial scans.
- Adds focused comparator regression tests and synthetic end-to-end fixture. Existing EEG statistics remain unchanged.

## v0.6.1. Additive interactive visualizations

- Added channel × metric Baseline vs. Test interactive heatmap to both athlete and standalone comparative HTML reports.
- Added Interactive Sleep Replay to SLEEP single EEG reports and each baseline/test sleep section in athlete comparative reports.
- Exported stage intervals in single-recording report JSON for replay.
- Kept existing statistical comparison tables and strict ±3 SD rule unchanged.
- Added tests for HTML/JS payload generation and syntax.
