# Neuro-buddy v0.6.0. Athlete and Team Comparative EEG Analysis

## What changed

This update is **additive**. All four original single-recording routes and their scientific functions are preserved:

- `GENERAL` / unspecified EEG
- `ODDBALL` (ERP and behavioral results)
- `MOTOR` (imagery or execution, subtype-matched)
- `SLEEP` (scored or unscored, including optional YASA and large-file streaming)

The new `neurobuddy_v2.comparative` module simply runs the **same single-recording pipeline** once per baseline recording and once for the test recording, then aggregates results. It does not invent replacement statistical tests.

### A. One EEG (original workflow)

```bash
python -m neurobuddy_v2 "$(osascript -e 'POSIX path of (choose file with prompt "Choose EEG .set file")')" --output reports/single
```

This creates the original complete HTML (with Brain Activity Player), JSON, figures, CSV graph tables, and `report.zip`.

### B. Athlete mode. Three baseline EEGs and one Test EEG

You need three **distinct** EEG recordings from the **same athlete** and compatible task/protocol, plus one test EEG. Companions (`.fdt`, `_events.tsv`, staging files, and BIDS JSON) must be accessible alongside their `.set` recordings. All four files must be usable by the same four-route decision engine.

```bash
python -m neurobuddy_v2 \
  --baseline "/absolute/path/baseline_1.set" "/absolute/path/baseline_2.set" "/absolute/path/baseline_3.set" \
  --test "/absolute/path/testing.set" \
  --team "team-01" --athlete "athlete-01" \
  --output reports/athlete-01
```

For long sleep recordings with mixed file sizes or PSD methods, use `--large-threshold-mb 0` to force all four scans through the **same** bounded-memory estimator. This may skip some original IONM event-comparison heuristics, clearly identified in each individual report. It does not create new EEG metrics.

The frontend can send an input manifest instead of executing CLI arguments:

```json
{
  "team_id": "team-01",
  "athlete_id": "athlete-01",
  "baseline_inputs": ["data/baseline_1.set", "data/baseline_2.set", "data/baseline_3.set"],
  "test_input": "data/testing.set"
}
```

Save this as `athlete.json`. Relative paths are interpreted relative to the manifest location:

```bash
python -m neurobuddy_v2 --compare-manifest athlete.json --output reports/athlete-01
```

Or call from your backend directly:

```python
from neurobuddy_v2.comparative import run_athlete_workflow, build_comparison

package = run_athlete_workflow(
    baseline_inputs, test_input, output_dir,
    team_id="team-01", athlete_id="athlete-01",
    large_threshold_mb=0  # optional when all records need one streaming method
)
# `build_comparison(existing_baseline_report_dicts, test_report_dict)`
# also allows the frontend to reuse reports that were already processed.
```

### C. Team mode. Multiple athletes, each compared to their own baseline

```json
{
  "team_id": "team-01",
  "athletes": [
    {"athlete_id":"athlete-01", "baseline_inputs":["a/one.set","a/two.set","a/three.set"], "test_input":"a/test.set"},
    {"athlete_id":"athlete-02", "baseline_inputs":["b/one.set","b/two.set","b/three.set"], "test_input":"b/test.set"}
  ]
}
```

```bash
python -m neurobuddy_v2 --team-manifest team.json --output reports/team-01
```

A Team report contains an index of athlete reports and **never combines their baselines**. Each athlete report has the same three sections and artifacts. Failed comparisons are reported individually.

## Exact report structure

```
athlete_report.html               Three collapsible sections:
                                  1. BASELINE
                                     - Each full individual baseline report
                                     - Longitudinal mean, sample SD, CV
                                  2. TEST EEG
                                     - Full original test report
                                  3. COMPARATIVE ANALYSIS
                                     - Table ONLY; no comparative graphs

athlete_report.json               Frontend contract with stable relative links
baseline/longitudinal.json        One mean + sample SD per compatible statistic
baseline/longitudinal.csv
baseline/longitudinal.html
baseline/01_*/report.html         Full single baseline report with charts and player
baseline/01_*/report.json         Its own unchanged statistics
baseline/01_*/report.zip
baseline/02_*/...
baseline/03_*/...
test/report.html                  Full single test report, graphs and player
test/report.json
test/report.zip
comparative_analysis.json         Fully structured per-metric comparisons
comparative_analysis.csv
comparative_analysis.html         Table only
manifest.json                     Archive entrypoints, file sizes and hashes
athlete_report.zip               Transferable combined package (no raw EEG)
```

## How the longitudinal statistics work

Every baseline report is produced by the same EEG analysis method. For each **matched scalar EEG statistic**, the longitudinal aggregator calculates:

- `baseline_mean` = arithmetic mean of per-recording results
- `baseline_sd` = **sample** standard deviation (`ddof=1`) between recordings
- `cv_percent` = `100 × baseline_sd / |baseline_mean|` when the mean is nonzero
- `z_score` = `(test_value − baseline_mean) / baseline_sd`
- **Abnormal** when **`abs(z_score) > 3`**, including below −3
- **Within range** when `abs(z_score) <= 3` (exact ±3 included)
- **Not comparable** when SD is zero, a measurement is missing, or there is a compatibility problem

The rightmost status near the z-score is highlighted red only for `Abnormal`. Every comparison row shows baseline mean, SD, test value, z-score, status, units, and provenance. Actual rows may contain additional fields, including lower/upper bounds and CV.

### What is comparable?

Only comparable scalar values from the **unchanged shared and specialized reports**: channel features, artifact/quality numeric measures, available IONM trial statistics, condition-based ERP/motor/sleep scalar metrics, etc. There is no second EEG analysis engine. Trial sample arrays, significance `p` values, ordinal labels and sample counts remain in their own individual reports but do not receive scientifically meaningless baseline z-scores.

To protect against invalid cross-task comparisons, the engine rejects mismatched `GENERAL/ODDBALL/MOTOR/SLEEP` family or subtype, obvious participant-ID mismatches, channel-name sets, sampling frequency, pipeline version and processing methods. It also rejects duplicate EEG SHA-256 hashes, even if files were copied or renamed. Some acquisition protocols, references, and environmental differences cannot yet be established from single-EEG JSON metadata. Ensure these are consistent before interpreting comparisons.

**Three baseline recordings are sufficient for software calculations, not for establishing a stable clinical reference range.** `Abnormal` means an exploratory statistical deviation from these baseline scans, not a diagnosis. A metric can have a baseline SD of 0, in which case the z-score is undefined and the status is `Not comparable`, not an invented infinity.

## Scientific and engineering boundaries

- All four single EEG pipelines and their plots, player, numeric tables, JSON and ZIP are reused.
- No comparative brain graphs are added. The comparison is deliberately a numerical table.
- Large `.fdt` files continue to use lazy/chunked reading when the existing streaming threshold is reached. **Four large sleep scans will take time**; full overnight scientific validation is pending.
- Mixed streaming/full-record estimators are rejected for longitudinal comparison. Reprocess all baselines/test with a consistent mode; `--large-threshold-mb 0` is one way to guarantee uniform streaming.
- Reports include `team_id` and `athlete_id` as application-supplied identifiers. No hardcoded athlete names.
- Original EEG signals are not included in transferable report ZIPs.
- A `comparison_error.json` is generated when all individual EEG reports are available but cannot be compared. The individual reports remain available for inspection.

## Tests

Run from the extracted project directory:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q tests
```

The regression suite covers `z=+3`, `z>3`, SD=0, missing metrics, distinct SHA hashes, task/subtype mismatch, mismatched sampling frequency, frontend manifest and team batch output. Run `tests/demo_comparison_run.py` for a **synthetic EEG** end-to-end demonstration. This is software verification, not evidence of human EEG physiology or medical validity.
