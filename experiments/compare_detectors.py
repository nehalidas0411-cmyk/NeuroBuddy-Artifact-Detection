
from pathlib import Path
import csv
import sys

import mne
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from neurobuddy_v2.legacy_combined import Settings, artifact_audit
from improved_artifact_detector import detect_window_artifacts

SFREQ = 250
DURATION = 20
N = SFREQ * DURATION
SEEDS = range(30)

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def make_clean_eeg(seed):
    rng = np.random.default_rng(seed)
    t = np.arange(N) / SFREQ
    return (
        20e-6 * np.sin(2 * np.pi * 10 * t)
        + 8e-6 * np.sin(2 * np.pi * 6 * t)
        + rng.normal(0, 3e-6, N)
    )


def inject_artifact(signal, kind):
    x = signal.copy()

    if kind == "single_spike":
        x[2500] += 500e-6

    elif kind == "multiple_spikes":
        for i in [1000, 2000, 3000, 4000]:
            x[i] += 500e-6

    elif kind == "flatline_segment":
        x[2000:3000] = 0

    elif kind == "high_amplitude_burst":
        t = np.arange(500) / SFREQ
        x[2000:2500] += 300e-6 * np.sin(2 * np.pi * 15 * t)

    elif kind in ("line_noise_50hz", "line_noise_60hz"):
        freq = 50 if kind == "line_noise_50hz" else 60
        t = np.arange(N) / SFREQ
        x += 40e-6 * np.sin(2 * np.pi * freq * t)

    return x


def run_original(signal, mains_frequency):
    info = mne.create_info(
        ["EEG001"], SFREQ, ch_types=["eeg"]
    )
    raw = mne.io.RawArray(signal[np.newaxis, :], info, verbose=False)
    filtered = raw.copy().filter(1.0, 45.0, verbose=False)

    cfg = Settings(mains_frequency_hz=mains_frequency)
    return artifact_audit(
        raw, filtered, ["EEG001"], cfg
    )["EEG001"]


def metrics(rows, column):
    tp = fp = tn = fn = 0

    for row in rows:
        actual = row["expected_artifact"]
        predicted = row[column]

        if actual == 1 and predicted == 1:
            tp += 1
        elif actual == 0 and predicted == 1:
            fp += 1
        elif actual == 0 and predicted == 0:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def main():
    cases = [
        "clean",
        "single_spike",
        "multiple_spikes",
        "flatline_segment",
        "high_amplitude_burst",
        "line_noise_50hz",
        "line_noise_60hz",
    ]

    rows = []

    for seed in SEEDS:
        clean = make_clean_eeg(seed)

        for case in cases:
            signal = inject_artifact(clean, case)
            mains = 60 if case == "line_noise_60hz" else 50

            original = run_original(signal, mains)
            window = detect_window_artifacts(signal, SFREQ)

            old_flag = bool(original["artifact_present"])
            enhanced_flag = (
                old_flag or bool(window["artifact_present"])
            )

            rows.append({
                "seed": seed,
                "case": case,
                "expected_artifact": int(case != "clean"),
                "original_detected": int(old_flag),
                "enhanced_detected": int(enhanced_flag),
                "original_flags": ";".join(
                    original["artifact_flags"]
                ),
                "window_flags": ";".join(
                    window["artifact_flags"]
                ),
            })

    raw_path = RESULTS / "comparison_per_recording.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = [
        {"detector": "original", **metrics(rows, "original_detected")},
        {"detector": "enhanced", **metrics(rows, "enhanced_detected")},
    ]

    summary_path = RESULTS / "comparison_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    print("\nNEUROBUDDY ARTIFACT DETECTION COMPARISON")
    print(f"Recordings evaluated: {len(rows)}")

    for result in summary:
        print(f"\n{result['detector'].upper()}")
        for key, value in result.items():
            if key != "detector":
                print(f"  {key}: {value}")

    print(f"\nSaved: {raw_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()