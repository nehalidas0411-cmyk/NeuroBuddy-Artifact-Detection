
from pathlib import Path
import csv
import sys

import mne
import numpy as np

# Allow execution from the project root or experiments folder.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neurobuddy_v2.legacy_combined import Settings, artifact_audit


SFREQ = 250
DURATION = 20
N = SFREQ * DURATION
RNG = np.random.default_rng(42)

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def make_clean_eeg():
    """Synthetic EEG-like signal, expressed in volts."""
    t = np.arange(N) / SFREQ

    signal = (
        20e-6 * np.sin(2 * np.pi * 10 * t)
        + 8e-6 * np.sin(2 * np.pi * 6 * t)
        + RNG.normal(0, 3e-6, N)
    )

    return signal


def inject_artifact(signal, kind):
    x = signal.copy()

    if kind == "single_spike":
        x[2500] += 500e-6

    elif kind == "multiple_spikes":
        for index in [1000, 2000, 3000, 4000]:
            x[index] += 500e-6

    elif kind == "flatline_segment":
        x[2000:3000] = 0

    elif kind == "high_amplitude_burst":
        x[2000:2500] += (
            300e-6 * np.sin(
                2 * np.pi * 15 * np.arange(500) / SFREQ
            )
        )

    elif kind == "line_noise_50hz":
        t = np.arange(N) / SFREQ
        x += 40e-6 * np.sin(2 * np.pi * 50 * t)

    elif kind == "line_noise_60hz":
        t = np.arange(N) / SFREQ
        x += 40e-6 * np.sin(2 * np.pi * 60 * t)

    return x


def run_detector(signal, mains_frequency):
    info = mne.create_info(
        ch_names=["EEG001"],
        sfreq=SFREQ,
        ch_types=["eeg"],
    )

    raw = mne.io.RawArray(
        signal[np.newaxis, :],
        info,
        verbose=False,
    )

    filtered = raw.copy().filter(
        l_freq=1.0,
        h_freq=45.0,
        verbose=False,
    )

    cfg = Settings(mains_frequency_hz=mains_frequency)

    return artifact_audit(
        raw,
        filtered,
        ["EEG001"],
        cfg,
    )["EEG001"]


def main():
    cases = [
        ("clean", None, 50),
        ("single_spike", "single_spike", 50),
        ("multiple_spikes", "multiple_spikes", 50),
        ("flatline_segment", "flatline_segment", 50),
        ("high_amplitude_burst", "high_amplitude_burst", 50),
        ("line_noise_50hz", "line_noise_50hz", 50),
        ("line_noise_60hz", "line_noise_60hz", 60),
    ]

    rows = []

    # Generate one identical clean reference for every case.
    clean_reference = make_clean_eeg()

    for name, artifact, mains_frequency in cases:
        signal = clean_reference.copy()

        if artifact is not None:
            signal = inject_artifact(signal, artifact)

        result = run_detector(signal, mains_frequency)

        row = {
            "case": name,
            "expected_artifact": int(artifact is not None),
            "detected_artifact": int(result["artifact_present"]),
            "correct": int(
                bool(artifact is not None)
                == bool(result["artifact_present"])
            ),
            "flags": ";".join(result["artifact_flags"]),
            "transient_count": result["transient_count"],
            "stimulation_count": result[
                "stimulation_threshold_count"
            ],
            "electrode_problem": result.get("electrode_problem"),
            "line_noise_detected": result.get(
                "line_noise", {}
            ).get("detected"),
        }

        rows.append(row)

        print(
            f"{name:25s} "
            f"expected={row['expected_artifact']} "
            f"detected={row['detected_artifact']} "
            f"flags={row['flags']}"
        )

    output = RESULTS_DIR / "baseline_artifacts.csv"

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved baseline results to: {output}")


if __name__ == "__main__":
    main()