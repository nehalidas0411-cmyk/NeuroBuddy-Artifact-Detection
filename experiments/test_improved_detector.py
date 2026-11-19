
import numpy as np

from improved_artifact_detector import detect_window_artifacts


SFREQ = 250
DURATION = 20
N = SFREQ * DURATION

rng = np.random.default_rng(42)
t = np.arange(N) / SFREQ

clean = (
    20e-6 * np.sin(2 * np.pi * 10 * t)
    + 8e-6 * np.sin(2 * np.pi * 6 * t)
    + rng.normal(0, 3e-6, N)
)

flatline = clean.copy()
flatline[2000:3000] = 0

burst = clean.copy()
burst[2000:2500] += (
    300e-6 * np.sin(
        2 * np.pi * 15 * np.arange(500) / SFREQ
    )
)

for name, signal in [
    ("clean", clean),
    ("flatline_segment", flatline),
    ("high_amplitude_burst", burst),
]:
    result = detect_window_artifacts(signal, SFREQ)

    print(f"\n{name}")
    print("Detected:", result["artifact_present"])
    print("Flags:", result["artifact_flags"])
    print("Intervals:", result["artifact_intervals"])
