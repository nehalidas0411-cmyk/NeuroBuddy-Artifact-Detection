
import numpy as np


def detect_window_artifacts(
    signal,
    sfreq,
    window_seconds=1.0,
    flatline_ptp_volts=2e-6,
    burst_rms_ratio=3.0,
):
    """
    Detect localized flatline and high-amplitude artifacts.

    Parameters
    ----------
    signal : array-like
        One EEG channel, in volts.
    sfreq : float
        Sampling frequency in Hz.
    window_seconds : float
        Analysis window length in seconds.

    Returns
    -------
    dict
        Recording-level detection and artifact intervals.
    """
    x = np.asarray(signal, dtype=float)

    if x.ndim != 1:
        raise ValueError("Expected a one-dimensional EEG signal.")

    if not np.isfinite(x).all():
        return {
            "artifact_present": True,
            "artifact_flags": ["non_finite"],
            "artifact_intervals": [],
        }

    window_size = max(1, int(round(window_seconds * sfreq)))

    windows = []
    for start in range(0, len(x), window_size):
        end = min(start + window_size, len(x))
        segment = x[start:end]

        if len(segment) < window_size:
            continue

        windows.append({
            "start_s": start / sfreq,
            "end_s": end / sfreq,
            "ptp": float(np.ptp(segment)),
            "rms": float(np.sqrt(np.mean(segment ** 2))),
        })

    if not windows:
        return {
            "artifact_present": False,
            "artifact_flags": [],
            "artifact_intervals": [],
        }

    rms_values = np.array([w["rms"] for w in windows])

    # Lower-half median limits the influence of high-RMS bursts.
    reference_rms = float(np.median(rms_values))

    intervals = []
    flags = set()

    for window in windows:
        kind = None

        if window["ptp"] <= flatline_ptp_volts:
            kind = "flatline_segment"

        elif (
            reference_rms > 0
            and window["rms"] > burst_rms_ratio * reference_rms
        ):
            kind = "high_amplitude_burst"

        if kind:
            flags.add(kind)
            intervals.append({
                "start_s": window["start_s"],
                "end_s": window["end_s"],
                "type": kind,
            })

    return {
        "artifact_present": bool(flags),
        "artifact_flags": sorted(flags),
        "artifact_intervals": intervals,
        "reference_rms": reference_rms,
    }