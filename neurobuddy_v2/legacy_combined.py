#!/usr/bin/env python3
"""Neuro-buddy combined EEG + IONM analysis (research/engineering use only).

Source components:
* Original EEG pipeline: per-channel signal statistics, periodogram-derived
  spectral metrics, bands/ratios, waveform/PSD/band-power plots, HTML report.
* Original IONM pipeline: format loading, EEG filtering, channel-quality and
  artifact audits, annotations, paired event markers, epochs, trial statistics,
  Kruskal-Wallis/Bonferroni tests, change screening and JSON output.

IMPORTANT: Filtering is not equivalent to removing physiological artifacts.
The pipeline FLAGs suspicious data and excludes flagged epochs from event
statistics when possible. It does not automatically perform ICA/ASR or infer
clinical intraoperative diagnoses. Rule-based classifications are unvalidated.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import platform
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save figures even from terminals/servers without a GUI
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy import signal
from scipy.stats import kurtosis, kruskal, skew


@dataclass(frozen=True)
class Settings:
    high_pass_hz: float = 1.0
    low_pass_hz: float = 45.0
    mains_frequency_hz: float = 60.0
    event_tmin_s: float = -0.2
    event_tmax_s: float = 0.8
    baseline_start_s: float = -0.2
    baseline_end_s: float = 0.0
    response_start_s: float = 0.0
    response_end_s: float = 0.8
    transient_z: float = 6.0
    stimulation_z: float = 8.0
    flatline_ptp_volts: float = 1e-12
    clipping_edge_fraction: float = 0.01
    line_noise_prominence: float = 10.0
    robust_z_threshold: float = 3.5
    low_rms_ratio: float = 0.2
    high_rms_ratio: float = 5.0
    change_z_threshold: float = 2.0
    amplitude_change_fraction: float = 0.20
    minimum_consecutive_changes: int = 3
    statistical_alpha: float = 0.05
    display_seconds: float = 10.0
    reject_flagged_epochs: bool = True
    event_mode: str = "auto"


BANDS = {"delta": (0.5, 4.0), "theta": (4.0, 8.0),
         "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}
MARKER_MAP = {str(i + 10): str(i) for i in range(1, 7)}


def choose_file_macos() -> Path:
    if platform.system() != "Darwin":
        raise ValueError("Provide an input file: python neurobuddy_combined_pipeline.py /path/to/recording.set")
    script = ('tell application "Finder"\n'
              'activate\n'
              'set selectedFile to choose file with prompt "Select EEG / IONM recording"\n'
              'return POSIX path of selectedFile\n'
              'end tell')
    r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        raise RuntimeError("No EEG recording selected")
    return Path(r.stdout.strip())


def load_recording(path: Path) -> mne.io.BaseRaw:
    p = str(path).lower()
    if p.endswith(".set"):
        reader = mne.io.read_raw_eeglab
    elif p.endswith(".edf"):
        reader = mne.io.read_raw_edf
    elif p.endswith(".bdf"):
        reader = mne.io.read_raw_bdf
    elif p.endswith(".vhdr"):
        reader = mne.io.read_raw_brainvision
    elif p.endswith((".fif", ".fif.gz")):
        reader = mne.io.read_raw_fif
    else:
        raise ValueError(f"Unsupported format: {path.suffix}. Supported: .set, .edf, .bdf, .vhdr, .fif[.gz]")
    # Companion .fdt, .eeg, .vmrk and similar files must be in the expected location.
    return reader(str(path), preload=True, verbose="ERROR")


def channel_roles(raw: mne.io.BaseRaw) -> dict[str, str]:
    roles = {}
    for name, kind in zip(raw.ch_names, raw.get_channel_types()):
        u = name.upper()
        if "EOG" in u or u in {"HEOG", "VEOG"}:
            role = "EOG"
        elif "ECG" in u or "EKG" in u:
            role = "ECG"
        elif kind == "eeg":
            role = "EEG"
        else:
            role = kind.upper()
        roles[name] = role
    return roles


def metadata(raw: mne.io.BaseRaw) -> dict:
    return {"n_channels": len(raw.ch_names), "n_samples": int(raw.n_times),
            "sampling_frequency_hz": float(raw.info["sfreq"]),
            "duration_seconds": float(raw.n_times / raw.info["sfreq"]),
            "channel_names": list(raw.ch_names),
            "channel_types": dict(zip(raw.ch_names, raw.get_channel_types())),
            "annotation_count": len(raw.annotations),
            "signal_unit": "V (EEG amplitudes, as returned by MNE)"}


def prepare_filtered(raw: mne.io.BaseRaw, eeg_channels: list[str], cfg: Settings):
    sfreq = float(raw.info["sfreq"])
    nyquist = sfreq / 2
    if cfg.high_pass_hz >= nyquist:
        raise ValueError("Sampling rate too low for the requested high-pass cutoff")
    effective_low = min(cfg.low_pass_hz, nyquist * 0.9)
    if cfg.high_pass_hz >= effective_low:
        raise ValueError("Invalid band-pass range for the recording's sampling rate")
    filtered = raw.copy().pick(eeg_channels)
    filtered.filter(cfg.high_pass_hz, effective_low, verbose="ERROR")
    return filtered, effective_low


def safe_number(value):
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [safe_json(v) for v in value]
    return safe_number(value)


def band_power(f: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    """Numerical PSD integration with equal-frequency bins, in V^2."""
    mask = (f >= low) & (f < high)
    if not np.any(mask):
        return float("nan")
    df = float(f[1] - f[0]) if len(f) > 1 else float("nan")
    return float(np.sum(psd[mask]) * df)


def eeg_statistics(x: np.ndarray, sfreq: float) -> dict:
    """All original EEG statistical and spectral metric families, on RAW EEG."""
    finite = np.isfinite(x)
    finite_fraction = float(np.mean(finite))
    if not np.all(finite):
        # Do not invisibly interpolate missing measurements.
        return {"finite_fraction": finite_fraction, "metrics_available": False,
                "reason": "Non-finite samples; no statistics calculated"}
    if x.size < 2:
        return {"finite_fraction": finite_fraction, "metrics_available": False,
                "reason": "Too few samples"}

    mean = float(np.mean(x))
    std = float(np.std(x))
    median = float(np.median(x))
    rms = float(np.sqrt(np.mean(x*x)))
    crossings = int(np.count_nonzero(np.diff(np.signbit(x))))
    f, psd = signal.periodogram(x, fs=sfreq, detrend="constant", scaling="density")
    full_power = float(np.sum(psd) * (f[1] - f[0])) if len(f) > 1 else 0.0
    band = {name: band_power(f, psd, lo, hi) for name, (lo, hi) in BANDS.items()}
    valid_band = [v for v in band.values() if np.isfinite(v)]
    total_band = float(np.sum(valid_band)) if valid_band else 0.0
    ratios = {}
    for label, a, b in [("theta_delta_ratio", "theta", "delta"),
                        ("alpha_theta_ratio", "alpha", "theta"),
                        ("beta_alpha_ratio", "beta", "alpha"),
                        ("beta_theta_ratio", "beta", "theta")]:
        divisor = band[b]
        ratios[label] = float(band[a] / divisor) if np.isfinite(divisor) and divisor > 0 else None
    p = psd / np.sum(psd) if np.sum(psd) > 0 else np.zeros_like(psd)
    p_nonzero = p[p > 0]
    alpha_mask = (f >= 8) & (f < 13)
    peak_alpha = float(f[alpha_mask][np.argmax(psd[alpha_mask])]) if np.any(alpha_mask) else None
    spectral_edge = None
    if np.sum(psd) > 0:
        idx = np.searchsorted(np.cumsum(psd), 0.95 * np.sum(psd))
        spectral_edge = float(f[min(idx, len(f)-1)])
    res = {
        "metrics_available": True,
        "finite_fraction": finite_fraction,
        "mean": mean, "standard_deviation": std, "variance": float(np.var(x)),
        "rms": rms, "peak_to_peak": float(np.ptp(x)),
        "min": float(np.min(x)), "max": float(np.max(x)),
        "median": median, "mad": float(np.median(np.abs(x - median))),
        "skewness": float(skew(x)), "kurtosis": float(kurtosis(x)),
        "zero_crossings": crossings, "zero_crossing_rate": crossings/len(x),
        "crest_factor": float(np.max(np.abs(x))/rms) if rms > 0 else None,
        "high_amplitude_fraction": float(np.mean(np.abs(x - mean) > 5*std)) if std > 0 else 0.0,
        "total_spectral_power": full_power,
        "dominant_frequency": float(f[np.argmax(psd)]),
        "peak_alpha_frequency": peak_alpha,
        "spectral_entropy": float(-np.sum(p_nonzero * np.log2(p_nonzero))),
        "spectral_edge": spectral_edge,
        "total_band_power": total_band,
    }
    for name, v in band.items():
        res[name + "_power"] = v
        res[name + "_relative"] = float(v/total_band) if total_band > 0 and np.isfinite(v) else None
    res.update(ratios)
    return res


def snr_proxy(x: np.ndarray):
    """Original IONM diff-based proxy. NOT a calibrated signal-to-noise ratio."""
    if len(x) < 3 or not np.all(np.isfinite(x)):
        return None
    sig = float(np.mean(x*x))
    noise = float(np.mean(np.diff(x)**2))
    if noise == 0:
        return None
    return float(10 * np.log10(sig/noise)) if sig > 0 else None


def clipping_edge_fraction(x: np.ndarray):
    """Fraction occurring at observed min/max (proxy only; device rails unknown)."""
    if len(x) < 10 or not np.all(np.isfinite(x)):
        return None
    # Exact extrema are meaningful only for quantized/plateau-like data.
    return float(np.mean((x == np.min(x)) | (x == np.max(x))))


def line_noise_audit(x: np.ndarray, sfreq: float, cfg: Settings) -> dict:
    mains = cfg.mains_frequency_hz
    if sfreq / 2 < mains + 5 or not np.all(np.isfinite(x)):
        return {"assessable": False, "reason": "Mains frequency outside usable Nyquist range"}
    f, p = signal.periodogram(x, fs=sfreq)
    target = (f >= mains-1) & (f <= mains+1)
    neighborhood = (f >= mains-5) & (f <= mains+5) & ~target
    if not np.any(target) or not np.any(neighborhood):
        return {"assessable": False, "reason": "Insufficient spectral bins"}
    background = float(np.median(p[neighborhood]))
    target_power = float(np.max(p[target]))
    ratio = (target_power / background) if background > 0 else None
    return {"assessable": True, "prominence_ratio": ratio,
            "detected": bool(ratio is not None and ratio >= cfg.line_noise_prominence),
            "mains_frequency_hz": mains}


def artifact_audit(raw: mne.io.BaseRaw, filtered: mne.io.BaseRaw,
                   eeg_channels: list[str], cfg: Settings):
    """IONM-style artifact flags; detect line noise on unfiltered raw!"""
    sfreq = float(raw.info["sfreq"])
    out = {}
    for ch in eeg_channels:
        orig = raw.get_data(picks=[ch])[0]
        filt = filtered.get_data(picks=[ch])[0]
        if not np.all(np.isfinite(orig)) or not np.all(np.isfinite(filt)):
            out[ch] = {"artifact_flags": ["non_finite"], "artifact_present": True,
                       "transient_count": 0, "stimulation_threshold_count": 0,
                       "transient_first_times_s": [], "line_noise": {"assessable": False}}
            continue
        sd = float(np.std(filt))
        z = np.abs((filt-np.mean(filt))/sd) if sd > 0 else np.zeros_like(filt)
        transient = np.flatnonzero(z >= cfg.transient_z)
        stim = np.flatnonzero(z >= cfg.stimulation_z)
        line_noise = line_noise_audit(orig, sfreq, cfg)
        if np.ptp(filt) == 0:
            electrode_issue = "flatline"
        elif np.std(filt) <= 0.01 * np.ptp(filt):
            electrode_issue = "low_variability"
        else:
            electrode_issue = None
        flags = []
        if line_noise.get("detected"):
            flags.append("line_noise")
        if len(transient):
            flags.append("transient")
        if len(stim):
            flags.append("stimulation_pattern_not_cause")
        if electrode_issue:
            flags.append("electrode_problem")
        out[ch] = {"artifact_flags": flags, "artifact_present": bool(flags),
                   "transient_count": int(len(transient)),
                   "stimulation_threshold_count": int(len(stim)),
                   "transient_first_times_s": (transient[:50]/sfreq).tolist(),
                   "line_noise": line_noise, "electrode_problem": electrode_issue}
    return out


def quality_audit(filtered: mne.io.BaseRaw, eeg_channels: list[str], cfg: Settings):
    out = {}
    for ch in eeg_channels:
        x = filtered.get_data(picks=[ch])[0]
        finite = float(np.mean(np.isfinite(x)))
        if finite < 1.0:
            out[ch] = {"quality": "bad", "finite_fraction": finite,
                       "flags": ["non_finite"]}
            continue
        r = float(np.sqrt(np.mean(x*x)))
        sd = float(np.std(x))
        ptp = float(np.ptp(x))
        flat = ptp <= cfg.flatline_ptp_volts
        clip = clipping_edge_fraction(x)
        # Original IONM uses an uncalibrated diff-SNR proxy for grading.
        proxy = snr_proxy(x)
        possible_clip = bool(clip is not None and clip >= cfg.clipping_edge_fraction)
        if flat or possible_clip:
            grade = "bad"
        elif proxy is not None and proxy < 0:
            grade = "bad"
        elif proxy is not None and proxy < 5:
            grade = "poor"
        elif proxy is not None and proxy < 10:
            grade = "fair"
        else:
            grade = "good"
        flags = (["flatline"] if flat else []) + (["possible_clipping"] if possible_clip else [])
        out[ch] = {"quality": grade, "finite_fraction": finite,
                   "mean": float(np.mean(x)), "rms": r,
                   "standard_deviation": sd, "variance": float(np.var(x)),
                   "peak_to_peak": ptp, "minimum": float(np.min(x)),
                   "maximum": float(np.max(x)),
                   "snr_difference_proxy_db": proxy,
                   "flatline": flat, "possible_clipping": possible_clip,
                   "clipping_edge_fraction": clip,
                   "flags": flags}
    return out


def outlier_audit(filtered: mne.io.BaseRaw, eeg_channels: list[str], quality: dict, cfg: Settings):
    rms_values = np.array([quality[ch].get("rms", np.nan) for ch in eeg_channels])
    sd_values = np.array([quality[ch].get("standard_deviation", np.nan) for ch in eeg_channels])
    med_rms = float(np.nanmedian(rms_values)) if np.any(np.isfinite(rms_values)) else None
    med_sd = float(np.nanmedian(sd_values)) if np.any(np.isfinite(sd_values)) else None
    out = {}
    for ch in eeg_channels:
        x = filtered.get_data(picks=[ch])[0]
        if not np.all(np.isfinite(x)):
            out[ch] = {"outlier": True, "review_reasons": ["non_finite"]}
            continue
        med = float(np.median(x))
        mad = float(np.median(np.abs(x-med)))
        robust_z = float(np.max(np.abs(0.6744897501960817*(x-med)/mad))) if mad > 0 else None
        ratio = float(quality[ch]["rms"]/med_rms) if med_rms and med_rms > 0 else None
        reasons = []
        if ratio is not None and ratio < cfg.low_rms_ratio:
            reasons.append("low_amplitude_vs_channels")
        if ratio is not None and ratio > cfg.high_rms_ratio:
            reasons.append("high_amplitude_vs_channels")
        if med_sd is not None and med_sd > 0 and quality[ch]["standard_deviation"] > 5*med_sd:
            reasons.append("high_variability_vs_channels")
        if quality[ch]["flatline"]:
            reasons.append("flatline")
        if robust_z is not None and robust_z >= cfg.robust_z_threshold:
            reasons.append("extreme_sample_robust_z")
        out[ch] = {"mad": mad, "robust_z_max": robust_z,
                   "rms_ratio_to_median_channel": ratio,
                   "outlier": bool(reasons), "review_reasons": reasons}
    return out


def annotation_audit(raw: mne.io.BaseRaw) -> dict:
    labels = [str(a) for a in raw.annotations.description]
    onsets = np.asarray(raw.annotations.onset, dtype=float)
    intervals = np.diff(np.sort(onsets))
    return {"total_annotations": len(labels), "unique_descriptions": sorted(set(labels)),
            "counts": dict(Counter(labels)),
            "first_annotation_time": float(np.min(onsets)) if len(onsets) else None,
            "last_annotation_time": float(np.max(onsets)) if len(onsets) else None,
            "median_annotation_interval": float(np.median(intervals)) if len(intervals) else None}


def extract_events(raw: mne.io.BaseRaw, cfg: Settings):
    """Use MNE's event sample alignment; avoid guessing marker meanings.

    Auto: use mapped marker->event sequences if present; otherwise use discrete
    annotations (non-BAD, non-boundary, duration<=0.05s) with literal labels.
    """
    if len(raw.annotations) == 0:
        return None, {}, [], "no_annotations"
    all_events, all_ids = mne.events_from_annotations(
        raw, regexp=r"^(?![Bb][Aa][Dd]|[Ee][Dd][Gg][Ee]).*$", verbose="ERROR")
    id_to_label = {v: k for k, v in all_ids.items()}
    event_entries = sorted([(int(row[0]), id_to_label[int(row[2])]) for row in all_events],
                           key=lambda item: item[0])
    pairs = []
    for i, (sample, desc) in enumerate(event_entries):
        if desc not in MARKER_MAP:
            continue
        expected = MARKER_MAP[desc]
        for subsequent, actual in event_entries[i+1:]:
            if actual == expected:
                pairs.append({"marker": desc, "event": actual,
                              "marker_sample": sample, "event_sample": subsequent,
                              "interval_seconds": float((subsequent-sample)/raw.info["sfreq"])})
                break
    selected = []
    source = "paired_markers"
    if cfg.event_mode == "paired" or (cfg.event_mode == "auto" and pairs):
        selected = [(p["event_sample"], p["event"]) for p in pairs]
    else:
        # Only instantaneous annotations are treated as events. Sleep stages and
        # continuous state annotations are not assumed to be discrete stimuli.
        for onset, duration, desc in zip(raw.annotations.onset,
                                         raw.annotations.duration, raw.annotations.description):
            if duration > 0.05 or str(desc).lower().startswith(("bad", "edge", "boundary")):
                continue
            # mne.events_from_annotations already handles file-specific offsets.
            candidate = [sample for sample, label in event_entries if label == desc]
            if candidate:
                selected.append((candidate.pop(0), str(desc)))
                event_entries.remove((selected[-1][0], str(desc)))
        source = "discrete_annotations"
    # Event samples must be unique for MNE epochs. Keep first occurrence.
    seen, unique = set(), []
    for sample, label in sorted(selected):
        if sample not in seen:
            unique.append((sample, label))
            seen.add(sample)
    if not unique:
        return None, {}, pairs, source + "_none_matched"
    labels = sorted({name for _, name in unique})
    ids = {name: i+1 for i, name in enumerate(labels)}
    event_array = np.array([[s, 0, ids[name]] for s, name in unique], dtype=int)
    return event_array, ids, pairs, source


def boundary_intervals(raw: mne.io.BaseRaw):
    return [(float(onset), float(onset+duration))
            for onset, duration, desc in zip(raw.annotations.onset,
                                              raw.annotations.duration,
                                              raw.annotations.description)
            if str(desc).strip().lower() == "boundary"]


def make_epochs(filtered: mne.io.BaseRaw, raw: mne.io.BaseRaw,
                events: np.ndarray | None, event_ids: dict, cfg: Settings,
                quality: dict, artifacts: dict):
    if events is None:
        return None, {"candidate_events": 0, "mne_accepted": 0,
                      "rejected_boundary": 0, "rejected_artifact": 0,
                      "accepted_epochs": 0, "notes": "No compatible discrete event annotations"}
    # Filtered was made by picking channels from raw; both share first sample.
    epochs = mne.Epochs(filtered, events, event_id=event_ids,
                        tmin=cfg.event_tmin_s, tmax=cfg.event_tmax_s,
                        baseline=None, reject_by_annotation=True,
                        preload=True, verbose="ERROR")
    mne_n = len(epochs)
    boundaries = boundary_intervals(raw)
    to_drop = []
    boundary_bad, art_bad = 0, 0
    for idx, event in enumerate(epochs.events):
        seconds = (event[0]-raw.first_samp)/float(raw.info["sfreq"])
        intersects = any(seconds + cfg.event_tmin_s <= end and
                         seconds + cfg.event_tmax_s >= start
                         for start, end in boundaries)
        if intersects:
            boundary_bad += 1
            to_drop.append(idx)
            continue
        if not cfg.reject_flagged_epochs:
            continue
        epoch = epochs.get_data(copy=False)[idx]
        suspicious = False
        for ci, ch in enumerate(epochs.ch_names):
            if quality[ch]["quality"] == "bad":
                # Bad channel should not discard all epochs: channel exclusion
                # is handled later via quality flags in event statistics.
                continue
            sd = quality[ch].get("standard_deviation", 0.0)
            mean = quality[ch].get("mean", 0.0)
            if sd > 0 and np.any(np.abs((epoch[ci]-mean)/sd) >= cfg.transient_z):
                suspicious = True
                break
        if suspicious:
            art_bad += 1
            to_drop.append(idx)
    if to_drop:
        epochs.drop(sorted(set(to_drop)), reason="flagged_artifact_or_boundary", verbose="ERROR")
    return epochs, {"candidate_events": int(len(events)),
                    "mne_accepted": mne_n,
                    "rejected_by_mne": int(len(events)-mne_n),
                    "rejected_boundary": boundary_bad,
                    "rejected_artifact": art_bad,
                    "accepted_epochs": len(epochs),
                    "reject_flagged_epochs": cfg.reject_flagged_epochs}


def event_response_records(epochs, quality: dict, cfg: Settings):
    if epochs is None or len(epochs) == 0:
        return []
    arr = epochs.get_data(copy=True)
    times = epochs.times
    b = (times >= cfg.baseline_start_s) & (times < cfg.baseline_end_s)
    r = (times >= cfg.response_start_s) & (times <= cfg.response_end_s)
    if not np.any(b) or not np.any(r):
        return []
    reverse_ids = {v: k for k, v in epochs.event_id.items()}
    out = []
    for trial, epoch in enumerate(arr):
        event_label = reverse_ids[int(epochs.events[trial, 2])]
        for ci, ch in enumerate(epochs.ch_names):
            if quality[ch]["quality"] == "bad":
                continue
            xb, xr = epoch[ci, b], epoch[ci, r]
            base_mean, resp_mean = float(xb.mean()), float(xr.mean())
            base_rms = float(np.sqrt(np.mean(xb*xb)))
            resp_rms = float(np.sqrt(np.mean(xr*xr)))
            peak_ix, trough_ix = int(np.argmax(xr)), int(np.argmin(xr))
            out.append({
                "trial": trial, "event_code": event_label, "channel": ch,
                "event_sample": int(epochs.events[trial, 0]),
                "baseline_mean": base_mean, "event_mean": resp_mean,
                "mean_change": resp_mean-base_mean,
                "percent_change": 100*(resp_mean-base_mean)/abs(base_mean) if abs(base_mean)>1e-12 else None,
                "baseline_rms": base_rms, "event_rms": resp_rms,
                "rms_change": resp_rms-base_rms,
                "rms_percent_change": 100*(resp_rms-base_rms)/base_rms if base_rms>0 else None,
                "baseline_peak_to_peak": float(np.ptp(xb)),
                "event_peak_to_peak": float(np.ptp(xr)),
                "response_peak": float(np.max(xr)),
                "response_trough": float(np.min(xr)),
                "positive_peak_latency_s": float(times[r][peak_ix]),
                "negative_peak_latency_s": float(times[r][trough_ix]),
            })
    return out


def trial_statistics(records: list[dict]):
    groups = defaultdict(list)
    for item in records:
        groups[(item["event_code"], item["channel"])].append(item)
    out = []
    for (event, channel), group in sorted(groups.items()):
        changes = np.array([v["mean_change"] for v in group])
        rms_changes = np.array([v["rms_change"] for v in group])
        out.append({"event_code": event, "channel": channel, "n_trials": len(group),
                    "mean_change": float(changes.mean()),
                    "median_change": float(np.median(changes)),
                    "standard_deviation": float(np.std(changes, ddof=1)) if len(changes)>1 else None,
                    "robust_spread_mad": float(np.median(np.abs(changes-np.median(changes)))),
                    "mean_rms_change": float(rms_changes.mean()),
                    "median_rms_change": float(np.median(rms_changes)),
                    "response_peak_mean": float(np.mean([r["response_peak"] for r in group])),
                    "response_trough_mean": float(np.mean([r["response_trough"] for r in group]))})
    return out


def statistical_comparisons(records: list[dict], cfg: Settings):
    by_channel = defaultdict(lambda: defaultdict(list))
    for r in records:
        by_channel[r["channel"]][r["event_code"]].append(r["mean_change"])
    out = []
    for channel, classes in sorted(by_channel.items()):
        # At least two trials per class avoids a misleading statistical test.
        groups = {k: v for k, v in classes.items() if len(v)>=2}
        if len(groups) < 2:
            continue
        try:
            h, p = kruskal(*groups.values())
        except ValueError:
            continue
        N = sum(len(x) for x in groups.values())
        k = len(groups)
        eps = max(0.0, float((h-k+1)/(N-k))) if N>k else None
        out.append({"channel": channel, "groups": list(groups.keys()),
                    "group_trial_counts": {name: len(values) for name, values in groups.items()},
                    "H_statistic": float(h), "p_value": float(p),
                    "epsilon_squared": eps,
                    "significant_uncorrected": bool(p<cfg.statistical_alpha)})
    corrected = cfg.statistical_alpha/len(out) if out else None
    for r in out:
        r["bonferroni_alpha"] = corrected
        r["significant_bonferroni"] = bool(r["p_value"]<corrected)
    return out


def evidence_summary(quality: dict, artifacts: dict, records: list[dict], cfg: Settings):
    """Rule-based engineering screening. No fabricated probability/confidence."""
    grouped = defaultdict(list)
    for r in records:
        grouped[r["channel"]].append(r)
    out = []
    for ch, q in quality.items():
        ordered = sorted(grouped[ch], key=lambda r: (r["event_sample"], r["event_code"]))
        changes = []
        if ordered:
            baselines = np.array([r["baseline_rms"] for r in ordered])
            baseline_mean = float(baselines.mean())
            baseline_sd = float(baselines.std())
            for trial in ordered:
                delta = abs(trial["event_rms"]-baseline_mean)
                if baseline_sd>1e-12:
                    changed = delta/baseline_sd >= cfg.change_z_threshold
                else:
                    changed = delta >= cfg.amplitude_change_fraction * max(baseline_mean, 1e-12)
                changes.append(bool(changed))
        max_consecutive, current_run = 0, 0
        for changed in changes:
            current_run = current_run+1 if changed else 0
            max_consecutive = max(max_consecutive, current_run)
        persistent = max_consecutive >= cfg.minimum_consecutive_changes
        has_change = any(changes)
        has_artifact = artifacts[ch]["artifact_present"]
        if q["quality"] in {"bad", "poor"}:
            classification = "uncertain_quality"
        elif has_artifact and not has_change:
            classification = "possible_artifact"
        elif has_change and persistent and not has_artifact:
            classification = "possible_physiological_change_needs_review"
        elif has_change:
            classification = "uncertain_change"
        else:
            classification = "no_change_detected_by_rules"
        out.append({"channel": ch, "quality": q["quality"],
                    "artifact_present": has_artifact, "change_detected": has_change,
                    "persistent_change": persistent, "max_consecutive_changes": max_consecutive,
                    "trials_evaluated": len(ordered), "classification": classification,
                    "note": "Unvalidated engineering screening, not a diagnosis"})
    return out


def save_graphs(out_dir: Path, raw: mne.io.BaseRaw,
                filtered: mne.io.BaseRaw, channels: list[str],
                eeg_stats: dict, quality: dict, records: list[dict], stats: list[dict],
                cfg: Settings) -> dict[str, str]:
    result = {}
    if not channels:
        return result
    sfreq = float(raw.info["sfreq"])
    count = min(raw.n_times, max(2, int(cfg.display_seconds*sfreq)))
    time = np.arange(count)/sfreq
    ch = channels[0]
    raw_x = raw.get_data(picks=[ch], start=0, stop=count)[0]
    filtered_x = filtered.get_data(picks=[ch], start=0, stop=count)[0]

    # Source signals are returned by MNE in volts; convert to microvolts on plots.
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(time, raw_x*1e6, lw=0.8)
    axes[0].set(title=f"Original EEG: {ch}", ylabel="Amplitude (µV)")
    axes[1].plot(time, filtered_x*1e6, lw=0.8)
    axes[1].set(title=f"Band-pass filtered EEG: {ch}", ylabel="Amplitude (µV)", xlabel="Time (seconds)")
    fig.tight_layout()
    fig.savefig(out_dir/"waveform_raw_vs_filtered.png", dpi=150)
    plt.close(fig)
    result["Raw vs. filtered EEG waveform"] = "waveform_raw_vs_filtered.png"

    f, p = signal.periodogram(raw.get_data(picks=[ch])[0], fs=sfreq)
    ff, pp = signal.periodogram(filtered.get_data(picks=[ch])[0], fs=sfreq)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(f, p, label="Original", lw=1)
    ax.plot(ff, pp, label="Filtered", lw=1, alpha=0.8)
    ax.set(xlim=(0, min(50, sfreq/2)), xlabel="Frequency (Hz)", ylabel="PSD (V²/Hz)",
           title=f"Power spectrum: {ch}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir/"power_spectrum.png", dpi=150)
    plt.close(fig)
    result["EEG power spectrum"] = "power_spectrum.png"

    n = len(channels)
    fig, ax = plt.subplots(figsize=(max(11, min(20, n*0.38)), 5))
    positions = np.arange(n)
    width = 0.20
    for i, band in enumerate(BANDS):
        vals = [100*eeg_stats[c].get(band+"_relative", 0)
                if eeg_stats[c].get(band+"_relative") is not None else 0 for c in channels]
        ax.bar(positions+(i-1.5)*width, vals, width, label=band.title())
    ax.set_xticks(positions, labels=channels, rotation=90)
    ax.set(xlabel="EEG channel", ylabel="Relative power (%)", title="Relative EEG band power by channel")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir/"relative_band_power.png", dpi=150)
    plt.close(fig)
    result["Relative EEG band power"] = "relative_band_power.png"

    fig, ax = plt.subplots(figsize=(max(10, min(20, n*0.38)), 4))
    ax.bar(channels, [quality[c].get("rms", 0)*1e6 for c in channels])
    ax.tick_params(axis="x", rotation=90)
    ax.set(title="Filtered channel RMS", ylabel="RMS (µV)")
    fig.tight_layout()
    fig.savefig(out_dir/"channel_rms.png", dpi=150)
    plt.close(fig)
    result["Channel RMS"] = "channel_rms.png"

    qcodes = {"bad":0, "poor":1, "fair":2, "good":3}
    fig, ax = plt.subplots(figsize=(max(10, min(20,n*0.38)), 4))
    ax.bar(channels, [qcodes[quality[c]["quality"]] for c in channels])
    ax.set_yticks([0,1,2,3], labels=["Bad","Poor","Fair","Good"])
    ax.tick_params(axis="x", rotation=90)
    ax.set(title="IONM-style channel quality (heuristic)")
    fig.tight_layout()
    fig.savefig(out_dir/"channel_quality.png", dpi=150)
    plt.close(fig)
    result["Channel quality"] = "channel_quality.png"

    if records:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(np.array([r["mean_change"] for r in records])*1e6, bins=20)
        ax.set(xlabel="Response minus baseline mean (µV)", ylabel="Count",
               title="Event-related response changes")
        fig.tight_layout()
        fig.savefig(out_dir/"event_response_histogram.png", dpi=150)
        plt.close(fig)
        result["Event response changes"] = "event_response_histogram.png"
    if stats:
        fig, ax = plt.subplots(figsize=(max(10,min(20,len(stats)*0.38)),4))
        ax.bar([s["channel"] for s in stats],
               [-math.log10(max(s["p_value"], 1e-300)) for s in stats])
        ax.axhline(-math.log10(cfg.statistical_alpha), lw=1, ls="--", label="Uncorrected α=0.05")
        ax.set(ylabel="−log₁₀(p)", title="Kruskal-Wallis results by channel")
        ax.tick_params(axis="x", rotation=90)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir/"event_statistics.png", dpi=150)
        plt.close(fig)
        result["Event statistical comparisons"] = "event_statistics.png"
    return result


def h(value):
    return html.escape(str(value), quote=True)


def fmt(v):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.5g}"
    return str(v)


def html_table(rows: list[dict], fields: list[str] | None = None) -> str:
    if not rows:
        return "<p>No applicable results.</p>"
    fields = fields or list(rows[0])
    head = "".join(f"<th>{h(k.replace('_',' '))}</th>" for k in fields)
    body = "".join("<tr>"+"".join(f"<td>{h(fmt(r.get(k)))}</td>" for k in fields)+"</tr>" for r in rows)
    return f'<div class="table-scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def save_html(out_dir: Path, source: Path, report: dict, figures: dict[str, str]):
    metric_rows = [{"channel": ch, **values} for ch, values in report["eeg_statistics_raw"].items()]
    quality_rows = [{"channel": ch, **{k:v for k,v in q.items() if not isinstance(v, (list, dict))}}
                    for ch, q in report["ionm_quality_filtered"].items()]
    artifact_rows = [{"channel": ch,
                      "artifact_flags": ", ".join(a.get("artifact_flags", [])),
                      "transient_count": a.get("transient_count"),
                      "stimulation_threshold_count": a.get("stimulation_threshold_count"),
                      "line_noise_prominence_ratio": a.get("line_noise", {}).get("prominence_ratio")}
                     for ch,a in report["ionm_artifacts"].items()]
    evidence_rows = report["evidence_screening"]
    figure_html = "".join(f'<section><h2>{h(title)}</h2><img src="{h(file)}" alt="{h(title)}"></section>'
                          for title,file in figures.items())
    stats = report["event_statistics"]
    source_title = h(source.name)
    css = """
    :root{color-scheme:light}body{font:15px/1.55 -apple-system,BlinkMacSystemFont,Segoe UI,Arial,sans-serif;
    background:#f4f6f9;color:#1e293b;margin:0}main{max-width:1400px;margin:auto;padding:24px}
    h1{font-size:29px}h2{font-size:20px;margin:0 0 12px}section{background:white;padding:22px;border-radius:12px;
    margin:18px 0;box-shadow:0 1px 4px #00000010}.table-scroll{overflow:auto;max-height:650px}
    table{border-collapse:collapse;font-size:12px;white-space:nowrap;width:max-content;min-width:100%}
    th,td{padding:8px 10px;text-align:left;border:1px solid #e4e7ee}th{background:#eaf0f5;position:sticky;top:0}
    img{width:100%;max-width:1200px;display:block}small,.muted{color:#64748b}
    code{padding:2px 5px;background:#f1f5f9;border-radius:3px}
    """
    summary = report["metadata"]
    document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1"><title>Neuro-buddy Combined EEG + IONM Report</title>
    <style>{css}</style></head><body><main><h1>Neuro-buddy: EEG + IONM report</h1>
    <p class="muted">Research / engineering screening only. Not a clinical IONM diagnosis.</p>
    <section><h2>Dataset</h2><p><b>File:</b> {source_title}</p>
    <p><b>Channels:</b> {summary['n_channels']} total; {len(report['eeg_statistics_raw'])} EEG &nbsp;
    <b>Sampling:</b> {summary['sampling_frequency_hz']:.3f} Hz &nbsp;
    <b>Duration:</b> {summary['duration_seconds']:.2f} seconds</p>
    <p><b>EEG statistics:</b> unfiltered recording (including 0.5–1 Hz delta).
    <b>IONM quality & event statistics:</b> band-pass filtered EEG.</p>
    <p><b>Events:</b> {h(report['event_source'])}. <b>Accepted epochs:</b> {report['epochs']['accepted_epochs']}.</p>
    <p><b>Amplitude units:</b> original values in volts (V) in JSON/tables; waveform graphs use µV.</p></section>
    <section><h2>Original EEG statistical + spectral metrics</h2>
    <p class="muted">Periodogram PSD integrated over each band; frequencies in Hz, powers in V².</p>
    {html_table(metric_rows)}</section>
    <section><h2>IONM channel quality on filtered EEG</h2>
    <p class="muted">SNR is a difference-based proxy and the grade is heuristic, not validated.</p>
    {html_table(quality_rows)}</section>
    <section><h2>Artifact detection</h2>
    <p class="muted">Artifact detection is not artifact removal. Mains noise is assessed on original (unfiltered) EEG.</p>
    {html_table(artifact_rows)}</section>
    <section><h2>Event trial statistics</h2>{html_table(report['event_trial_statistics'])}</section>
    <section><h2>Kruskal-Wallis / Bonferroni</h2>{html_table(stats)}</section>
    <section><h2>Evidence screening</h2>{html_table(evidence_rows)}</section>
    {figure_html}
    <section><p>Complete machine-readable details: <code>combined_evidence_report.json</code>.</p>
    <small>Interpretation warning: amplitude excursions, event codes, and heuristic confidence cannot alone establish physiological cause.</small></section>
    </main></body></html>'''
    (out_dir/"combined_eeg_ionm_report.html").write_text(document, encoding="utf-8")


def run_pipeline(source: Path, output_dir: Path | None = None,
                 cfg: Settings | None = None):
    cfg = cfg or Settings()
    source = Path(source).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    out_dir = (Path(output_dir).expanduser().resolve() if output_dir is not None else
               source.parent / (source.stem + "_NEUROBUDDY_COMBINED_REPORT"))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[1/9] Loading {source.name}")
    raw = load_recording(source)
    roles = channel_roles(raw)
    eeg_channels = [ch for ch in raw.ch_names if roles[ch] == "EEG"]
    if not eeg_channels:
        raise ValueError("No channels identified as EEG. Check original channel types and names.")
    print(f"[2/9] Keeping {len(eeg_channels)} EEG channels; filtering separate copy")
    filtered, actual_lowpass = prepare_filtered(raw, eeg_channels, cfg)
    print("[3/9] Calculating original EEG metrics on raw signals")
    eeg_stats = {ch: eeg_statistics(raw.get_data(picks=[ch])[0], raw.info["sfreq"])
                 for ch in eeg_channels}
    print("[4/9] IONM channel-quality, outlier and artifact audits")
    quality = quality_audit(filtered, eeg_channels, cfg)
    outliers = outlier_audit(filtered, eeg_channels, quality, cfg)
    artifacts = artifact_audit(raw, filtered, eeg_channels, cfg)
    print("[5/9] Reading event annotations and building epochs")
    annotations = annotation_audit(raw)
    event_array, event_ids, pairs, event_source = extract_events(raw, cfg)
    epochs, epoch_counts = make_epochs(filtered, raw, event_array, event_ids, cfg, quality, artifacts)
    print(f"[6/9] Calculating event statistics ({epoch_counts['accepted_epochs']} accepted epochs)")
    records = event_response_records(epochs, quality, cfg)
    trials = trial_statistics(records)
    comparisons = statistical_comparisons(records, cfg)
    evidence = evidence_summary(quality, artifacts, records, cfg)
    print("[7/9] Generating plots")
    figures = save_graphs(out_dir, raw, filtered, eeg_channels, eeg_stats, quality,
                          records, comparisons, cfg)
    print("[8/9] Writing JSON + HTML reports")
    report = {
        "pipeline": "Neuro-buddy Combined EEG + IONM v1.0",
        "analysis_type": "Research/engineering only; not validated for clinical use",
        "source_recording": str(source),
        "parameters": {**asdict(cfg), "effective_low_pass_hz": actual_lowpass},
        "metadata": metadata(raw), "channel_roles": roles,
        "eeg_statistics_raw": eeg_stats,
        "ionm_quality_filtered": quality, "ionm_outliers_filtered": outliers,
        "ionm_artifacts": artifacts,
        "annotations": annotations, "validated_marker_event_pairs": pairs,
        "event_source": event_source, "event_classes": event_ids,
        "epochs": epoch_counts, "event_response_records": records,
        "event_trial_statistics": trials, "event_statistics": comparisons,
        "evidence_screening": evidence, "figures": figures,
        "warnings": [
            "SNR is an uncalibrated difference-power proxy, not physical SNR.",
            "Filtering does not itself remove ocular, muscle, or motion artifacts.",
            "Artifact-based epoch rejection is heuristic and may reject real responses.",
            "Evidence classes are screening labels, not diagnoses or calibrated probabilities.",
            "Markers 11-16 paired with 1-6 have no inferred biological meaning.",
            "Frequency-band results require sufficient sampling rate and recording duration.",
            "Spectral statistics are from original recording; quality/event metrics from filtered EEG.",
        ]}
    json_path = out_dir/"combined_evidence_report.json"
    json_path.write_text(json.dumps(safe_json(report), indent=2, allow_nan=False), encoding="utf-8")
    save_html(out_dir, source, report, figures)
    print(f"[9/9] Complete: {out_dir}")
    return report, out_dir


def main():
    parser = argparse.ArgumentParser(description="Neuro-buddy: combined EEG and IONM analysis")
    parser.add_argument("file", nargs="?", help="EEG recording (.set/.edf/.bdf/.vhdr/.fif); macOS Finder if omitted")
    parser.add_argument("--output", type=Path, help="Output folder")
    parser.add_argument("--mains", type=float, choices=[50.0, 60.0], default=60.0,
                        help="Electrical mains frequency (50 or 60 Hz; default 60)")
    parser.add_argument("--events", choices=["auto", "paired", "annotations"], default="auto",
                        help="Auto mapped marker pairs, or discrete annotations")
    parser.add_argument("--keep-flagged-epochs", action="store_true",
                        help="Do not discard event epochs flagged as transient-contaminated")
    args = parser.parse_args()
    source = Path(args.file) if args.file else choose_file_macos()
    cfg = Settings(mains_frequency_hz=args.mains, event_mode=args.events,
                   reject_flagged_epochs=not args.keep_flagged_epochs)
    _, output = run_pipeline(source, args.output, cfg)
    print("HTML report:", output / "combined_eeg_ionm_report.html")
    print("JSON report:", output / "combined_evidence_report.json")


if __name__ == "__main__":
    main()
