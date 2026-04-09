import os
import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(obj: Dict, path: str):
    folder = os.path.dirname(path)
    if folder:
        ensure_dir(folder)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def extract_pick_index_from_probability(prob_p: np.ndarray, threshold: float = 0.5) -> int:
    """
    prob_p: [L]
    规则:
    1. 找第一个 >= threshold 的点
    2. 若没有，则取最大概率点
    """
    idx = np.where(prob_p >= threshold)[0]
    if len(idx) > 0:
        return int(idx[0])
    return int(np.argmax(prob_p))


def compute_pick_metrics(true_pick: int, pred_pick: int, tolerance_samples: int = 40) -> Dict[str, Optional[float]]:
    if true_pick < 0 and pred_pick < 0:
        return {"tp": 0, "fp": 0, "fn": 0, "tn": 1, "residual": None}

    if true_pick < 0 and pred_pick >= 0:
        return {"tp": 0, "fp": 1, "fn": 0, "tn": 0, "residual": None}

    if true_pick >= 0 and pred_pick < 0:
        return {"tp": 0, "fp": 0, "fn": 1, "tn": 0, "residual": None}

    residual = pred_pick - true_pick
    if abs(residual) <= tolerance_samples:
        return {"tp": 1, "fp": 0, "fn": 0, "tn": 0, "residual": float(residual)}
    else:
        return {"tp": 0, "fp": 1, "fn": 1, "tn": 0, "residual": float(residual)}


def summarize_pick_metrics(rows: List[Dict[str, Optional[float]]]) -> Dict[str, Optional[float]]:
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    tn = sum(r["tn"] for r in rows)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    residuals = [r["residual"] for r in rows if r["residual"] is not None]
    residual_mean = float(np.mean(residuals)) if len(residuals) > 0 else None
    residual_std = float(np.std(residuals)) if len(residuals) > 0 else None

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "residual_mean": residual_mean,
        "residual_std": residual_std,
    }


def estimate_snr(waveform_1d: np.ndarray, pick_idx: int, win: int = 500):
    if pick_idx is None or pick_idx < win or pick_idx + win > len(waveform_1d):
        return None
    noise = waveform_1d[pick_idx - win: pick_idx]
    signal = waveform_1d[pick_idx: pick_idx + win]
    noise_std = float(np.std(noise))
    signal_std = float(np.std(signal))
    if noise_std < 1e-12:
        return None
    return signal_std / noise_std


def extract_features(waveform: np.ndarray, sample_rate_hz: float = 6.25e6) -> Dict[str, float]:
    """
    waveform: [L] 或 [C, L]
    默认取第一通道
    """
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 2:
        waveform_1d = waveform[0]
    else:
        waveform_1d = waveform

    peak_amplitude = float(np.max(np.abs(waveform_1d)))
    rms = float(np.sqrt(np.mean(np.square(waveform_1d))))
    energy = float(np.sum(np.square(waveform_1d)))
    duration_samples = int(len(waveform_1d))

    fft_vals = np.fft.rfft(waveform_1d)
    freqs = np.fft.rfftfreq(len(waveform_1d), d=1.0 / sample_rate_hz)
    amp = np.abs(fft_vals)

    if np.sum(amp) < 1e-12:
        main_freq = 0.0
        spectral_centroid = 0.0
        bandwidth = 0.0
    else:
        main_freq = float(freqs[np.argmax(amp)])
        spectral_centroid = float(np.sum(freqs * amp) / np.sum(amp))
        bandwidth = float(np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * amp) / np.sum(amp)))

    return {
        "peak_amplitude": peak_amplitude,
        "rms": rms,
        "energy": energy,
        "duration_samples": duration_samples,
        "main_freq_hz": main_freq,
        "spectral_centroid_hz": spectral_centroid,
        "bandwidth_hz": bandwidth,
    }


def plot_waveform_with_pick(waveform: np.ndarray, true_pick: int, pred_pick: int, save_path: str, title: str = ""):
    folder = os.path.dirname(save_path)
    if folder:
        ensure_dir(folder)

    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = waveform[0]

    plt.figure(figsize=(12, 4))
    plt.plot(waveform, linewidth=1)

    if true_pick is not None and true_pick >= 0:
        plt.axvline(true_pick, linestyle="--", linewidth=1.5, label="true_pick")
    if pred_pick is not None and pred_pick >= 0:
        plt.axvline(pred_pick, linestyle="-", linewidth=1.5, label="pred_pick")

    if title:
        plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()