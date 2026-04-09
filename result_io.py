from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.infer_service import PickResult


def ensure_parent_dir(file_path: str) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def append_pick_result(csv_path: str, result: PickResult) -> None:
    ensure_parent_dir(csv_path)
    path = Path(csv_path)

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_path": result.source_path,
        "dataset_key": result.dataset_key or "",
        "pick_index": result.pick_index,
        "confidence": f"{result.confidence:.6f}",
        "accepted": int(result.accepted),
        "threshold": f"{result.threshold:.6f}",
        "sampling_rate": f"{result.sampling_rate:.6f}",
        "time_seconds": f"{result.time_seconds:.9f}",
        "time_microseconds": f"{result.time_microseconds:.3f}",
        "waveform_length": result.waveform_length,
    }

    write_header = not path.exists()

    with path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_pick_figure(
    figure_path: str,
    waveform: np.ndarray,
    result: PickResult,
    max_points_to_plot: int = 20000,
    line_width: float = 1.0,
    pick_line_width: float = 1.5,
    pick_line_style: str = "--",
) -> None:
    ensure_parent_dir(figure_path)

    y = np.asarray(waveform, dtype=np.float32).reshape(-1)
    n = y.size

    if n > max_points_to_plot:
        idx = np.linspace(0, n - 1, num=max_points_to_plot, dtype=np.int32)
        x_plot = idx
        y_plot = y[idx]
    else:
        x_plot = np.arange(n)
        y_plot = y

    plt.figure(figsize=(12, 5))
    plt.plot(x_plot, y_plot, linewidth=line_width)
    plt.axvline(result.pick_index, linewidth=pick_line_width, linestyle=pick_line_style)

    title = (
        f"P-wave pick @ index={result.pick_index}, "
        f"time={result.time_microseconds:.2f} us, "
        f"conf={result.confidence:.4f}"
    )
    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150)
    plt.close()