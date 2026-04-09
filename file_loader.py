# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


SUPPORTED_SUFFIXES = {".npy", ".txt", ".csv"}


class WaveformLoadError(Exception):
    pass


def list_supported_files(folder: str | Path) -> List[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    files: List[Path] = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append(p)
    files.sort()
    return files


def _unwrap_object_array(arr: np.ndarray) -> np.ndarray:
    if arr.dtype != object:
        return arr

    if arr.ndim == 0:
        item = arr.item()
        return _extract_numeric_array(item)

    if arr.ndim == 1 and arr.size == 1:
        return _extract_numeric_array(arr[0])

    try:
        return arr.astype(np.float32)
    except Exception as exc:
        raise WaveformLoadError(
            "该 .npy 文件是 object 数组，且内部不是可直接识别的纯数字波形。"
        ) from exc


def _extract_numeric_array(obj: Any) -> np.ndarray:
    if isinstance(obj, np.ndarray):
        return _unwrap_object_array(obj)

    if isinstance(obj, (list, tuple)):
        arr = np.asarray(obj)
        return _unwrap_object_array(arr)

    if isinstance(obj, dict):
        preferred_keys = [
            "waveform",
            "signal",
            "data",
            "trace",
            "x",
            "y",
            "arr",
            "array",
        ]
        for k in preferred_keys:
            if k in obj:
                return _extract_numeric_array(obj[k])
        for _, v in obj.items():
            try:
                return _extract_numeric_array(v)
            except Exception:
                continue
        raise WaveformLoadError("dict 内未找到可识别的数字波形字段。")

    raise WaveformLoadError(f"无法从对象类型 {type(obj)} 提取数字波形。")


def _choose_best_1d_from_2d(arr: np.ndarray, channel: int = 0) -> np.ndarray:
    arr = np.asarray(arr)

    if arr.ndim != 2:
        raise WaveformLoadError("内部错误：仅支持从 2D 数组中提取 1D 波形。")

    rows, cols = arr.shape

    # 情况1：N x C，典型表格
    if cols <= 16 and rows > cols:
        if cols == 1:
            return arr[:, 0]
        # 如果第一列近似单调递增，视作 index/time，优先取第二列
        first_col = arr[:, 0]
        if np.all(np.diff(first_col) >= 0) and cols >= 2:
            ch = min(max(channel + 1, 1), cols - 1)
            return arr[:, ch]
        ch = min(max(channel, 0), cols - 1)
        return arr[:, ch]

    # 情况2：C x N，典型多通道波形
    if rows <= 16 and cols > rows:
        ch = min(max(channel, 0), rows - 1)
        return arr[ch, :]

    # 情况3：不规则 2D，择优取方差最大的长向量
    row_vars = np.var(arr, axis=1)
    col_vars = np.var(arr, axis=0)

    if cols >= rows:
        idx = int(np.argmax(row_vars))
        return arr[idx, :]
    else:
        idx = int(np.argmax(col_vars))
        return arr[:, idx]


def _normalize_array_shape(
    arr: np.ndarray,
    dataset_index: int = 0,
    channel: int = 0,
) -> np.ndarray:
    arr = np.asarray(arr)

    if arr.dtype == object:
        arr = _unwrap_object_array(arr)

    if arr.ndim == 0:
        raise WaveformLoadError("读到的是标量，不是波形。")

    if arr.ndim == 1:
        return arr.astype(np.float32)

    if arr.ndim == 2:
        return _choose_best_1d_from_2d(arr, channel=channel).astype(np.float32)

    # >2维：先按 dataset_index 取一层，再继续压到1维
    idx = min(max(int(dataset_index), 0), arr.shape[0] - 1)
    sub = arr[idx]
    return _normalize_array_shape(sub, dataset_index=0, channel=channel).astype(np.float32)


def _load_npy(path: Path, dataset_index: int = 0, channel: int = 0) -> np.ndarray:
    try:
        arr = np.load(path, allow_pickle=False)
    except ValueError:
        # 兼容旧 object 数组
        try:
            arr = np.load(path, allow_pickle=True)
        except Exception as exc:
            raise WaveformLoadError(
                f".npy 文件读取失败：{path.name}"
            ) from exc
    except Exception as exc:
        raise WaveformLoadError(f".npy 文件读取失败：{path.name}") from exc

    if isinstance(arr, np.lib.npyio.NpzFile):
        keys = list(arr.keys())
        if not keys:
            raise WaveformLoadError(f"{path.name} 中没有数组内容。")
        arr = arr[keys[0]]

    arr = _extract_numeric_array(arr)
    return _normalize_array_shape(arr, dataset_index=dataset_index, channel=channel)


def _load_txt(path: Path, channel: int = 0) -> np.ndarray:
    for delimiter in [None, ",", "\t", " "]:
        try:
            data = np.genfromtxt(path, delimiter=delimiter, dtype=np.float32)
            if data.size == 0:
                continue
            if np.all(np.isnan(data)):
                continue
            data = np.nan_to_num(data, nan=0.0)
            return _normalize_array_shape(data, channel=channel)
        except Exception:
            continue

    # 最后兜底：pandas
    try:
        df = pd.read_csv(path, sep=None, engine="python")
        num = df.select_dtypes(include=[np.number])
        if num.shape[1] == 0:
            raise WaveformLoadError(f"{path.name} 中没有数值列。")
        return _normalize_array_shape(num.values, channel=channel)
    except Exception as exc:
        raise WaveformLoadError(f".txt 文件读取失败：{path.name}") from exc


def _load_csv(path: Path, channel: int = 0) -> np.ndarray:
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise WaveformLoadError(f".csv 文件读取失败：{path.name}") from exc

    num = df.select_dtypes(include=[np.number])
    if num.shape[1] == 0:
        raise WaveformLoadError(f"{path.name} 中没有数值列。")

    return _normalize_array_shape(num.values, channel=channel)


def load_waveform(
    path: str | Path,
    dataset_index: int = 0,
    channel: int = 0,
) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise WaveformLoadError(f"文件不存在：{path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise WaveformLoadError(
            f"暂不支持 {suffix}，目前只支持：{', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )

    if suffix == ".npy":
        waveform = _load_npy(path, dataset_index=dataset_index, channel=channel)
    elif suffix == ".txt":
        waveform = _load_txt(path, channel=channel)
    elif suffix == ".csv":
        waveform = _load_csv(path, channel=channel)
    else:
        raise WaveformLoadError(f"暂不支持的格式：{suffix}")

    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if waveform.size < 16:
        raise WaveformLoadError(f"波形长度过短，无法识别：{path.name}")

    return {
        "path": str(path),
        "name": path.name,
        "suffix": suffix,
        "waveform": waveform,
        "length": int(waveform.size),
    }