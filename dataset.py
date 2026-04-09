# -*- coding: utf-8 -*-
import os
import json
from typing import Dict, List, Tuple, Optional, Any

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_h5_key(s: str) -> str:
    s = str(s).strip().replace("\\", "/")
    s = os.path.basename(s)
    s = os.path.splitext(s)[0]
    return s


def normalize_list_key(s: str) -> str:
    """
    data_list 中一行常见形如:
    AE_976_s_0102_50MPa_Min40_50_10/SR2
    训练样本 key 取前半段:
    AE_976_s_0102_50MPa_Min40_50_10
    """
    s = str(s).strip().replace("\\", "/")
    parts = [x for x in s.split("/") if x]
    if len(parts) >= 2:
        return parts[-2].strip()

    s = os.path.basename(s)
    s = os.path.splitext(s)[0]
    return s


def normalize_csv_key(s: str) -> str:
    return normalize_list_key(s)


def read_manifest_file(txt_path: str) -> List[str]:
    items: List[str] = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                items.append(s)
    return items


def build_p_pick_map(quake_phase_csv: str) -> Dict[str, int]:
    """
    从 quake_phase.csv 中读取人工 P 波标签。
    兼容：
    1) P/S 在 phase_type
    2) P/S 实际写在 phase_amplitude
    """
    if not os.path.exists(quake_phase_csv):
        raise FileNotFoundError(f"找不到标签文件: {quake_phase_csv}")

    df = pd.read_csv(quake_phase_csv)

    required_cols = {"file_name", "phase_index"}
    miss = required_cols - set(df.columns)
    if miss:
        raise ValueError(f"{quake_phase_csv} 缺少字段: {miss}，当前列={list(df.columns)}")

    df = df.copy()
    df["phase_index"] = pd.to_numeric(df["phase_index"], errors="coerce")
    df = df.dropna(subset=["phase_index"]).copy()
    df["phase_index"] = df["phase_index"].astype(int)

    phase_col = None

    if "phase_type" in df.columns:
        pt = df["phase_type"].astype(str).str.strip().str.upper()
        if pt.isin(["P", "S"]).any():
            df["phase_kind"] = pt
            phase_col = "phase_type"

    if phase_col is None and "phase_amplitude" in df.columns:
        pa = df["phase_amplitude"].astype(str).str.strip().str.upper()
        if pa.isin(["P", "S"]).any():
            df["phase_kind"] = pa
            phase_col = "phase_amplitude"

    if phase_col is None:
        raise ValueError(
            f"{quake_phase_csv} 中未找到有效的 P/S 标记列，可检查 phase_type / phase_amplitude。"
        )

    print(f"[INFO] P/S 标签列使用: {phase_col}")

    df = df[df["phase_kind"] == "P"].copy()
    if len(df) == 0:
        raise ValueError(f"{quake_phase_csv} 中没有 P 标签记录。")

    df["norm_key"] = df["file_name"].astype(str).map(normalize_csv_key)

    pick_map: Dict[str, int] = {}
    for k, g in df.groupby("norm_key"):
        pick_map[k] = int(g["phase_index"].min())

    return pick_map


def zscore_per_channel(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (x - mean) / std


def make_gaussian_target(length: int, center: int, sigma: float) -> np.ndarray:
    t = np.arange(length, dtype=np.float32)
    y = np.exp(-0.5 * ((t - float(center)) / float(sigma)) ** 2)
    return y.astype(np.float32)


def _looks_like_waveform_array(arr: np.ndarray) -> bool:
    if not isinstance(arr, np.ndarray):
        return False
    if arr.size == 0:
        return False
    if arr.dtype.kind not in ("f", "i", "u"):
        return False
    if arr.ndim not in (1, 2):
        return False

    if arr.ndim == 1:
        return arr.shape[0] >= 64

    a, b = arr.shape
    if max(a, b) < 64:
        return False
    if min(a, b) > 64:
        return False
    return True


def _to_channel_first(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)

    if arr.ndim == 1:
        arr = arr[None, :]  # [1, T]
    elif arr.ndim == 2:
        if arr.shape[0] > arr.shape[1]:
            arr = arr.T
    else:
        raise ValueError(f"不支持的波形维度: shape={arr.shape}")

    return arr.astype(np.float32)


def _recursive_find_numeric_dataset(
    obj: Any,
    path: str = "",
    depth: int = 0,
    max_depth: int = 6,
) -> Optional[Tuple[str, np.ndarray]]:
    """
    递归在 H5 group 中寻找“最像波形”的数值数组。
    返回 (dataset_path, array)
    """
    if depth > max_depth:
        return None

    if isinstance(obj, h5py.Dataset):
        try:
            arr = np.asarray(obj[()])
            if _looks_like_waveform_array(arr):
                return path, arr
        except Exception:
            return None
        return None

    if isinstance(obj, h5py.Group):
        preferred_keys = [
            "waveform", "data", "signal", "trace",
            "wave", "sig", "x", "y",
            "AE", "ae", "raw", "raw_data"
        ]

        for k in preferred_keys:
            if k in obj:
                sub = obj[k]
                sub_path = f"{path}/{k}" if path else k
                found = _recursive_find_numeric_dataset(sub, sub_path, depth + 1, max_depth)
                if found is not None:
                    return found

        for k in obj.keys():
            sub = obj[k]
            sub_path = f"{path}/{k}" if path else k
            found = _recursive_find_numeric_dataset(sub, sub_path, depth + 1, max_depth)
            if found is not None:
                return found

    return None


def safe_read_h5_waveform(h5f: h5py.File, raw_key: str) -> np.ndarray:
    """
    list 中 raw_key 形如:
    AE_976_s_0102_50MPa_Min40_50_10/SR2

    H5 顶层 key 形如:
    AE_976_s_0102_50MPa_Min40_50_10
    """
    sample_key = normalize_list_key(raw_key)
    candidates = [sample_key, raw_key]

    found_key = None
    for k in candidates:
        if k in h5f:
            found_key = k
            break

    if found_key is None:
        raise KeyError(f"H5 中找不到样本：raw_key={raw_key}, sample_key={sample_key}")

    obj = h5f[found_key]

    if isinstance(obj, h5py.Dataset):
        arr = np.asarray(obj[()])
        if not _looks_like_waveform_array(arr):
            raise ValueError(f"H5 key={found_key} 是 Dataset，但形状不符合波形要求: shape={arr.shape}")
        return _to_channel_first(arr)

    if isinstance(obj, h5py.Group):
        found = _recursive_find_numeric_dataset(obj, path=found_key, depth=0, max_depth=8)
        if found is None:
            child_keys = list(obj.keys())
            raise ValueError(
                f"H5 key={found_key} 是 Group，递归后仍未找到可用波形数组。"
                f"该 Group 子键有: {child_keys[:20]}"
            )

        _, arr = found
        return _to_channel_first(arr)

    raise ValueError(f"无法解析 H5 样本: key={found_key}, type={type(obj)}")


class AEH5Dataset(Dataset):
    """
    用 quake_phase.csv 的真实人工 P 波标签构建数据集。
    """
    def __init__(
        self,
        h5_path: str,
        list_path: str,
        quake_phase_csv: str,
        label_sigma: float = 20.0,
        return_meta: bool = False,
    ):
        self.h5_path = h5_path
        self.list_path = list_path
        self.quake_phase_csv = quake_phase_csv
        self.label_sigma = float(label_sigma)
        self.return_meta = return_meta

        if not os.path.exists(self.h5_path):
            raise FileNotFoundError(f"h5 不存在: {self.h5_path}")
        if not os.path.exists(self.list_path):
            raise FileNotFoundError(f"list 不存在: {self.list_path}")
        if not os.path.exists(self.quake_phase_csv):
            raise FileNotFoundError(f"quake_phase.csv 不存在: {self.quake_phase_csv}")

        self.raw_keys = read_manifest_file(self.list_path)
        self.pick_map = build_p_pick_map(self.quake_phase_csv)

        self.samples: List[Tuple[str, int]] = []
        drop_no_pick = 0

        for raw_key in self.raw_keys:
            sample_key = normalize_list_key(raw_key)
            if sample_key not in self.pick_map:
                drop_no_pick += 1
                continue
            true_pick = int(self.pick_map[sample_key])
            self.samples.append((raw_key, true_pick))

        if len(self.samples) == 0:
            raise RuntimeError(
                "没有构建出有效样本。请检查：\n"
                "1) data_list_*.txt 是否是完整训练/测试列表\n"
                "2) quake_phase.csv 的 file_name 是否与 list 同一命名体系\n"
                "3) H5 顶层 key 是否对应样本名"
            )

        print("=" * 80)
        print(f"[AEH5Dataset] list_path = {self.list_path}")
        print(f"[AEH5Dataset] 原始样本数 = {len(self.raw_keys)}")
        print(f"[AEH5Dataset] 有效P标签样本数 = {len(self.samples)}")
        print(f"[AEH5Dataset] 缺少P标签被丢弃 = {drop_no_pick}")
        print("=" * 80)

        self._h5: Optional[h5py.File] = None

    def _get_h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        raw_key, true_pick = self.samples[idx]
        h5f = self._get_h5()

        x = safe_read_h5_waveform(h5f, raw_key)   # [C, T]
        x = zscore_per_channel(x)

        _, t = x.shape
        true_pick = int(np.clip(true_pick, 0, t - 1))
        y = make_gaussian_target(t, true_pick, self.label_sigma)

        x_t = torch.from_numpy(x).float()         # [C, T]
        y_t = torch.from_numpy(y).float()         # [T]
        pick_t = torch.tensor(true_pick, dtype=torch.long)

        if self.return_meta:
            sample_key = normalize_list_key(raw_key)

            meta = {
                "raw_key": raw_key,
                "sample_key": sample_key,
                "norm_key": sample_key,
                "length": t,
            }
            return x_t, y_t, pick_t, meta

        return x_t, y_t, pick_t

    def __del__(self):
        try:
            if self._h5 is not None:
                self._h5.close()
        except Exception:
            pass