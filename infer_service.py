# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.file_loader import load_waveform, list_supported_files
from src.user_model_adapter import restore_model_from_checkpoint


class InferenceError(Exception):
    pass


class PWaveInferenceService:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.device = self.config.get("device", "cpu")
        self.threshold = float(self.config.get("threshold", 0.50))
        self.normalize = str(self.config.get("normalize", "zscore")).lower()
        self.sampling_rate = float(self.config.get("sampling_rate", 1.0))
        self.model_path = str(
            self.config.get(
                "default_model_path",
                str(Path(__file__).resolve().parents[1] / "models" / "best_model.pth"),
            )
        )
        self.model = None
        self.loaded_model_path = None

    def load_model(self, model_path: Optional[str] = None, device: Optional[str] = None) -> None:
        if model_path:
            self.model_path = model_path
        if device:
            self.device = device

        model_file = Path(self.model_path)
        if not model_file.exists():
            raise InferenceError(f"模型文件不存在：{model_file}")

        try:
            self.model = restore_model_from_checkpoint(str(model_file), self.device)
            self.model.eval()
            self.loaded_model_path = str(model_file)
        except Exception as exc:
            raise InferenceError(f"模型加载失败：{exc}") from exc

    def _preprocess_waveform(self, waveform: np.ndarray) -> np.ndarray:
        x = np.asarray(waveform, dtype=np.float32).reshape(-1)

        if self.normalize == "zscore":
            mean = float(np.mean(x))
            std = float(np.std(x))
            if std < 1e-8:
                std = 1.0
            x = (x - mean) / std
        elif self.normalize == "minmax":
            xmin = float(np.min(x))
            xmax = float(np.max(x))
            if abs(xmax - xmin) < 1e-8:
                x = x - xmin
            else:
                x = (x - xmin) / (xmax - xmin)
        elif self.normalize == "none":
            pass
        else:
            # 默认 zscore
            mean = float(np.mean(x))
            std = float(np.std(x))
            if std < 1e-8:
                std = 1.0
            x = (x - mean) / std

        return x.astype(np.float32)

    def _extract_probability_sequence(self, model_output: Any, target_len: int) -> np.ndarray:
        out = model_output

        if isinstance(out, dict):
            for key in ["prob", "probs", "probability", "output", "logits", "pred"]:
                if key in out:
                    out = out[key]
                    break
            else:
                out = next(iter(out.values()))

        if isinstance(out, (list, tuple)):
            out = out[0]

        if not torch.is_tensor(out):
            out = torch.as_tensor(out)

        out = out.detach().float().cpu()

        # 尽量压成 1D 序列
        while out.ndim > 1:
            if out.shape[0] == 1:
                out = out.squeeze(0)
            else:
                break

        if out.ndim == 2:
            # 可能是 [C, L] 或 [L, C]
            if out.shape[0] <= 8 and out.shape[1] > out.shape[0]:
                out = out[0]
            elif out.shape[1] <= 8 and out.shape[0] > out.shape[1]:
                out = out[:, 0]
            else:
                out = out.reshape(-1)

        if out.ndim != 1:
            out = out.reshape(-1)

        # 若不是概率，尝试 sigmoid
        out_np = out.numpy().astype(np.float32)
        if np.min(out_np) < 0.0 or np.max(out_np) > 1.0:
            out = torch.sigmoid(torch.from_numpy(out_np))
            out_np = out.numpy().astype(np.float32)

        if out_np.size <= 1:
            raise InferenceError("模型输出不是有效的序列概率。")

        if out_np.size != target_len:
            src_x = np.linspace(0.0, 1.0, out_np.size)
            dst_x = np.linspace(0.0, 1.0, target_len)
            out_np = np.interp(dst_x, src_x, out_np).astype(np.float32)

        return out_np

    def predict_waveform(
        self,
        waveform: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        if self.model is None:
            self.load_model()

        thr = self.threshold if threshold is None else float(threshold)
        x = self._preprocess_waveform(waveform)
        target_len = int(x.size)

        tensor = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0).to(self.device)

        try:
            with torch.no_grad():
                out = self.model(tensor)
            prob = self._extract_probability_sequence(out, target_len=target_len)
        except Exception as exc:
            raise InferenceError(f"模型推理失败：{exc}") from exc

        peak_idx = int(np.argmax(prob))
        peak_prob = float(prob[peak_idx])

        detected = peak_prob >= thr
        pick_index = peak_idx if detected else None
        pick_time = None
        if detected and self.sampling_rate > 0:
            pick_time = float(pick_index / self.sampling_rate)

        return {
            "ok": True,
            "detected": bool(detected),
            "pick_index": pick_index,
            "pick_time": pick_time,
            "confidence": peak_prob,
            "threshold": thr,
            "probability_seq": prob,
            "wave_length": target_len,
        }

    def predict_file(
        self,
        file_path: str,
        threshold: Optional[float] = None,
        dataset_index: int = 0,
        channel: int = 0,
    ) -> Dict[str, Any]:
        item = load_waveform(file_path, dataset_index=dataset_index, channel=channel)
        result = self.predict_waveform(item["waveform"], threshold=threshold)
        result.update(
            {
                "file_path": item["path"],
                "file_name": item["name"],
                "waveform": item["waveform"],
                "suffix": item["suffix"],
                "length": item["length"],
            }
        )
        return result

    def predict_folder(
        self,
        folder: str,
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        files = list_supported_files(folder)
        results: List[Dict[str, Any]] = []

        for p in files:
            try:
                r = self.predict_file(str(p), threshold=threshold)
                r["status"] = "成功" if r["detected"] else "未过阈值"
                r["message"] = ""
            except Exception as exc:
                r = {
                    "file_path": str(p),
                    "file_name": p.name,
                    "suffix": p.suffix.lower(),
                    "length": None,
                    "ok": False,
                    "detected": False,
                    "pick_index": None,
                    "pick_time": None,
                    "confidence": None,
                    "threshold": threshold if threshold is not None else self.threshold,
                    "status": "失败",
                    "message": str(exc),
                }
            results.append(r)

        return results