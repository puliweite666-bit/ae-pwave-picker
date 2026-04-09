from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


CANDIDATE_MODULE_NAMES = [
    "src.model",
    "src.models",
    "src.network",
    "src.net",
]

COMMON_BUILDER_NAMES = [
    "build_model",
    "get_model",
    "create_model",
    "make_model",
    "build_network",
    "get_network",
    "create_network",
    "make_net",
]

COMMON_META_KEYS = [
    "config",
    "model_config",
    "model_kwargs",
    "model_args",
    "args",
    "hparams",
    "hyper_parameters",
    "hyperparams",
]

COMMON_HINT_KEYS = [
    "model_name",
    "model_class",
    "arch",
    "architecture",
    "net_name",
    "network_name",
    "class_name",
]


# =========================================================
# 基础工具
# =========================================================
def _torch_load_compat(checkpoint_path: str) -> Any:
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def _is_tensor_like(x: Any) -> bool:
    return isinstance(x, (torch.Tensor, np.ndarray))


def _looks_like_state_dict(obj: Any) -> bool:
    if not isinstance(obj, dict) or not obj:
        return False
    tensor_like_count = 0
    for _, value in obj.items():
        if _is_tensor_like(value):
            tensor_like_count += 1
        else:
            return False
    return tensor_like_count > 0


def _extract_meta_dict(obj: Any) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}

    if not isinstance(obj, dict):
        return meta

    for key in COMMON_META_KEYS:
        value = obj.get(key)
        if isinstance(value, dict):
            meta.update(value)

    for key in COMMON_HINT_KEYS:
        if key in obj and obj[key] is not None:
            meta[key] = obj[key]

    for key, value in obj.items():
        if key in meta:
            continue
        if isinstance(value, (str, int, float, bool)):
            meta[key] = value

    return meta


def _extract_state_dict_and_meta(obj: Any) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any], Optional[nn.Module]]:
    if isinstance(obj, nn.Module):
        return None, {}, obj

    meta = _extract_meta_dict(obj)

    if not isinstance(obj, dict):
        return None, meta, None

    preferred_keys = [
        "state_dict",
        "model_state_dict",
        "net_state_dict",
        "network_state_dict",
        "weights",
        "params",
    ]

    for key in preferred_keys:
        if key in obj and _looks_like_state_dict(obj[key]):
            return obj[key], meta, None

    for key in ["model", "net", "network"]:
        if key in obj:
            if isinstance(obj[key], nn.Module):
                return None, meta, obj[key]
            if _looks_like_state_dict(obj[key]):
                return obj[key], meta, None

    if _looks_like_state_dict(obj):
        return obj, meta, None

    return None, meta, None


def _import_candidate_modules() -> List[Any]:
    modules: List[Any] = []
    for module_name in CANDIDATE_MODULE_NAMES:
        try:
            modules.append(importlib.import_module(module_name))
        except Exception:
            pass
    return modules


def _normalized_hint_names(meta: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    for key in COMMON_HINT_KEYS:
        value = meta.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text and text not in names:
            names.append(text)
    return names


def _collect_kwargs(meta: Dict[str, Any]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    for key, value in meta.items():
        if isinstance(value, dict):
            kwargs.update(value)
        else:
            kwargs[key] = value
    return kwargs


def _call_factory(factory: Any, kwargs: Dict[str, Any]) -> Optional[nn.Module]:
    try:
        sig = inspect.signature(factory)
    except Exception:
        sig = None

    try:
        if sig is None:
            result = factory()
        else:
            params = list(sig.parameters.values())

            if len(params) == 0:
                result = factory()
            elif len(params) == 1 and params[0].name in {"config", "cfg", "args", "hparams", "kwargs"}:
                result = factory(kwargs)
            else:
                usable_kwargs = {}
                has_var_kw = False
                for p in params:
                    if p.kind == inspect.Parameter.VAR_KEYWORD:
                        has_var_kw = True
                    elif p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
                        if p.name in kwargs:
                            usable_kwargs[p.name] = kwargs[p.name]

                if usable_kwargs:
                    result = factory(**usable_kwargs)
                elif has_var_kw:
                    result = factory(**kwargs)
                else:
                    result = factory()

        if isinstance(result, nn.Module):
            return result
    except Exception:
        return None

    return None


def _iter_candidate_factories(modules: Iterable[Any], hint_names: List[str]) -> Iterable[Tuple[str, Any]]:
    yielded = set()

    for module in modules:
        for hint in hint_names:
            if hasattr(module, hint):
                factory = getattr(module, hint)
                key = (module.__name__, hint)
                if key not in yielded:
                    yielded.add(key)
                    yield "%s.%s" % (module.__name__, hint), factory

    for module in modules:
        for name in COMMON_BUILDER_NAMES:
            if hasattr(module, name):
                factory = getattr(module, name)
                key = (module.__name__, name)
                if key not in yielded:
                    yielded.add(key)
                    yield "%s.%s" % (module.__name__, name), factory

    for module in modules:
        for name, obj in inspect.getmembers(module, inspect.isclass):
            try:
                if not issubclass(obj, nn.Module):
                    continue
            except Exception:
                continue

            if obj in {nn.Module, nn.Sequential, nn.ModuleList, nn.ModuleDict}:
                continue

            key = (module.__name__, name)
            if key not in yielded:
                yielded.add(key)
                yield "%s.%s" % (module.__name__, name), obj


def _strip_prefix_if_needed(state_dict: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(k.startswith(prefix) for k in keys):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def _candidate_state_dicts(state_dict: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    variants: List[Tuple[str, Dict[str, Any]]] = [("raw", state_dict)]

    v1 = _strip_prefix_if_needed(state_dict, "module.")
    if v1 is not state_dict:
        variants.append(("strip_module", v1))

    v2 = _strip_prefix_if_needed(state_dict, "model.")
    if v2 is not state_dict and v2 is not v1:
        variants.append(("strip_model", v2))

    v3 = _strip_prefix_if_needed(v1, "model.")
    if v3 is not v1 and v3 is not v2 and v3 is not state_dict:
        variants.append(("strip_module_then_model", v3))

    return variants


def _try_load_into_model(model: nn.Module, state_dict: Dict[str, Any]) -> Tuple[bool, str]:
    for variant_name, variant_sd in _candidate_state_dicts(state_dict):
        try:
            model.load_state_dict(variant_sd, strict=True)
            return True, "strict=True, variant=%s" % variant_name
        except Exception:
            pass

        try:
            missing, unexpected = model.load_state_dict(variant_sd, strict=False)
            return True, "strict=False, variant=%s, missing=%d, unexpected=%d" % (
                variant_name,
                len(missing),
                len(unexpected),
            )
        except Exception:
            pass

    return False, ""


# =========================================================
# 模型恢复
# =========================================================
def restore_model_from_checkpoint(checkpoint_path: str, device: str = "cpu") -> nn.Module:
    checkpoint_path = str(Path(checkpoint_path))

    obj = _torch_load_compat(checkpoint_path)
    state_dict, meta, full_model = _extract_state_dict_and_meta(obj)

    if isinstance(full_model, nn.Module):
        model = full_model.to(device)
        model.eval()
        return model

    if state_dict is None:
        raise RuntimeError("checkpoint 已读取，但既不是完整 nn.Module，也没有找到可用 state_dict。")

    modules = _import_candidate_modules()
    if not modules:
        raise RuntimeError("没有找到可搜索的模型模块。请确认至少存在 src/model.py 或 src/models.py。")

    hint_names = _normalized_hint_names(meta)
    kwargs = _collect_kwargs(meta)

    tried_messages: List[str] = []

    for factory_name, factory in _iter_candidate_factories(modules, hint_names):
        model = _call_factory(factory, kwargs)
        if model is None:
            tried_messages.append("[跳过] %s 无法实例化" % factory_name)
            continue

        ok, load_msg = _try_load_into_model(model, state_dict)
        if ok:
            model = model.to(device)
            model.eval()
            return model

        tried_messages.append("[失败] %s 可实例化，但 state_dict 不匹配" % factory_name)

    joined = "\n".join(tried_messages[:20])
    raise RuntimeError(
        "best_model.pth 已读取到，但无法自动恢复成可推理模型。\n"
        "这通常说明当前是 state_dict，而你的真实模型类还没被正确定位，"
        "或者模型构造参数与 checkpoint 不一致。\n\n"
        "已尝试：\n%s" % joined
    )


# =========================================================
# 兼容旧 infer_service.py 的老接口
# =========================================================
def _resolve_checkpoint_path(config: Any = None, checkpoint_path: Optional[str] = None) -> str:
    if checkpoint_path:
        return str(checkpoint_path)

    if isinstance(config, str):
        return config

    if isinstance(config, dict):
        for key in [
            "default_model_path",
            "model_path",
            "checkpoint",
            "checkpoint_path",
            "pth_path",
            "weights_path",
        ]:
            value = config.get(key)
            if value:
                return str(value)

    raise KeyError(
        "没有找到模型路径，请在 config/app_config.json 中至少提供下面任意一个字段：\n"
        "default_model_path / model_path / checkpoint / checkpoint_path"
    )


def _resolve_device(config: Any = None, device: Optional[str] = None) -> str:
    if device:
        return str(device)

    if isinstance(config, dict):
        value = config.get("device")
        if value:
            return str(value)

    return "cpu"


def build_model_and_load_weights(
    config: Any = None,
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None,
    **kwargs: Any,
) -> nn.Module:
    ckpt = _resolve_checkpoint_path(config=config, checkpoint_path=checkpoint_path)
    dev = _resolve_device(config=config, device=device)
    return restore_model_from_checkpoint(ckpt, device=dev)


# =========================================================
# 推理输入/输出适配
# =========================================================
def _to_numpy_waveform(waveform: Any) -> np.ndarray:
    if isinstance(waveform, np.ndarray):
        arr = waveform
    elif isinstance(waveform, torch.Tensor):
        arr = waveform.detach().cpu().numpy()
    else:
        arr = np.asarray(waveform)

    if arr.dtype == object:
        raise ValueError("波形数据是 object dtype，不能直接推理，请换成纯数字数组。")

    arr = np.asarray(arr, dtype=np.float32)
    return arr


def _normalize_waveform(arr: np.ndarray, mode: str = "zscore") -> np.ndarray:
    mode = (mode or "none").lower()

    if mode in {"none", "raw", ""}:
        return arr

    if mode == "zscore":
        mean = float(arr.mean())
        std = float(arr.std())
        if std < 1e-8:
            std = 1.0
        return (arr - mean) / std

    if mode in {"minmax", "min_max"}:
        vmin = float(arr.min())
        vmax = float(arr.max())
        denom = vmax - vmin
        if denom < 1e-8:
            denom = 1.0
        return (arr - vmin) / denom

    return arr


def _waveform_to_tensor(waveform: Any, normalize: str = "zscore") -> torch.Tensor:
    arr = _to_numpy_waveform(waveform)
    arr = np.squeeze(arr)

    if arr.ndim == 0:
        raise ValueError("波形为空。")

    if arr.ndim == 1:
        arr = _normalize_waveform(arr, normalize)
        tensor = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)  # [1,1,T]
        return tensor

    if arr.ndim == 2:
        # 自动判断是 [C,T] 还是 [T,C]
        h, w = arr.shape
        if h <= 8 and w > h:
            arr = np.asarray([_normalize_waveform(ch, normalize) for ch in arr], dtype=np.float32)
            tensor = torch.from_numpy(arr).float().unsqueeze(0)  # [1,C,T]
            return tensor
        elif w <= 8 and h > w:
            arr = arr.T
            arr = np.asarray([_normalize_waveform(ch, normalize) for ch in arr], dtype=np.float32)
            tensor = torch.from_numpy(arr).float().unsqueeze(0)  # [1,C,T]
            return tensor
        else:
            # 当成单通道二维数据拍平
            arr = arr.reshape(-1)
            arr = _normalize_waveform(arr, normalize)
            tensor = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
            return tensor

    if arr.ndim >= 3:
        arr = arr.reshape(-1)
        arr = _normalize_waveform(arr, normalize)
        tensor = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
        return tensor

    raise ValueError("不支持的波形维度。")


def _extract_prediction_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, (list, tuple)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item

    if isinstance(output, dict):
        for key in [
            "prob",
            "probs",
            "probability",
            "probabilities",
            "logits",
            "output",
            "out",
            "pred",
            "prediction",
            "pick",
            "score",
            "scores",
            "y",
        ]:
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value

        for _, value in output.items():
            if isinstance(value, torch.Tensor):
                return value

    raise RuntimeError("模型 forward 已执行，但没有提取到 tensor 输出。")


def _to_probability_1d(pred: torch.Tensor, target_len: Optional[int] = None) -> np.ndarray:
    x = pred.detach().float().cpu()

    while x.ndim > 2:
        x = x[0]
    if x.ndim == 2:
        if x.shape[0] == 1:
            x = x[0]
        elif x.shape[1] == 1:
            x = x[:, 0]
        else:
            x = x[0]

    if x.ndim != 1:
        x = x.reshape(-1)

    x_np = x.numpy()

    if x_np.size == 0:
        raise RuntimeError("模型输出为空。")

    vmin = float(x_np.min())
    vmax = float(x_np.max())

    if vmin < 0.0 or vmax > 1.0:
        x = torch.sigmoid(torch.from_numpy(x_np)).float()
        x_np = x.numpy()

    if target_len is not None and target_len > 0 and x_np.shape[0] != target_len:
        xx = torch.from_numpy(x_np).float().view(1, 1, -1)
        xx = F.interpolate(xx, size=target_len, mode="linear", align_corners=False)
        x_np = xx.view(-1).numpy()

    return x_np.astype(np.float32)


def predict_probability_sequence(
    model: nn.Module,
    waveform: Any,
    device: str = "cpu",
    normalize: str = "zscore",
    **kwargs: Any,
) -> np.ndarray:
    if model is None:
        raise RuntimeError("model 为空，无法推理。")

    tensor = _waveform_to_tensor(waveform, normalize=normalize)
    target_len = int(tensor.shape[-1])

    dev = torch.device(device)
    model = model.to(dev)
    tensor = tensor.to(dev)

    model.eval()
    with torch.no_grad():
        output = model(tensor)
        pred = _extract_prediction_tensor(output)
        prob = _to_probability_1d(pred, target_len=target_len)

    return prob


# =========================================================
# 其他兼容别名
# =========================================================
def load_user_model(checkpoint_path: str, device: str = "cpu") -> nn.Module:
    return restore_model_from_checkpoint(checkpoint_path, device=device)


def load_model_from_checkpoint(checkpoint_path: str, device: str = "cpu") -> nn.Module:
    return restore_model_from_checkpoint(checkpoint_path, device=device)


def auto_restore_torch_model(checkpoint_path: str, device: str = "cpu") -> nn.Module:
    return restore_model_from_checkpoint(checkpoint_path, device=device)


def build_inference_model(checkpoint_path: str, device: str = "cpu") -> nn.Module:
    return restore_model_from_checkpoint(checkpoint_path, device=device)


__all__ = [
    "restore_model_from_checkpoint",
    "build_model_and_load_weights",
    "predict_probability_sequence",
    "load_user_model",
    "load_model_from_checkpoint",
    "auto_restore_torch_model",
    "build_inference_model",
]