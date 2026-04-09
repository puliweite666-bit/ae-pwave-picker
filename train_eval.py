# -*- coding: utf-8 -*-
import os
import sys
import json
import random
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import load_config, AEH5Dataset
from src.model import SimplePPickerNet


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def split_train_val_indices(n: int, val_ratio: float = 0.1, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)
    n_val = max(1, int(n * val_ratio)) if n >= 10 else max(1, min(n - 1, 1)) if n > 1 else 0
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    if len(train_idx) == 0 and len(val_idx) > 0:
        train_idx = val_idx[:1]
    return train_idx, val_idx


class SubsetDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]


def pick_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits, dim=1)


def calc_metrics_from_picks(pred_picks: List[int], true_picks: List[int], tol: int) -> Dict[str, Any]:
    n = len(pred_picks)
    if n == 0:
        return {
            "num_samples": 0,
            "tp_tolerance_samples": tol,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "mae": None,
            "rmse": None,
            "hit_rate": 0.0,
        }

    pred_arr = np.asarray(pred_picks, dtype=np.float32)
    true_arr = np.asarray(true_picks, dtype=np.float32)
    err = pred_arr - true_arr
    abs_err = np.abs(err)
    hit = (abs_err <= float(tol)).astype(np.int32)

    tp = int(hit.sum())
    fp = int(n - tp)
    fn = int(n - tp)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    mae = float(abs_err.mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    hit_rate = float(hit.mean())

    return {
        "num_samples": int(n),
        "tp_tolerance_samples": int(tol),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mae": mae,
        "rmse": rmse,
        "hit_rate": hit_rate,
    }


class PickerLoss(nn.Module):
    """
    冲 F1 专项损失：
    1) BCEWithLogitsLoss：学习整条概率曲线
    2) soft-argmax 回归损失：直接约束峰值位置靠近 true_pick
    """
    def __init__(self, pos_weight: float = 8.0, reg_weight: float = 0.30):
        super().__init__()
        self.register_buffer("pos_weight_tensor", torch.tensor([float(pos_weight)], dtype=torch.float32))
        self.reg_weight = float(reg_weight)

    def forward(self, logits: torch.Tensor, target_curve: torch.Tensor, true_pick: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        logits:      [B, T]
        target_curve:[B, T]
        true_pick:   [B]
        """
        pos_weight = self.pos_weight_tensor.to(logits.device)
        bce = F.binary_cross_entropy_with_logits(
            logits,
            target_curve,
            pos_weight=pos_weight
        )

        # soft-argmax 概率分布
        prob = torch.softmax(logits, dim=1)  # [B, T]
        t = torch.arange(logits.size(1), device=logits.device, dtype=torch.float32).unsqueeze(0)  # [1, T]
        pred_pick_soft = (prob * t).sum(dim=1)  # [B]

        reg = F.smooth_l1_loss(pred_pick_soft, true_pick.float(), reduction="mean")

        # 用长度归一化回归项，避免量级过大
        reg = reg / max(float(logits.size(1)), 1.0)

        total = bce + self.reg_weight * reg

        return total, {
            "bce": float(bce.detach().cpu().item()),
            "reg": float(reg.detach().cpu().item()),
            "total": float(total.detach().cpu().item()),
        }


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    criterion: PickerLoss,
    device: torch.device,
    train: bool,
    pick_tol: int,
):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_bce = 0.0
    total_reg = 0.0
    pred_picks_all: List[int] = []
    true_picks_all: List[int] = []

    for batch in loader:
        x, y, true_pick = batch
        x = x.to(device)                 # [B, C, T]
        y = y.to(device)                 # [B, T]
        true_pick = true_pick.to(device) # [B]

        with torch.set_grad_enabled(train):
            logits = model(x)            # [B, T]
            loss, loss_info = criterion(logits, y, true_pick)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_bce += loss_info["bce"] * bs
        total_reg += loss_info["reg"] * bs

        pred_pick = pick_from_logits(logits)
        pred_picks_all.extend(pred_pick.detach().cpu().tolist())
        true_picks_all.extend(true_pick.detach().cpu().tolist())

    avg_loss = total_loss / max(len(loader.dataset), 1)
    avg_bce = total_bce / max(len(loader.dataset), 1)
    avg_reg = total_reg / max(len(loader.dataset), 1)

    metrics = calc_metrics_from_picks(pred_picks_all, true_picks_all, tol=pick_tol)
    metrics["loss"] = float(avg_loss)
    metrics["bce_loss"] = float(avg_bce)
    metrics["reg_loss"] = float(avg_reg)
    return avg_loss, metrics


def evaluate_and_dump(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    outputs_dir: str,
    pick_tol: int,
):
    model.eval()
    rows = []

    with torch.no_grad():
        for batch in loader:
            x, y, true_pick, meta = batch
            x = x.to(device)

            logits = model(x)
            probs = torch.sigmoid(logits)
            pred_pick = torch.argmax(probs, dim=1)

            probs_np = probs.cpu().numpy()
            pred_pick_np = pred_pick.cpu().numpy()
            true_pick_np = true_pick.numpy()

            bs = x.size(0)
            for i in range(bs):
                rows.append({
                    "raw_key": meta["raw_key"][i],
                    "norm_key": meta["norm_key"][i],
                    "length": int(meta["length"][i]),
                    "true_pick": int(true_pick_np[i]),
                    "pred_pick": int(pred_pick_np[i]),
                    "abs_error": int(abs(int(pred_pick_np[i]) - int(true_pick_np[i]))),
                    "peak_prob": float(probs_np[i, pred_pick_np[i]]),
                })

    df = pd.DataFrame(rows)
    metrics = calc_metrics_from_picks(
        pred_picks=df["pred_pick"].tolist() if len(df) > 0 else [],
        true_picks=df["true_pick"].tolist() if len(df) > 0 else [],
        tol=pick_tol,
    )

    picks_csv = os.path.join(outputs_dir, "picks.csv")
    metrics_json = os.path.join(outputs_dir, "metrics.json")

    df.to_csv(picks_csv, index=False, encoding="utf-8-sig")
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return picks_csv, metrics_json, metrics


def main():
    print("=" * 90)
    print("开始运行 train_eval.py")
    print("=" * 90)

    config_path = os.path.join(PROJECT_ROOT, "config", "dataset_config.json")
    cfg = load_config(config_path)

    set_seed(int(cfg.get("seed", 42)))

    models_dir = os.path.join(PROJECT_ROOT, "models")
    outputs_dir = os.path.join(PROJECT_ROOT, "outputs")
    ensure_dir(models_dir)
    ensure_dir(outputs_dir)

    use_cuda = (cfg.get("device", "cuda") == "cuda") and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print(f"device      = {device}")
    if torch.cuda.is_available():
        print(f"GPU名称      = {torch.cuda.get_device_name(0)}")
        print(f"GPU数量      = {torch.cuda.device_count()}")

    print(f"h5_path     = {cfg['h5_path']}")
    print(f"train_list  = {cfg['train_list']}")
    print(f"test_list   = {cfg['test_list']}")
    print(f"quake_phase = {cfg['quake_phase_csv']}")

    full_train_ds = AEH5Dataset(
        h5_path=cfg["h5_path"],
        list_path=cfg["train_list"],
        quake_phase_csv=cfg["quake_phase_csv"],
        label_sigma=float(cfg.get("label_sigma", 24.0)),
        return_meta=False,
    )

    train_idx, val_idx = split_train_val_indices(
        n=len(full_train_ds),
        val_ratio=float(cfg.get("val_ratio", 0.1)),
        seed=int(cfg.get("seed", 42)),
    )

    train_ds = SubsetDataset(full_train_ds, train_idx)
    val_ds = SubsetDataset(full_train_ds, val_idx)

    test_ds = AEH5Dataset(
        h5_path=cfg["h5_path"],
        list_path=cfg["test_list"],
        quake_phase_csv=cfg["quake_phase_csv"],
        label_sigma=float(cfg.get("label_sigma", 24.0)),
        return_meta=False,
    )

    test_eval_ds = AEH5Dataset(
        h5_path=cfg["h5_path"],
        list_path=cfg["test_list"],
        quake_phase_csv=cfg["quake_phase_csv"],
        label_sigma=float(cfg.get("label_sigma", 24.0)),
        return_meta=True,
    )

    batch_size = int(cfg.get("batch_size", 32))
    num_workers = int(cfg.get("num_workers", 0))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(cfg.get("eval_batch_size", batch_size)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    test_eval_loader = DataLoader(
        test_eval_ds,
        batch_size=int(cfg.get("eval_batch_size", batch_size)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    x0, _, _ = full_train_ds[0]
    in_channels = int(x0.shape[0])

    model = SimplePPickerNet(
        in_channels=in_channels,
        base_channels=int(cfg.get("base_channels", 48)),
        dropout=float(cfg.get("dropout", 0.08)),
    ).to(device)

    criterion = PickerLoss(
        pos_weight=float(cfg.get("pos_weight", 8.0)),
        reg_weight=float(cfg.get("reg_weight", 0.30)),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 8e-4)),
        weight_decay=float(cfg.get("weight_decay", 5e-5)),
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(cfg.get("lr_factor", 0.5)),
        patience=int(cfg.get("lr_patience", 3)),
        min_lr=float(cfg.get("min_lr", 1e-6)),
    )

    epochs = int(cfg.get("epochs", 80))
    pick_tol = int(cfg.get("tp_tolerance_samples", 50))

    best_model_path = os.path.join(models_dir, "best_model.pth")
    history_path = os.path.join(outputs_dir, "train_history.json")

    best_val_f1 = -1.0
    best_epoch = -1
    history = []

    for epoch in range(1, epochs + 1):
        train_loss, train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            train=True,
            pick_tol=pick_tol,
        )

        with torch.no_grad():
            val_loss, val_metrics = run_one_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                criterion=criterion,
                device=device,
                train=False,
                pick_tol=pick_tol,
            )

        scheduler.step(val_metrics["f1"])
        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)

        print(
            f"[Epoch {epoch:03d}/{epochs:03d}] "
            f"train_loss={train_metrics['loss']:.6f} "
            f"train_bce={train_metrics['bce_loss']:.6f} "
            f"train_reg={train_metrics['reg_loss']:.6f} "
            f"train_f1={train_metrics['f1']:.4f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_bce={val_metrics['bce_loss']:.6f} "
            f"val_reg={val_metrics['reg_loss']:.6f} "
            f"val_f1={val_metrics['f1']:.4f} "
            f"val_mae={val_metrics['mae']:.2f} "
            f"lr={current_lr:.6e}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": cfg,
                    "in_channels": in_channels,
                    "best_val_f1": best_val_f1,
                },
                best_model_path,
            )
            print(f"[INFO] 已保存最佳模型 -> {best_model_path}")

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_val_f1": best_val_f1,
                "history": history,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 90)
    print("训练完成，开始加载最佳模型做测试集评估")
    print("=" * 90)

    ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        _, test_metrics = run_one_epoch(
            model=model,
            loader=test_loader,
            optimizer=None,
            criterion=criterion,
            device=device,
            train=False,
            pick_tol=pick_tol,
        )

    picks_csv, metrics_json, metrics = evaluate_and_dump(
        model=model,
        loader=test_eval_loader,
        device=device,
        outputs_dir=outputs_dir,
        pick_tol=pick_tol,
    )

    print("=" * 90)
    print(f"[DONE] best_epoch      = {best_epoch}")
    print(f"[DONE] best_val_f1     = {best_val_f1:.6f}")
    print(f"[DONE] test_f1         = {test_metrics['f1']:.6f}")
    print(f"[DONE] test_mae        = {test_metrics['mae']:.6f}")
    print(f"[DONE] best_model      = {best_model_path}")
    print(f"[DONE] train_history   = {history_path}")
    print(f"[DONE] picks.csv       = {picks_csv}")
    print(f"[DONE] metrics.json    = {metrics_json}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("=" * 90)


if __name__ == "__main__":
    main()