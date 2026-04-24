"""Evaluation metrics in the scaled and un-scaled (real-value) domain."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class RegressionMetrics:
    mse: float
    mae: float
    rmse: float


def regression_metrics(pred: torch.Tensor, true: torch.Tensor) -> RegressionMetrics:
    diff = (pred - true).float()
    mse = float(torch.mean(diff * diff).item())
    mae = float(torch.mean(torch.abs(diff)).item())
    return RegressionMetrics(mse=mse, mae=mae, rmse=float(np.sqrt(mse)))


def binary_classification_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.0) -> float:
    preds = (logits > threshold).float()
    return float((preds == labels).float().mean().item())


def _iter_batches(loader: DataLoader):
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            yield batch[0], batch[1]
        else:
            yield batch["x"], batch["y"]


@torch.no_grad()
def evaluate_regression(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype | None = None,
) -> RegressionMetrics:
    model.eval().to(device)
    autocast_ctx = torch.autocast(
        device_type=device.type, dtype=amp_dtype or torch.bfloat16, enabled=amp_enabled
    )
    sum_sq = 0.0
    sum_abs = 0.0
    n = 0
    for xb, yb in _iter_batches(loader):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        with autocast_ctx:
            pred = model(xb)
        diff = (pred.float() - yb.float()).detach()
        sum_sq += float((diff * diff).sum().item())
        sum_abs += float(diff.abs().sum().item())
        n += yb.numel()
    mse = sum_sq / max(n, 1)
    mae = sum_abs / max(n, 1)
    return RegressionMetrics(mse=mse, mae=mae, rmse=float(np.sqrt(mse)))


@torch.no_grad()
def evaluate_multi_output(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    regression_slice: slice,
    classification_slice: slice,
    *,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype | None = None,
) -> dict[str, float]:
    """Combined regression+classification evaluation for the multi-output LSTM."""
    model.eval().to(device)
    autocast_ctx = torch.autocast(
        device_type=device.type, dtype=amp_dtype or torch.bfloat16, enabled=amp_enabled
    )
    reg_sq = 0.0
    reg_abs = 0.0
    reg_n = 0
    cls_correct = 0
    cls_n = 0
    for xb, yb in _iter_batches(loader):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        with autocast_ctx:
            pred = model(xb)
        pred = pred.float()
        yb = yb.float()
        if regression_slice.stop > regression_slice.start:
            diff = pred[:, regression_slice] - yb[:, regression_slice]
            reg_sq += float((diff * diff).sum().item())
            reg_abs += float(diff.abs().sum().item())
            reg_n += diff.numel()
        if classification_slice.stop > classification_slice.start:
            cls_logits = pred[:, classification_slice]
            cls_true = yb[:, classification_slice]
            cls_correct += int(((cls_logits > 0) == (cls_true > 0.5)).sum().item())
            cls_n += cls_logits.numel()
    out = {}
    if reg_n:
        out["reg_mse"] = reg_sq / reg_n
        out["reg_mae"] = reg_abs / reg_n
        out["reg_rmse"] = float(np.sqrt(out["reg_mse"]))
    if cls_n:
        out["cls_accuracy"] = cls_correct / cls_n
    return out
