"""Centralized baseline + FL v3 trainer for the `ul_bler_{t+1} > 0.1` classification task.

Why both in one file: the Centralized run is the SCIENTIFIC GATE. If the LSTM
can't beat persistence when trained centrally, FL definitely can't. So we only
move to FL experiments after Centralized proves learnable signal exists.
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from ..data_v2.encoders import (
    ContinuousScaler,
    FeatureSchema,
    apply_continuous_scaler,
    federated_fit_scaler,
    fit_continuous_scaler,
)
from ..data_v2.sequences import build_run_sequences
from ..data_v2.split import ood_split_by_tr
from ..data_v2.targets_v2 import add_classification_target
from ..federated import weighted_average_state_dicts
from ..federated.client_v2 import train_one_client_capped
from ..logging_utils import get_logger
from ..models.forecaster_v2 import ForecasterV2
from ..utils import autocast_dtype, log_cuda_info, pick_device, seed_everything

log = get_logger(__name__)


V3_CATEGORICAL = ["bs_id", "slice_id", "sched", "tr"]
V3_CAT_SIZES = {"bs_id": 8, "slice_id": 4, "sched": 4, "tr": 29}
V3_CONTINUOUS = [
    "num_ues", "slice_prb",
    "sum_requested_prbs", "sum_granted_prbs",
    "tx_brate_dl_Mbps", "rx_brate_ul_Mbps",
    "tx_pkts_dl", "rx_pkts_ul",
    "dl_buffer_bytes", "ul_buffer_bytes",
    "dl_bler", "ul_bler",
    "dl_mcs", "ul_mcs", "dl_cqi", "ul_sinr", "ul_rssi",
]


@dataclass
class V3Config:
    name: str = "v3_centralized"
    unified_parquet: Path = field(default_factory=lambda: Path("data/coloran_raw_unified.parquet"))
    sample_ratio: float = 0.2   # subsample for speed; 0.2 = 20% of 18M rows
    seq_len: int = 5
    threshold: float = 0.10
    train_tr: list[int] = field(default_factory=lambda: list(range(22)))
    val_tr: list[int] = field(default_factory=lambda: [22, 23, 24])
    test_tr: list[int] = field(default_factory=lambda: [25, 26, 27])

    # Training
    batch_size: int = 256
    lr: float = 1e-3
    grad_clip: float = 1.0
    seed: int = 42

    # Shared gradient-step budget — both centralized and FL target this total.
    total_gradient_steps: int = 50_000

    # Centralized-specific (epochs is auto-computed unless overridden)
    centralized_epochs: int | None = None  # None → auto from total_gradient_steps

    # FL-specific
    num_rounds: int = 20
    clients_per_round: int = 5
    max_steps_per_round: int = 500
    lr_warmup_rounds: int = 2

    # Runtime
    device: str = "auto"
    mixed_precision: str = "bf16"
    output_dir: Path = field(default_factory=lambda: Path("artifacts"))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["unified_parquet"] = str(self.unified_parquet)
        d["output_dir"] = str(self.output_dir)
        return d


# ----------------------------------------------------------------------------
# Shared data pipeline
# ----------------------------------------------------------------------------

def _load_and_prepare(cfg: V3Config) -> tuple[pd.DataFrame, FeatureSchema]:
    if not cfg.unified_parquet.exists():
        raise FileNotFoundError(cfg.unified_parquet)
    log.info("loading %s", cfg.unified_parquet)
    df = pd.read_parquet(cfg.unified_parquet)
    if cfg.sample_ratio < 1.0:
        n = int(len(df) * cfg.sample_ratio)
        df = df.sample(n=n, random_state=cfg.seed).sort_index().reset_index(drop=True)
        log.info("sampled %s / %s rows (ratio=%.2f)",
                 f"{len(df):,}", f"{n:,}", cfg.sample_ratio)

    # Target: binary ul_bler_{t+1} > threshold
    df = add_classification_target(
        df, column="ul_bler", threshold=cfg.threshold,
        target_name="y_sla_next",
    )

    schema = FeatureSchema(
        categorical=V3_CATEGORICAL,
        categorical_sizes=V3_CAT_SIZES,
        continuous=V3_CONTINUOUS,
    )
    return df, schema


def _build_client_splits(df: pd.DataFrame, feat_cols: list[str]):
    """Per-(bs_id) sequence shards, for FL."""
    shards: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for bs, g in df.groupby("bs_id", observed=True):
        X, Y = build_run_sequences(g, feat_cols, ["y_sla_next"], seq_len=5)
        if len(X) > 0:
            shards[int(bs)] = (X, Y)
    return shards


def _metrics(y_true: np.ndarray, logits: np.ndarray) -> dict:
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "positive_rate_pred": float(preds.mean()),
        "positive_rate_true": float(y_true.mean()),
    }
    if len(np.unique(y_true)) == 2:
        out["auc"] = float(roc_auc_score(y_true, probs))
    return out


def _batched_predict(model: nn.Module, X_cat: torch.Tensor, X_cont: torch.Tensor,
                     device: torch.device, batch_size: int = 8192) -> np.ndarray:
    model.eval()
    parts = []
    with torch.no_grad():
        for i in range(0, len(X_cat), batch_size):
            cb = X_cat[i:i + batch_size].to(device, non_blocking=True)
            ob = X_cont[i:i + batch_size].to(device, non_blocking=True)
            parts.append(model(cb, ob).cpu().numpy())
    model.train()
    return np.concatenate(parts, axis=0)


# ----------------------------------------------------------------------------
# Centralized training
# ----------------------------------------------------------------------------

def run_centralized(cfg: V3Config) -> dict:
    seed_everything(cfg.seed)
    device = pick_device(cfg.device)
    log_cuda_info(device)
    amp_enabled, amp_dtype = autocast_dtype(cfg.mixed_precision)

    df, schema = _load_and_prepare(cfg)
    feat_cols = schema.categorical + schema.continuous

    split = ood_split_by_tr(df, cfg.train_tr, cfg.val_tr, cfg.test_tr)
    X_tr, Y_tr = build_run_sequences(split.train, feat_cols, ["y_sla_next"], seq_len=cfg.seq_len)
    X_va, Y_va = build_run_sequences(split.val, feat_cols, ["y_sla_next"], seq_len=cfg.seq_len)
    X_te, Y_te = build_run_sequences(split.test, feat_cols, ["y_sla_next"], seq_len=cfg.seq_len)
    log.info("centralized sequences: train=%s val=%s test=%s",
             f"{len(X_tr):,}", f"{len(X_va):,}", f"{len(X_te):,}")

    # Baselines on test set (in classification terms).
    y_true_te = Y_te[:, 0].astype(int)
    # Persistence / "current" classifier: predict current ul_bler > threshold.
    cur_ul_bler = X_te[:, -1, feat_cols.index("ul_bler")]
    persist_logits = (cur_ul_bler - cfg.threshold) * 100   # fake large logit from sign
    persist_metrics = _metrics(y_true_te, persist_logits)
    majority_metrics = {"accuracy": max(y_true_te.mean(), 1 - y_true_te.mean()),
                         "positive_rate_true": float(y_true_te.mean())}
    log.info("persistence classifier: %s", persist_metrics)
    log.info("majority baseline:      %s", majority_metrics)

    # Fit continuous scaler on ALL train data (centralized = no client boundaries).
    # Here we use the pooled fit since there's no privacy concern — but the
    # function is identical in math to the federated version.
    scaler = fit_continuous_scaler({0: X_tr}, schema)

    # Auto-compute epochs to hit total_gradient_steps budget.
    n_batches_per_epoch = max(1, len(X_tr) // cfg.batch_size)
    epochs = cfg.centralized_epochs
    if epochs is None:
        epochs = max(1, cfg.total_gradient_steps // n_batches_per_epoch)
    log.info("centralized: %d epochs × %d batches/epoch = %d grad steps (target %d)",
             epochs, n_batches_per_epoch, epochs * n_batches_per_epoch, cfg.total_gradient_steps)

    def to_tensors(X: np.ndarray, Y: np.ndarray):
        cat, cont = apply_continuous_scaler(X, schema, scaler)
        return (
            torch.from_numpy(cat),       # int64
            torch.from_numpy(cont),      # float32
            torch.from_numpy(Y),         # float32
        )

    tr_cat, tr_cont, tr_y = to_tensors(X_tr, Y_tr)
    va_cat, va_cont, va_y = to_tensors(X_va, Y_va)
    te_cat, te_cont, te_y = to_tensors(X_te, Y_te)

    model = ForecasterV2(schema=schema, task="classification", seq_len=cfg.seq_len).to(device)
    log.info("model: %s params", sum(p.numel() for p in model.parameters()))

    # Class imbalance weight.
    pos_rate = y_true_te.mean()
    pos_weight = torch.tensor([max((1 - pos_rate) / pos_rate, 1.0)], device=device)
    log.info("pos_weight for BCE: %.3f", pos_weight.item())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_val_auc, best_state = 0.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        perm = torch.randperm(len(tr_cat))
        total_loss = 0.0
        steps = 0
        for i in range(0, len(perm), cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            cb = tr_cat[idx].to(device)
            ob = tr_cont[idx].to(device)
            yb = tr_y[idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype or torch.bfloat16,
                                 enabled=amp_enabled):
                logits = model(cb, ob)
                loss = loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            total_loss += loss.item()
            steps += 1
        # Val metrics.
        val_logits = _batched_predict(model, va_cat, va_cont, device)
        val_m = _metrics(va_y[:, 0].numpy().astype(int), val_logits[:, 0])
        log.info("epoch %d  train_loss=%.4f  val_auc=%.4f  val_acc=%.4f  dt=%.1fs",
                 epoch, total_loss / max(steps, 1), val_m.get("auc", 0), val_m["accuracy"],
                 time.time() - t0)
        if val_m.get("auc", 0) > best_val_auc:
            best_val_auc = val_m["auc"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    test_logits = _batched_predict(model, te_cat, te_cont, device)
    test_metrics = _metrics(y_true_te, test_logits[:, 0])

    result = {
        "config": cfg.to_dict(),
        "majority_baseline": majority_metrics,
        "persistence_classifier": persist_metrics,
        "centralized_lstm": test_metrics,
    }

    out_dir = Path(cfg.output_dir)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs" / f"{cfg.name}_summary.json").write_text(
        json.dumps(result, indent=2, default=str)
    )
    if best_state is not None:
        torch.save(best_state, out_dir / "models" / f"{cfg.name}_best.pt")

    log.info("=" * 70)
    log.info("CENTRALIZED GATE CHECK (target = ul_bler_{t+1} > %.2f)", cfg.threshold)
    log.info("  Majority baseline   acc=%.4f", majority_metrics["accuracy"])
    log.info("  Persistence classifier acc=%.4f  AUC=%.4f",
             persist_metrics["accuracy"], persist_metrics.get("auc", 0))
    log.info("  Centralized LSTM    acc=%.4f  AUC=%.4f  F1=%.4f",
             test_metrics["accuracy"], test_metrics.get("auc", 0), test_metrics["f1"])
    log.info("=" * 70)
    return result
