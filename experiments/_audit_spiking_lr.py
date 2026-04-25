"""Audit-only: train SpikingForecaster with override hyperparameters.

Bypasses the preregistered ARCH_REGISTRY in run_v6_arch_sweep.py to test
whether the Stage 1 result is sensitive to the learning-rate choice
(lr=1e-4 with 1250-step warmup) or to the gradient-step budget (5000).
The output goes to artifacts/v6_arch_sweep_audit/ to avoid polluting
the main S1 deliverable.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from fl_oran.data_v2.encoders import apply_continuous_scaler, fit_continuous_scaler
from fl_oran.data_v2.sequences import build_run_sequences
from fl_oran.data_v2.split import ood_split_by_tr
from fl_oran.evaluation.energy_metrics import estimate_energy_pJ_per_inference
from fl_oran.logging_utils import get_logger
from fl_oran.models.spiking_forecaster import SpikingForecaster
from fl_oran.training.centralized_v3 import V3Config, _load_and_prepare
from fl_oran.utils import autocast_dtype, log_cuda_info, pick_device, seed_everything

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=1250)
    parser.add_argument("--total-steps", type=int, default=5000)
    parser.add_argument("--label", type=str, required=True,
                        help="output dir suffix, e.g. lr5e4_w750 or lr1e4_25k")
    args = parser.parse_args()

    cfg = V3Config(
        unified_parquet=Path("data/coloran_raw_unified.parquet"),
        sample_ratio=1.0, total_gradient_steps=args.total_steps,
        batch_size=64, grad_clip=1.0, seq_len=5, threshold=0.10,
        mixed_precision="bf16",
    )
    df, schema = _load_and_prepare(cfg)
    device = pick_device(cfg.device)
    log_cuda_info(device)

    seed_everything(args.seed)
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    feat_cols = schema.categorical + schema.continuous
    split = ood_split_by_tr(df, cfg.train_tr, cfg.val_tr, cfg.test_tr)
    X_tr, Y_tr = build_run_sequences(split.train, feat_cols, ["y_sla_next"], seq_len=cfg.seq_len)
    X_va, Y_va = build_run_sequences(split.val, feat_cols, ["y_sla_next"], seq_len=cfg.seq_len)
    X_te, Y_te = build_run_sequences(split.test, feat_cols, ["y_sla_next"], seq_len=cfg.seq_len)

    scaler = fit_continuous_scaler({0: X_tr}, schema)

    def _to_tensors(X, Y):
        cat, cont = apply_continuous_scaler(X, schema, scaler)
        return torch.from_numpy(cat), torch.from_numpy(cont), torch.from_numpy(Y).float()

    tr_cat, tr_cont, tr_y = _to_tensors(X_tr, Y_tr)
    va_cat, va_cont, va_y = _to_tensors(X_va, Y_va)
    te_cat, te_cont, te_y = _to_tensors(X_te, Y_te)

    pos_rate = float(tr_y[:, 0].mean())
    pos_weight = torch.tensor(
        [max((1.0 - pos_rate) / max(pos_rate, 1e-6), 1.0)], device=device
    )
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = SpikingForecaster(schema=schema, task="classification", seq_len=cfg.seq_len).to(device)
    log.info("[audit %s] params=%d  lr=%g  warmup=%d  steps=%d",
             args.label, sum(p.numel() for p in model.parameters()),
             args.lr, args.warmup_steps, args.total_steps)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    amp_enabled, amp_dtype = autocast_dtype(cfg.mixed_precision)

    best_val_auc, best_state = 0.0, None
    history = []
    perm = torch.randperm(len(tr_cat))
    perm_idx = 0
    floor = args.lr * 1e-3
    for step in range(1, args.total_steps + 1):
        if perm_idx + cfg.batch_size > len(perm):
            perm = torch.randperm(len(tr_cat))
            perm_idx = 0
        idx = perm[perm_idx:perm_idx + cfg.batch_size]
        perm_idx += cfg.batch_size
        cb, ob, yb = tr_cat[idx].to(device), tr_cont[idx].to(device), tr_y[idx].to(device)

        if step < args.warmup_steps:
            lr_now = floor + (args.lr - floor) * (step + 1) / args.warmup_steps
        else:
            lr_now = args.lr
        for g in opt.param_groups:
            g["lr"] = lr_now

        model.train()
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype or torch.bfloat16, enabled=amp_enabled):
            logits = model(cb, ob)
            loss = bce(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if step % 250 == 0 or step == args.total_steps:
            model.eval()
            val_logits = []
            with torch.no_grad():
                for i in range(0, len(va_cat), 8192):
                    val_logits.append(
                        model(va_cat[i:i+8192].to(device), va_cont[i:i+8192].to(device)).cpu().float().numpy()
                    )
            val_logits = np.concatenate(val_logits, axis=0)
            val_y_int = va_y[:, 0].numpy().astype(int)
            val_auc = roc_auc_score(val_y_int, val_logits[:, 0])
            history.append({"step": step, "train_loss": float(loss.item()),
                            "val_auc": float(val_auc), "lr": float(lr_now)})
            log.info("step=%5d  train_loss=%.4f  val_auc=%.4f", step, loss.item(), val_auc)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits = []
        for i in range(0, len(te_cat), 8192):
            test_logits.append(
                model(te_cat[i:i+8192].to(device), te_cont[i:i+8192].to(device)).cpu().float().numpy()
            )
    test_logits = np.concatenate(test_logits, axis=0)
    te_y_int = te_y[:, 0].numpy().astype(int)
    test_auc = roc_auc_score(te_y_int, test_logits[:, 0])
    test_pred = (test_logits[:, 0] > 0).astype(int)
    test_f1 = f1_score(te_y_int, test_pred, zero_division=0)
    test_acc = accuracy_score(te_y_int, test_pred)

    out_dir = Path("artifacts/v6_arch_sweep_audit") / f"spiking_s{args.seed}_{args.label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "label": args.label,
        "seed": args.seed,
        "lr": args.lr,
        "warmup_steps": args.warmup_steps,
        "total_steps": args.total_steps,
        "best_val_auc": float(best_val_auc),
        "test_auc": float(test_auc),
        "test_f1": float(test_f1),
        "test_accuracy": float(test_acc),
    }
    out_dir.joinpath("summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] {args.label}  test_auc={test_auc:.4f}  test_f1={test_f1:.4f}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
