"""Stage 1 (Path B) — centralized 3-architecture sweep on ColO-RAN slice SLA.

Per ADR-001 D-20 / S1-W2: trains LSTM (`ForecasterV2`), Mamba (`MambaForecaster`,
in-tree pure PyTorch), and Spiking-SSM (`SpikingForecaster`) backbones with the
same encoder + classifier head, the same OOD split convention, and a per-arch
hyperparameter table that mirrors M5 except for arch-specific overrides:

* LSTM    : Adam lr=5e-4, weight_decay=0, dropout=0.1
* Mamba   : Adam lr=5e-4, weight_decay=0, dropout=0.1
* Spiking : Adam lr=1e-4 (surrogate-gradient stability), weight_decay=0, dropout=0.0
* Linear LR warm-up over the first 750 steps (1250 for Spiking).
* batch_size=64, grad_clip=1.0, mixed_precision="bf16", cudnn_deterministic=True.
* total_gradient_steps = 5000 (matches M5's federation-wide step budget).

Outputs go to ``artifacts/v6_arch_sweep/<arch>_s<seed>/``:
  summary.json, history.csv, best_state.pt (gitignored), energy.json.

Reuses, does not duplicate:
  ``training.centralized_v3._load_and_prepare`` for parquet load + target build,
  ``training.centralized_v3._metrics`` and ``_batched_predict``,
  ``data_v2.{sequences,split,encoders,targets_v2}`` for the data pipeline,
  ``evaluation.energy_metrics`` for FLOPs + sops.
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
from fl_oran.models.forecaster_v2 import ForecasterV2
from fl_oran.models.mamba_forecaster import MambaForecaster
from fl_oran.models.mamba3_forecaster import Mamba3Forecaster
from fl_oran.models.spiking_forecaster import SpikingForecaster
from fl_oran.models.xlstm_forecaster import xLSTMForecaster
from fl_oran.training.centralized_v3 import (
    V3Config,
    _load_and_prepare,
)
from fl_oran.utils import autocast_dtype, log_cuda_info, pick_device, seed_everything

log = get_logger(__name__)


# Per-ADR-D-20 hyperparameter table (no HPO inside Stage 1).
ARCH_REGISTRY = {
    "lstm": {
        "ctor": ForecasterV2,
        "lr": 5e-4,
        "warmup_steps": 750,
        "weight_decay": 0.0,
        "kwargs": {},  # ForecasterV2 defaults: dropout=0.1, lstm_hidden1=64, lstm_hidden2=32
    },
    "mamba": {
        "ctor": MambaForecaster,
        "lr": 5e-4,
        "warmup_steps": 750,
        "weight_decay": 0.0,
        "kwargs": {},  # tuned defaults: backbone_d_model=64, expand=1, n_blocks=2
    },
    "mamba_expand2": {
        # Audit ablation per ADR-001 §7 limitations (Stage 1 final round).
        # Tests whether the canonical Mamba-S6 design (expand=2) outperforms
        # the dense LSTM baseline once parameter parity is preserved by
        # shrinking d_model to 48.
        "ctor": MambaForecaster,
        "lr": 5e-4,
        "warmup_steps": 750,
        "weight_decay": 0.0,
        "kwargs": {"backbone_d_model": 48, "backbone_expand": 2, "n_blocks": 2},
    },
    "spiking": {
        "ctor": SpikingForecaster,
        "lr": 1e-4,
        "warmup_steps": 1250,
        "weight_decay": 0.0,
        "kwargs": {"t_inner": 1},  # tuned defaults: backbone_d_model=80, n_blocks=2, dropout=0
    },
    "spiking_expand2": {
        # Tier B.2 ablation: tests whether SSM-style channel expansion
        # helps the Spiking arm the way it would Mamba. d_model shrunk
        # to 56 to satisfy the ±10% parity constraint with LSTM 44k:
        # 56 × expand=2 = 112 d_inner channels, total ~43.6K params.
        "ctor": SpikingForecaster,
        "lr": 5e-4,
        "warmup_steps": 750,
        "weight_decay": 0.0,
        "kwargs": {"backbone_d_model": 56, "backbone_expand": 2, "t_inner": 1},
    },
    "xlstm": {
        # Path D extension: Beck et al. xLSTM-sLSTM (arXiv:2405.04517,
        # NeurIPS 2024). Single-head sLSTM (scalar memory) with
        # exponential input gate + normalizer + stabilizer state. Wraps
        # the same FeatureSchema encoder + fc/relu/head as ForecasterV2;
        # only the temporal trunk differs. Targets ~42K params at
        # hidden_size=48, n_layers=2.
        "ctor": xLSTMForecaster,
        "lr": 5e-4,
        "warmup_steps": 750,
        "weight_decay": 0.0,
        "kwargs": {},
    },
    "mamba3": {
        # Path D extension: Lahoti et al. Mamba-3 (arXiv:2603.15569,
        # Mar 2026). Adds two innovations on top of Mamba-2's selective
        # SSM:
        #
        #   1. Exponential-Trapezoidal discretization (§3.1, Proposition
        #      1) — adds a previous-step input contribution β_t · B_{t-1}
        #      · x_{t-1} with data-dependent trapezoidal mix λ_t.
        #   2. Complex-valued SSM via RoPE-style 2x2 rotation (§3.2,
        #      Proposition 2) — rotation+decay on paired (re, im) state
        #      dims, with data-dependent rotation angle θ_t.
        #
        # MIMO (Innovation 3) deferred — it's a hardware-efficiency
        # optimization for LLM-scale models and doesn't help at 40K
        # params. Default kwargs reuse Mamba's d_model=64, expand=1,
        # n_blocks=2; d_state=16 (8 complex pairs). Targets ~40K params.
        "ctor": Mamba3Forecaster,
        "lr": 5e-4,
        "warmup_steps": 750,
        "weight_decay": 0.0,
        "kwargs": {},
    },
}


def _linear_warmup_lr(step: int, base_lr: float, warmup_steps: int) -> float:
    if step >= warmup_steps:
        return base_lr
    # Ramp linearly from base_lr * 1e-3 to base_lr.
    floor = base_lr * 1e-3
    frac = (step + 1) / warmup_steps
    return floor + (base_lr - floor) * frac


def _batched_logits(model: nn.Module, X_cat: torch.Tensor, X_cont: torch.Tensor,
                    device: torch.device, batch_size: int = 8192) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(X_cat), batch_size):
            cb = X_cat[i:i + batch_size].to(device, non_blocking=True)
            ob = X_cont[i:i + batch_size].to(device, non_blocking=True)
            out.append(model(cb, ob).cpu().float().numpy())
    return np.concatenate(out, axis=0)


def _classification_metrics(y_true: np.ndarray, logits: np.ndarray) -> dict:
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


def run_cell(arch: str, seed: int, cfg: V3Config, df, schema, device: torch.device,
             output_dir: Path, val_every: int = 100) -> dict:
    seed_everything(seed)
    if cfg.mixed_precision and device.type == "cuda":
        # Match M5 perf settings; cudnn deterministic for reproducibility.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    arch_cfg = ARCH_REGISTRY[arch]
    feat_cols = schema.categorical + schema.continuous
    split = ood_split_by_tr(df, cfg.train_tr, cfg.val_tr, cfg.test_tr)
    X_tr, Y_tr = build_run_sequences(split.train, feat_cols, ["y_sla_next"], seq_len=cfg.seq_len)
    X_va, Y_va = build_run_sequences(split.val, feat_cols, ["y_sla_next"], seq_len=cfg.seq_len)
    X_te, Y_te = build_run_sequences(split.test, feat_cols, ["y_sla_next"], seq_len=cfg.seq_len)
    log.info("[%s s=%d] sequences train=%d val=%d test=%d",
             arch, seed, len(X_tr), len(X_va), len(X_te))

    scaler = fit_continuous_scaler({0: X_tr}, schema)

    def _to_tensors(X: np.ndarray, Y: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    model = arch_cfg["ctor"](schema=schema, task="classification",
                              seq_len=cfg.seq_len, **arch_cfg["kwargs"]).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("[%s s=%d] params=%d  base_lr=%g  warmup_steps=%d",
             arch, seed, n_params, arch_cfg["lr"], arch_cfg["warmup_steps"])

    opt = torch.optim.Adam(
        model.parameters(), lr=arch_cfg["lr"], weight_decay=arch_cfg["weight_decay"]
    )
    amp_enabled, amp_dtype = autocast_dtype(cfg.mixed_precision)

    history: list[dict] = []
    best_val_auc = 0.0
    best_state: dict | None = None
    n_steps = cfg.total_gradient_steps

    perm = torch.randperm(len(tr_cat))
    perm_idx = 0
    for step in range(1, n_steps + 1):
        if perm_idx + cfg.batch_size > len(perm):
            perm = torch.randperm(len(tr_cat))
            perm_idx = 0
        idx = perm[perm_idx:perm_idx + cfg.batch_size]
        perm_idx += cfg.batch_size

        cb = tr_cat[idx].to(device, non_blocking=True)
        ob = tr_cont[idx].to(device, non_blocking=True)
        yb = tr_y[idx].to(device, non_blocking=True)

        lr_now = _linear_warmup_lr(step, arch_cfg["lr"], arch_cfg["warmup_steps"])
        for g in opt.param_groups:
            g["lr"] = lr_now

        model.train()
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,
                            dtype=amp_dtype or torch.bfloat16,
                            enabled=amp_enabled):
            logits = model(cb, ob)
            loss = bce(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if step % val_every == 0 or step == n_steps:
            val_logits = _batched_logits(model, va_cat, va_cont, device)
            val_y_int = va_y[:, 0].numpy().astype(int)
            val_auc = (
                roc_auc_score(val_y_int, val_logits[:, 0])
                if len(np.unique(val_y_int)) == 2
                else 0.5
            )
            history.append({
                "step": step,
                "train_loss": float(loss.item()),
                "val_auc": float(val_auc),
                "lr": float(lr_now),
            })
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # Restore best by val AUC and compute test metrics.
    if best_state is not None:
        model.load_state_dict(best_state)
    test_logits = _batched_logits(model, te_cat, te_cont, device)
    test_metrics = _classification_metrics(te_y[:, 0].numpy().astype(int), test_logits[:, 0])

    # Energy estimate on a representative test slice (fixed 64 rows for stability).
    energy_slice = min(64, len(te_cat))
    energy = estimate_energy_pJ_per_inference(
        model.cpu(),
        te_cat[:energy_slice],
        te_cont[:energy_slice],
    )
    model.to(device)

    summary = {
        "arch": arch,
        "seed": seed,
        "params_count": int(n_params),
        "best_val_auc": float(best_val_auc),
        "test_auc": float(test_metrics.get("auc", 0.0)),
        "test_f1": float(test_metrics["f1"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "positive_rate_test": float(test_metrics["positive_rate_true"]),
        "n_steps": int(n_steps),
        "config": cfg.to_dict(),
        "arch_hparams": {
            "lr": arch_cfg["lr"],
            "warmup_steps": arch_cfg["warmup_steps"],
            "weight_decay": arch_cfg["weight_decay"],
        },
    }

    suffix = ARCH_REGISTRY[arch].get("output_suffix", "")
    cell_dir = output_dir / f"{arch}_s{seed}{suffix}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    (cell_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (cell_dir / "history.csv").write_text(
        "step,train_loss,val_auc,lr\n"
        + "\n".join(f"{h['step']},{h['train_loss']},{h['val_auc']},{h['lr']}" for h in history)
        + "\n"
    )
    (cell_dir / "energy.json").write_text(
        json.dumps({k: float(v) for k, v in energy.items()}, indent=2)
    )
    if best_state is not None:
        torch.save(best_state, cell_dir / "best_state.pt")
    log.info("[%s s=%d] DONE  test_auc=%.4f  test_f1=%.4f  energy_pJ=%.2e",
             arch, seed, summary["test_auc"], summary["test_f1"],
             energy["total_energy_pJ"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 centralized 3-arch sweep")
    parser.add_argument("--arch", type=str, required=True,
                        help="comma-separated archs: lstm,mamba,spiking")
    parser.add_argument("--seeds", type=str, required=True,
                        help="comma-separated seeds")
    parser.add_argument("--total-steps", type=int, default=5000,
                        help="gradient steps per cell (default 5000 = M5 federation budget)")
    parser.add_argument("--output-dir", type=str, default="artifacts/v6_arch_sweep")
    parser.add_argument("--sample-ratio", type=float, default=1.0)
    parser.add_argument("--unified-parquet", type=str,
                        default="data/coloran_raw_unified.parquet")
    parser.add_argument("--val-every", type=int, default=100)
    parser.add_argument("--spiking-t-inner", type=int, default=1,
                        help="LIF integrations per sequence position for SpikingSSMBlock (D-21 recovery: try 5)")
    parser.add_argument("--output-suffix", type=str, default="",
                        help="optional suffix appended to per-cell directory name to keep recovery runs separate")
    parser.add_argument("--spiking-lr", type=float, default=None,
                        help="override the SpikingForecaster Adam lr (default 1e-4 was post-hoc audited as undertrained; 5e-4 matches LSTM/Mamba)")
    parser.add_argument("--spiking-warmup-steps", type=int, default=None,
                        help="override SpikingForecaster linear-warmup step count")
    parser.add_argument("--spiking-decode-mode", type=str, default=None,
                        choices=("majority", "sum"),
                        help="Spiking sub-step spike aggregation; default majority is non-differentiable for t_inner > 1, 'sum' fixes that")
    parser.add_argument("--spiking-lif-threshold", type=float, default=None,
                        help="Override LIF firing threshold (D-21 recovery pass); default 1.0")
    parser.add_argument("--spiking-lif-beta", type=float, default=None,
                        help="Override LIF leak beta (D-21 recovery pass); default 0.9")
    parser.add_argument("--early-stop-patience", type=int, default=0,
                        help="0 = no early stopping (run full --total-steps); >0 = stop after this many val_every checks without val_auc improvement")
    args = parser.parse_args()
    if args.spiking_t_inner != 1:
        ARCH_REGISTRY["spiking"]["kwargs"]["t_inner"] = args.spiking_t_inner
    if args.spiking_decode_mode is not None:
        ARCH_REGISTRY["spiking"]["kwargs"]["decode_mode"] = args.spiking_decode_mode
    if args.spiking_lif_threshold is not None:
        ARCH_REGISTRY["spiking"]["kwargs"]["lif_threshold"] = args.spiking_lif_threshold
    if args.spiking_lif_beta is not None:
        ARCH_REGISTRY["spiking"]["kwargs"]["lif_beta"] = args.spiking_lif_beta
    if args.spiking_lr is not None:
        ARCH_REGISTRY["spiking"]["lr"] = args.spiking_lr
    if args.spiking_warmup_steps is not None:
        ARCH_REGISTRY["spiking"]["warmup_steps"] = args.spiking_warmup_steps
    if args.output_suffix:
        for arch_cfg in ARCH_REGISTRY.values():
            arch_cfg["output_suffix"] = args.output_suffix

    archs = [a.strip() for a in args.arch.split(",") if a.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    for arch in archs:
        if arch not in ARCH_REGISTRY:
            raise ValueError(f"unknown arch: {arch!r} (known: {sorted(ARCH_REGISTRY)})")

    cfg = V3Config(
        unified_parquet=Path(args.unified_parquet),
        sample_ratio=args.sample_ratio,
        total_gradient_steps=args.total_steps,
        batch_size=64,
        grad_clip=1.0,
        seq_len=5,
        threshold=0.10,
        mixed_precision="bf16",
    )
    df, schema = _load_and_prepare(cfg)

    device = pick_device(cfg.device)
    log_cuda_info(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overall_t0 = time.time()
    for seed in seeds:
        for arch in archs:
            cell_t0 = time.time()
            run_cell(arch, seed, cfg, df, schema, device, output_dir, val_every=args.val_every)
            log.info("[cell %s s=%d] wall_time=%.1fs", arch, seed, time.time() - cell_t0)
    log.info("[sweep] %d cells in %.1fs", len(archs) * len(seeds), time.time() - overall_t0)


if __name__ == "__main__":
    main()
