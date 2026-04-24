"""FL trainer for the v2 forecasting task built from raw ColO-RAN data.

Differences from ``trainer.py``:
- Loads ``data/coloran_raw_unified.parquet`` (produced by ``data_raw.build_unified_parquet``)
- Uses ``data_v2.engineer_features`` / ``ood_split_by_tr`` / ``build_run_sequences``
- Client partition = bs_id (same as before); each client has its own (tr∈TRAIN) rows
- Evaluates on the held-out OOD tr-set, plus computes persistence + GBM baselines
  so we can tell whether the FL model is actually adding value
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

from ..baselines import PersistenceBaseline, gbm_baseline
from ..data_v2 import (
    CLASSIFICATION_TARGETS,
    CLEAN_FEATURES,
    REGRESSION_TARGETS,
    build_run_sequences,
    engineer_features,
    ood_split_by_tr,
)
from ..federated import (
    PrivacyAccountant,
    train_one_client_cuda_graph,
    train_one_client_gpu_resident,
    weighted_average_state_dicts,
)
from ..logging_utils import get_logger
from ..models import LSTMMultiOutput, MultiOutputSpec
from ..utils import autocast_dtype, log_cuda_info, pick_device, seed_everything

log = get_logger(__name__)


@dataclass
class V2Config:
    name: str = "v107_1_v2"
    unified_parquet: Path = field(default_factory=lambda: Path("data/coloran_raw_unified.parquet"))
    seq_len: int = 5
    train_tr: list[int] = field(default_factory=lambda: list(range(22)))
    val_tr: list[int] = field(default_factory=lambda: [22, 23, 24])
    test_tr: list[int] = field(default_factory=lambda: [25, 26, 27])

    num_total_clients: int = 7
    num_rounds: int = 30
    clients_per_round: int = 5
    local_epochs: int = 2
    client_lr: float = 5e-4
    batch_size: int = 64
    random_state: int = 42
    early_stopping_patience: int = 0   # full run by default

    device: str = "auto"
    mixed_precision: str = "bf16"
    output_dir: Path = field(default_factory=lambda: Path("artifacts"))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["unified_parquet"] = str(self.unified_parquet)
        d["output_dir"] = str(self.output_dir)
        return d


@dataclass
class V2RoundResult:
    round: int
    train_loss: float
    val_loss: float
    val_cls_acc: float | None = None
    duration_s: float = 0.0


# ----------------------------------------------------------------------------
# Pipeline helpers
# ----------------------------------------------------------------------------

def _build_client_shards(
    df_split: pd.DataFrame,
    feat_cols: list[str],
    tgt_cols: list[str],
    seq_len: int,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Per-BS numpy sequence arrays."""
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for bs, g in df_split.groupby("bs_id", observed=True):
        X, Y = build_run_sequences(g, feat_cols, tgt_cols, seq_len=seq_len)
        if len(X) > 0:
            out[int(bs)] = (X, Y)
    return out


def _per_client_standardise(
    train_shards: dict[int, tuple[np.ndarray, np.ndarray]],
    val_X: np.ndarray,
    val_Y: np.ndarray,
    test_X: np.ndarray,
    test_Y: np.ndarray,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Feature z-score per client, target scale globally.

    Returns scaled train_shards, val_X, val_Y, test_X, test_Y, and a
    bundle of scaler stats for inverse-transform later.
    """
    # Pool all training data for global target scaling (regression target).
    all_Y = np.concatenate([y for _, y in train_shards.values()])
    reg_mean = float(all_Y[:, 0].mean())
    reg_std = float(all_Y[:, 0].std() + 1e-8)

    # Per-client feature means/stds over the last-step (stable reference).
    client_stats: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    out_train: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for cid, (X, Y) in train_shards.items():
        last_step = X[:, -1, :]
        mean = last_step.mean(axis=0).astype(np.float32)
        std = (last_step.std(axis=0) + 1e-6).astype(np.float32)
        client_stats[cid] = (mean, std)
        Xs = ((X - mean) / std).astype(np.float32)
        Ys = Y.copy()
        Ys[:, 0] = (Ys[:, 0] - reg_mean) / reg_std
        out_train[cid] = (Xs, Ys)

    # For val/test, use the *mean of per-client* means — rough global scaler.
    means = np.mean([m for m, _ in client_stats.values()], axis=0).astype(np.float32)
    stds = np.mean([s for _, s in client_stats.values()], axis=0).astype(np.float32)
    vX = ((val_X - means) / stds).astype(np.float32)
    tX = ((test_X - means) / stds).astype(np.float32)
    vY = val_Y.copy(); vY[:, 0] = (vY[:, 0] - reg_mean) / reg_std
    tY = test_Y.copy(); tY[:, 0] = (tY[:, 0] - reg_mean) / reg_std

    bundle = {
        "reg_mean": reg_mean,
        "reg_std": reg_std,
        "feat_mean": means.tolist(),
        "feat_std": stds.tolist(),
    }
    return out_train, vX, vY, tX, tY, bundle


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def run_v2_experiment(cfg: V2Config) -> dict:
    """Full v2 pipeline: load → engineer → split → baselines → FL LSTM → report."""
    seed_everything(cfg.random_state)
    device = pick_device(cfg.device)
    log_cuda_info(device)
    amp_enabled, amp_dtype = autocast_dtype(cfg.mixed_precision)

    out_dir = Path(cfg.output_dir)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    # -- 1) Load raw-derived unified parquet + engineer features ---------------
    if not cfg.unified_parquet.exists():
        raise FileNotFoundError(
            f"{cfg.unified_parquet} not found. "
            f"Run `python -m fl_oran.data_raw.cli build --raw-root raw/colosseum-oran-coloran-dataset-master "
            f"--out-path data/coloran_raw_unified.parquet` first."
        )
    log.info("loading unified parquet %s", cfg.unified_parquet)
    df = pd.read_parquet(cfg.unified_parquet)
    log.info("raw shape: %s", df.shape)
    df = engineer_features(df)

    # -- 2) OOD split by training_config --------------------------------------
    split = ood_split_by_tr(df, cfg.train_tr, cfg.val_tr, cfg.test_tr)

    feat_cols = [c for c in CLEAN_FEATURES if c in split.train.columns]
    reg_tgt = REGRESSION_TARGETS[0]
    log.info("features (%d): %s", len(feat_cols), feat_cols)

    # -- 3) Build sequence shards ---------------------------------------------
    train_shards_raw = _build_client_shards(split.train, feat_cols, REGRESSION_TARGETS, cfg.seq_len)
    val_X, val_Y = build_run_sequences(split.val, feat_cols, REGRESSION_TARGETS, cfg.seq_len)
    test_X, test_Y = build_run_sequences(split.test, feat_cols, REGRESSION_TARGETS, cfg.seq_len)
    log.info("clients: %d  val seq: %d  test seq: %d",
             len(train_shards_raw), len(val_X), len(test_X))

    # -- 4) Baselines (on TEST set, in original scale) ------------------------
    baselines: dict = {}
    # Persistence: ŷ_{t+1} = y_t  (y_t = last-step tx_brate_dl_Mbps from X window)
    tx_idx = feat_cols.index("tx_brate_dl_Mbps")
    persist_pred = test_X[:, -1, tx_idx]
    baselines["persistence"] = PersistenceBaseline.evaluate(
        persist_pred.astype(np.float64), test_Y[:, 0].astype(np.float64)
    )
    log.info("persistence baseline: %s", baselines["persistence"])

    # GBM: train on a random subsample (full 14M would take hours).
    X_gbm_full = np.concatenate([x for x, _ in train_shards_raw.values()])
    Y_gbm_full = np.concatenate([y for _, y in train_shards_raw.values()])
    n_gbm = min(200_000, len(X_gbm_full))
    rng = np.random.default_rng(cfg.random_state)
    idx_gbm = rng.choice(len(X_gbm_full), n_gbm, replace=False)
    log.info("GBM training on %s-sample subsample (of %s train seqs)", f"{n_gbm:,}", f"{len(X_gbm_full):,}")
    baselines["gbm"] = gbm_baseline(
        X_gbm_full[idx_gbm], Y_gbm_full[idx_gbm], test_X, test_Y,
        task="regression", seq_mode="last_step",
        n_estimators=100, max_depth=5, random_state=cfg.random_state,
    )
    log.info("gbm baseline:         %s", baselines["gbm"])

    # -- 5) Scale data for LSTM -----------------------------------------------
    train_shards, val_X_s, val_Y_s, test_X_s, test_Y_s, scalers = _per_client_standardise(
        train_shards_raw, val_X, val_Y, test_X, test_Y
    )

    # -- 6) Build model (LSTM regression head only) --------------------------
    spec = MultiOutputSpec(regression_targets=[reg_tgt], classification_targets=[])
    model = LSTMMultiOutput(
        in_features=len(feat_cols),
        sequence_length=cfg.seq_len,
        output_spec=spec,
    ).to(device)
    log.info("model:\n%s", model)

    loss_fn = nn.MSELoss()

    # -- 7) Keep tensors on CPU; lazily move per-client to GPU per round ----
    cpu_tensors: dict[int, tuple[torch.Tensor, torch.Tensor]] = {
        cid: (torch.from_numpy(X), torch.from_numpy(Y))
        for cid, (X, Y) in train_shards.items()
    }
    # Val / test stay on CPU; eval runs in batches.
    vX_cpu = torch.from_numpy(val_X_s)
    vY_cpu = torch.from_numpy(val_Y_s)
    tX_cpu = torch.from_numpy(test_X_s)
    log.info("train shards held on CPU; total %d clients", len(cpu_tensors))

    EVAL_BATCH = 16384

    @torch.no_grad()
    def _batched_eval(m: nn.Module, X_cpu: torch.Tensor, Y_cpu: torch.Tensor) -> float:
        m.eval()
        total = 0.0
        n = 0
        for i in range(0, len(X_cpu), EVAL_BATCH):
            xb = X_cpu[i:i + EVAL_BATCH].to(device, non_blocking=True)
            yb = Y_cpu[i:i + EVAL_BATCH].to(device, non_blocking=True)
            pb = m(xb)
            total += float(((pb - yb) ** 2).sum().item())
            n += yb.numel()
        m.train()
        return total / max(n, 1)

    @torch.no_grad()
    def _batched_predict(m: nn.Module, X_cpu: torch.Tensor) -> np.ndarray:
        m.eval()
        chunks = []
        for i in range(0, len(X_cpu), EVAL_BATCH):
            xb = X_cpu[i:i + EVAL_BATCH].to(device, non_blocking=True)
            chunks.append(m(xb).cpu().numpy())
        m.train()
        return np.concatenate(chunks, axis=0)

    # -- 8) FL training loop --------------------------------------------------
    cids = sorted(train_shards.keys())
    rng = np.random.default_rng(cfg.random_state)
    history: list[V2RoundResult] = []
    best_val, best_state = float("inf"), None

    for r in range(1, cfg.num_rounds + 1):
        t0 = time.time()
        k = min(cfg.clients_per_round, len(cids))
        selected = rng.choice(cids, size=k, replace=False).tolist()
        global_state = {k_: v.detach().cpu() for k_, v in model.state_dict().items()}
        updates = []
        for cid in selected:
            lm = LSTMMultiOutput(
                in_features=len(feat_cols), sequence_length=cfg.seq_len, output_spec=spec,
            ).to(device)
            lm.load_state_dict(global_state, strict=True)
            # Move this client's data onto GPU for the round, free it after.
            Xc, Yc = cpu_tensors[cid]
            Xg = Xc.to(device, non_blocking=True)
            Yg = Yc.to(device, non_blocking=True)
            try:
                upd = train_one_client_cuda_graph(
                    cid, lm, Xg, Yg, loss_fn, device,
                    lr=cfg.client_lr, local_epochs=cfg.local_epochs, batch_size=cfg.batch_size,
                    amp_enabled=amp_enabled, amp_dtype=amp_dtype,
                    seed=cfg.random_state + r * 1000 + cid,
                )
            except RuntimeError as e:
                log.warning("CUDA graph failed for client %s (%s); fallback", cid, e)
                lm = LSTMMultiOutput(
                    in_features=len(feat_cols), sequence_length=cfg.seq_len, output_spec=spec,
                ).to(device)
                lm.load_state_dict(global_state, strict=True)
                upd = train_one_client_gpu_resident(
                    cid, lm, Xg, Yg, loss_fn, device,
                    lr=cfg.client_lr, local_epochs=cfg.local_epochs, batch_size=cfg.batch_size,
                    amp_enabled=amp_enabled, amp_dtype=amp_dtype,
                    seed=cfg.random_state + r * 1000 + cid,
                )
            updates.append(upd)
            # Free this client's GPU tensors before moving on.
            del Xg, Yg, lm
            torch.cuda.empty_cache()
        # Aggregate
        new_state = weighted_average_state_dicts(
            [u.state_dict for u in updates], [u.num_examples for u in updates]
        )
        model.load_state_dict(new_state, strict=True)

        # Eval on val (batched — val is 1.87M sequences).
        val_loss = _batched_eval(model, vX_cpu, vY_cpu)

        train_loss = float(np.average([u.train_loss for u in updates],
                                       weights=[u.num_examples for u in updates]))
        dt = time.time() - t0
        rr = V2RoundResult(round=r, train_loss=train_loss, val_loss=val_loss, duration_s=dt)
        history.append(rr)
        log.info("Round %d/%d  train=%.4f  val=%.4f  dt=%.1fs",
                 r, cfg.num_rounds, train_loss, val_loss, dt)

        if val_loss + 1e-9 < best_val:
            best_val = val_loss
            best_state = {k_: v.detach().cpu() for k_, v in model.state_dict().items()}

    # -- 9) Final test (both in scaled and original space) -------------------
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    test_pred_scaled = _batched_predict(model, tX_cpu)
    # Invert the target scaling so metrics are in real Mbps units.
    test_pred_real = test_pred_scaled[:, 0] * scalers["reg_std"] + scalers["reg_mean"]
    test_real = test_Y[:, 0]
    test_mse = float(np.mean((test_pred_real - test_real) ** 2))
    test_mae = float(np.mean(np.abs(test_pred_real - test_real)))
    test_rmse = float(np.sqrt(test_mse))
    ss_tot = float(np.sum((test_real - test_real.mean()) ** 2))
    ss_res = float(np.sum((test_pred_real - test_real) ** 2))
    test_r2 = 1 - ss_res / max(ss_tot, 1e-12)

    fl_metrics = {"rmse": test_rmse, "mae": test_mae, "r2": test_r2}
    log.info("LSTM-FL test:     %s", fl_metrics)

    # -- 10) Save artifacts -------------------------------------------------
    df_hist = pd.DataFrame([asdict(r) for r in history])
    hist_path = out_dir / "logs" / f"{cfg.name}_history.csv"
    df_hist.to_csv(hist_path, index=False)
    if best_state is not None:
        torch.save(best_state, out_dir / "models" / f"{cfg.name}_best.pt")

    summary = {
        "config": cfg.to_dict(),
        "n_features": len(feat_cols),
        "feat_cols": feat_cols,
        "scalers": scalers,
        "baselines": baselines,
        "fl_lstm": fl_metrics,
        "best_val_scaled": best_val,
    }
    (out_dir / "logs" / f"{cfg.name}_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info("=" * 60)
    log.info("FINAL COMPARISON (target = tx_brate_dl_Mbps forecast, unseen training_configs)")
    log.info("  Persistence RMSE: %.4f  MAE: %.4f  R²: %.4f",
             baselines["persistence"]["rmse"], baselines["persistence"]["mae"], baselines["persistence"]["r2"])
    log.info("  GBM         RMSE: %.4f  MAE: %.4f  R²: %.4f",
             baselines["gbm"]["rmse"], baselines["gbm"]["mae"], baselines["gbm"]["r2"])
    log.info("  FL LSTM     RMSE: %.4f  MAE: %.4f  R²: %.4f",
             test_rmse, test_mae, test_r2)
    log.info("=" * 60)
    return summary
