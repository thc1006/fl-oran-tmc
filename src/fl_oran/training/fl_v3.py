"""FL trainer for v3: classification task, ForecasterV2 (embeddings), step-capped client training.

Supports both IID (each client = one bs_id) and Non-IID (slice-restricted per client) modes.
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..data_v2.encoders import (
    FeatureSchema,
    apply_continuous_scaler,
    federated_fit_scaler,
)
from ..data_v2.partition import partition_clients
from ..data_v2.sequences import build_run_sequences
from ..data_v2.split import ood_split_by_tr
from ..data_v2.targets_v2 import add_classification_target
from ..federated import weighted_average_state_dicts
from ..federated.client_v2 import train_one_client_capped
from ..logging_utils import get_logger
from ..models.forecaster_v2 import ForecasterV2
from ..utils import autocast_dtype, log_cuda_info, pick_device, seed_everything
from .centralized_v3 import (
    V3Config,
    V3_CATEGORICAL,
    V3_CAT_SIZES,
    V3_CONTINUOUS,
    _batched_predict,
    _metrics,
)

log = get_logger(__name__)


def run_federated(cfg: V3Config, *, partition_mode: str, client_slice_map: dict | None = None) -> dict:
    seed_everything(cfg.seed)
    device = pick_device(cfg.device)
    log_cuda_info(device)
    amp_enabled, amp_dtype = autocast_dtype(cfg.mixed_precision)

    if not cfg.unified_parquet.exists():
        raise FileNotFoundError(cfg.unified_parquet)
    df = pd.read_parquet(cfg.unified_parquet)
    if cfg.sample_ratio < 1.0:
        df = df.sample(frac=cfg.sample_ratio, random_state=cfg.seed).sort_index().reset_index(drop=True)
    df = add_classification_target(df, column="ul_bler", threshold=cfg.threshold,
                                   target_name="y_sla_next")
    schema = FeatureSchema(
        categorical=V3_CATEGORICAL, categorical_sizes=V3_CAT_SIZES, continuous=V3_CONTINUOUS,
    )
    feat_cols = schema.categorical + schema.continuous

    split = ood_split_by_tr(df, cfg.train_tr, cfg.val_tr, cfg.test_tr)

    # Partition train
    client_dfs = partition_clients(split.train, mode=partition_mode,
                                   client_slice_map=client_slice_map)
    # Build sequences per client
    client_shards: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for cid, d in client_dfs.items():
        X, Y = build_run_sequences(d, feat_cols, ["y_sla_next"], seq_len=cfg.seq_len)
        if len(X) > 0:
            client_shards[cid] = (X, Y)
    log.info("FL clients: %d  rows/client: %s",
             len(client_shards), {c: len(x) for c, (x, _) in client_shards.items()})

    # Val / test sequences (combined across all clients)
    X_va, Y_va = build_run_sequences(split.val, feat_cols, ["y_sla_next"], seq_len=cfg.seq_len)
    X_te, Y_te = build_run_sequences(split.test, feat_cols, ["y_sla_next"], seq_len=cfg.seq_len)

    # FL-safe scaler fit: sufficient-stats aggregation (server sees only
    # per-client (n, sum_x, sum_x²), never raw rows). Mathematically equal
    # to the pooled version.
    scaler = federated_fit_scaler(
        {cid: X for cid, (X, _) in client_shards.items()}, schema
    )

    def to_tensors(X: np.ndarray, Y: np.ndarray):
        cat, cont = apply_continuous_scaler(X, schema, scaler)
        return (torch.from_numpy(cat), torch.from_numpy(cont),
                torch.from_numpy(Y))

    va_cat, va_cont, va_y = to_tensors(X_va, Y_va)
    te_cat, te_cont, te_y = to_tensors(X_te, Y_te)
    # Per-client scaled tensors (kept on CPU; moved per-round).
    client_cpu: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for cid, (X, Y) in client_shards.items():
        cat, cont = apply_continuous_scaler(X, schema, scaler)
        client_cpu[cid] = (
            torch.from_numpy(cat),
            torch.from_numpy(cont),
            torch.from_numpy(Y),
        )

    # Model / loss
    model = ForecasterV2(schema=schema, task="classification", seq_len=cfg.seq_len).to(device)
    pos_rate = float(Y_te.mean())
    pos_weight = torch.tensor([max((1 - pos_rate) / pos_rate, 1.0)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Packed model accepts (cat, cont) separately — wrap the step-capped trainer's
    # batching by providing X as a tuple (cat, cont) OR train with a custom loop.
    # For simplicity we override: pass categorical+continuous as concatenated float
    # tensor would lose semantics, so we write a simple loop here.

    cids = sorted(client_shards.keys())
    rng = np.random.default_rng(cfg.seed)

    history = []
    best_val_auc, best_state = 0.0, None

    for r in range(1, cfg.num_rounds + 1):
        t0 = time.time()
        k = min(cfg.clients_per_round, len(cids))
        selected = rng.choice(cids, size=k, replace=False).tolist()
        global_state = {k_: v.detach().cpu() for k_, v in model.state_dict().items()}
        updates = []

        # LR warmup
        lr_this = cfg.lr * min(1.0, r / max(cfg.lr_warmup_rounds, 1))

        for cid in selected:
            # Build a fresh local model, load global weights.
            lm = ForecasterV2(schema=schema, task="classification", seq_len=cfg.seq_len).to(device)
            lm.load_state_dict(global_state, strict=True)
            cat_c, cont_c, y_c = client_cpu[cid]
            # Move to GPU for this client round.
            cat_g = cat_c.to(device, non_blocking=True)
            cont_g = cont_c.to(device, non_blocking=True)
            y_g = y_c.to(device, non_blocking=True)

            # Step-capped training (inline loop because we have dual-input model).
            optimizer = torch.optim.Adam(lm.parameters(), lr=lr_this)
            n_local = cat_g.shape[0]
            total_l = 0.0
            for step in range(cfg.max_steps_per_round):
                idx = torch.randint(0, n_local, (cfg.batch_size,), device=device)
                cb = cat_g[idx]
                ob = cont_g[idx]
                yb = y_g[idx]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type,
                                    dtype=amp_dtype or torch.bfloat16, enabled=amp_enabled):
                    logits = lm(cb, ob)
                    loss = loss_fn(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(lm.parameters(), cfg.grad_clip)
                optimizer.step()
                total_l += loss.item()

            state = {k_: v.detach().cpu() for k_, v in lm.state_dict().items()}
            from ..federated import ClientUpdate
            updates.append(ClientUpdate(
                client_id=cid, state_dict=state,
                num_examples=cfg.max_steps_per_round * cfg.batch_size,
                train_loss=total_l / max(cfg.max_steps_per_round, 1),
            ))
            del cat_g, cont_g, y_g, lm
            torch.cuda.empty_cache()

        # FedAvg aggregate
        new_state = weighted_average_state_dicts(
            [u.state_dict for u in updates],
            [u.num_examples for u in updates],
        )
        model.load_state_dict(new_state)

        # Val
        val_logits = _batched_predict(model, va_cat, va_cont, device)
        val_m = _metrics(va_y[:, 0].numpy().astype(int), val_logits[:, 0])
        train_l = float(np.mean([u.train_loss for u in updates]))
        dt = time.time() - t0
        history.append({
            "round": r, "train_loss": train_l,
            "val_auc": val_m.get("auc", 0), "val_acc": val_m["accuracy"],
            "lr": lr_this, "duration_s": dt,
        })
        log.info("Round %d/%d  train=%.4f  val_auc=%.4f  val_acc=%.4f  dt=%.1fs",
                 r, cfg.num_rounds, train_l, val_m.get("auc", 0), val_m["accuracy"], dt)
        if val_m.get("auc", 0) > best_val_auc:
            best_val_auc = val_m["auc"]
            best_state = {k_: v.detach().cpu() for k_, v in model.state_dict().items()}

    # Test
    if best_state is not None:
        model.load_state_dict(best_state)
    test_logits = _batched_predict(model, te_cat, te_cont, device)
    test_m = _metrics(te_y[:, 0].numpy().astype(int), test_logits[:, 0])

    result = {
        "config": cfg.to_dict(),
        "partition_mode": partition_mode,
        "client_slice_map": client_slice_map,
        "history": history,
        "fl_lstm_test": test_m,
    }
    out_dir = Path(cfg.output_dir)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs" / f"{cfg.name}_summary.json").write_text(json.dumps(result, indent=2, default=str))
    pd.DataFrame(history).to_csv(out_dir / "logs" / f"{cfg.name}_history.csv", index=False)
    if best_state is not None:
        torch.save(best_state, out_dir / "models" / f"{cfg.name}_best.pt")

    log.info("=" * 70)
    log.info("FL v3 (%s) test: acc=%.4f AUC=%.4f F1=%.4f",
             partition_mode, test_m["accuracy"], test_m.get("auc", 0), test_m["f1"])
    log.info("=" * 70)
    return result
