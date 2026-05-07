"""REM-1: Local-only per-BS LSTM baseline (R2 reviewer §6.7 suggestion).

Trains an LSTM independently on EACH BS's local train data only — no
federation, no centralized pooling. This is the "what if each gNB
trained alone, with no FL?" alternative that the §6.7 baseline ladder
needs to complete the story:
    persistence < smoothed-persistence < logreg < logreg+cat
                       <
    LOCAL-ONLY (this script)            <  FL  ≤  centralized

Per-BS train data is much smaller than the global train pool (one BS's
train rows vs all 7 BS's). To make the comparison apples-to-apples
against C1 (centralized at 25k steps), we cap LOCAL-ONLY at 25k
gradient steps as well — same compute budget, just no federation.

Hardware: RTX 4080 (~30 sec/cell × 7 BS × 5 seeds = 35 cells × 0.5 min
= ~18 min wall) with mixed_precision=bf16, cudnn_deterministic=True.

Output: artifacts/r2_local_only_per_bs_lstm/results.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--parquet", type=Path,
        default=Path("/home/thc1006/dev/colosseum-oran-federated-slicing/data/coloran_raw_unified.parquet"),
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 42],
                    help="Match Phase 5 / C1 seed coverage.")
    ap.add_argument("--max-steps", type=int, default=25000,
                    help="Per-BS gradient step budget (default 25k = matches FL/C1).")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument(
        "--out", type=Path,
        default=Path("artifacts/r2_local_only_per_bs_lstm/results.json"),
    )
    args = ap.parse_args()

    if not args.parquet.exists():
        print(f"parquet not found: {args.parquet}", file=sys.stderr)
        return 1

    sys.path.insert(0, str(REPO_ROOT / "src"))
    import pandas as pd
    from sklearn.metrics import f1_score, roc_auc_score
    from fl_oran.utils.seed import seed_everything
    from fl_oran.data_v2.encoders import FeatureSchema, fit_continuous_scaler
    from fl_oran.data_v2.features import engineer_features
    from fl_oran.data_v2.split import ood_split_by_tr
    from fl_oran.data_v2.sequences import build_run_sequences
    from fl_oran.training.centralized_v3 import (
        V3_CATEGORICAL,
        V3_CAT_SIZES,
        V3_CONTINUOUS,
    )
    from fl_oran.models.forecaster_v2 import ForecasterV2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; CUDA available: {torch.cuda.is_available()}")

    print(f"Loading parquet {args.parquet} …")
    df = pd.read_parquet(args.parquet)
    df = engineer_features(df)
    split = ood_split_by_tr(df)

    schema = FeatureSchema(
        categorical=V3_CATEGORICAL,
        categorical_sizes=V3_CAT_SIZES,
        continuous=V3_CONTINUOUS,
    )
    feat_cols = list(schema.categorical) + list(schema.continuous)
    n_cat = schema.n_categorical

    # Fit scaler on GLOBAL train only (matches Phase 5 / C1 protocol; this
    # is the only piece that "leaks" across BS — but it's a fixed standard
    # the per-BS local trainers see at deploy time anyway).
    train_arr = split.train[feat_cols].to_numpy(dtype=np.float32)
    scaler = fit_continuous_scaler({0: train_arr}, schema)

    bs_ids = sorted(split.train["bs_id"].unique())
    print(f"Per-BS local-only training; bs_ids = {bs_ids}; seeds = {args.seeds}")

    per_bs_results: dict[int, dict] = {}
    cell_count = 0
    total_cells = len(bs_ids) * len(args.seeds)
    t_start = time.time()

    for bs_id in bs_ids:
        bs_train = split.train[split.train["bs_id"] == bs_id]
        bs_test  = split.test[ split.test["bs_id"]  == bs_id]
        Xtr, Ytr = build_run_sequences(
            bs_train, feat_cols, ["y_sla_violation_next"], seq_len=5,
        )
        Xte, Yte = build_run_sequences(
            bs_test, feat_cols, ["y_sla_violation_next"], seq_len=5,
        )
        if len(Ytr) == 0 or len(Yte) == 0:
            print(f"  bs={bs_id}: SKIP (empty seq)")
            continue
        cat_tr = torch.from_numpy(Xtr[..., :n_cat].astype(np.int64)).to(device)
        cont_tr = torch.from_numpy(((Xtr[..., n_cat:].astype(np.float32) - scaler.mean) / scaler.std)).to(device)
        y_tr = torch.from_numpy(Ytr.squeeze(-1).astype(np.float32)).to(device).unsqueeze(-1)
        cat_te = torch.from_numpy(Xte[..., :n_cat].astype(np.int64)).to(device)
        cont_te = torch.from_numpy(((Xte[..., n_cat:].astype(np.float32) - scaler.mean) / scaler.std)).to(device)
        y_te = torch.from_numpy(Yte.squeeze(-1).astype(np.float32)).to(device).unsqueeze(-1)

        n_train = len(y_tr)
        pos_rate = float(y_tr.mean())
        pos_weight = (1.0 - pos_rate) / max(pos_rate, 1e-6)

        per_seed = []
        for seed in args.seeds:
            cell_count += 1
            seed_everything(seed, deterministic=True)
            model = ForecasterV2(schema=schema, task="classification", seq_len=5).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=args.lr)
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
            rng = np.random.default_rng(seed)
            t_cell = time.time()
            for step in range(args.max_steps):
                idx = rng.choice(n_train, size=args.batch_size, replace=(n_train < args.batch_size))
                idx_t = torch.from_numpy(idx).to(device)
                cat_b = cat_tr.index_select(0, idx_t)
                cont_b = cont_tr.index_select(0, idx_t)
                y_b = y_tr.index_select(0, idx_t)
                opt.zero_grad()
                logit = model(cat_b, cont_b)
                loss = loss_fn(logit, y_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            # Eval
            model.eval()
            logits = []
            with torch.no_grad():
                for i in range(0, len(y_te), 4096):
                    out = model(cat_te[i:i + 4096], cont_te[i:i + 4096])
                    logits.append(out.cpu().numpy())
            y_np = y_te.cpu().numpy().reshape(-1)
            logits = np.concatenate(logits).reshape(-1)
            test_auc = float(roc_auc_score(y_np, logits)) if len(np.unique(y_np)) > 1 else float("nan")
            test_f1 = float(f1_score(y_np, (logits > 0).astype(np.int32))) if len(np.unique(y_np)) > 1 else float("nan")
            wall = time.time() - t_cell
            per_seed.append({"seed": seed, "test_auc": test_auc, "test_f1": test_f1, "wall_s": wall})
            print(f"  [{cell_count}/{total_cells}] bs={bs_id} s{seed}: auc={test_auc:.4f} f1={test_f1:.4f} ({wall:.1f}s)")

        aucs = np.array([r["test_auc"] for r in per_seed])
        per_bs_results[int(bs_id)] = {
            "per_seed": per_seed,
            "test_auc_mean": float(np.mean(aucs)),
            "test_auc_std": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
            "n_train_seqs": int(len(y_tr)),
            "n_test_seqs": int(len(y_te)),
        }

    # Aggregate across BSes
    all_means = [r["test_auc_mean"] for r in per_bs_results.values()]
    cross_bs_mean = float(np.mean(all_means)) if all_means else float("nan")
    cross_bs_std = float(np.std(all_means, ddof=1)) if len(all_means) > 1 else 0.0

    payload = {
        "computed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "description": "R2 REM-1 local-only per-BS LSTM baseline",
        "max_steps_per_cell": args.max_steps,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "n_seeds": len(args.seeds),
        "seeds": args.seeds,
        "n_bs": len(per_bs_results),
        "per_bs": per_bs_results,
        "cross_bs": {
            "test_auc_mean_of_means": cross_bs_mean,
            "test_auc_std_of_means": cross_bs_std,
            "interpretation": (
                "Per-BS local-only LSTM AUC averaged across the 7 base "
                "stations. Compare to: FL (Phase 5 LSTM × FedAvg × natural-"
                "by-BS) = 0.9159; centralized 25k-step (R2 C1) = 0.9243. "
                "If local-only < FL: federation provides a meaningful lift "
                "over per-gNB independent training (the FL value-add story)."
            ),
        },
        "wall_total_s": time.time() - t_start,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\nDone in {payload['wall_total_s']:.1f}s.")
    print(f"Local-only per-BS LSTM AUC (mean of per-BS means, n={len(per_bs_results)} BS, n_seeds={len(args.seeds)}): {cross_bs_mean:.4f} ± {cross_bs_std:.4f}")
    print(f"vs FL Phase 5 (LSTM × FedAvg × natural-by-BS): 0.9159")
    print(f"vs centralized 25k-step (C1): 0.9243")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
