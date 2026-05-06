"""R3.3: Centralized LSTM baseline for the rebuttal MC2-strong question.

Decomposes the naive-vs-FL gap (Round 1 finding: +0.26 AUC over LR) into:
  ML lift     = centralized LSTM AUC − centralized LR AUC
  FL cost     = centralized LSTM AUC − FL LSTM AUC

Two epoch counts:
  1 epoch (~56k steps, wall-clock match to FL cell):
    Tests "does FL match the same wall-clock budget as centralized?"
  3 epochs (~170k steps, convergence-match):
    Tests "what's the centralized ceiling FL is sacrificing?"

Output: artifacts/baselines/centralized_lstm_results.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--parquet",
        type=Path,
        default=Path("/home/thc1006/dev/colosseum-oran-federated-slicing/data/coloran_raw_unified.parquet"),
    )
    ap.add_argument("--epochs", type=int, nargs="+", default=[1, 3],
                    help="Epoch counts to run (default: 1 wall-match + 3 convergence)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=5e-4,
                    help="Match Phase 5 FL LSTM lr (5e-4)")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="Match Phase 5 FL batch_size (64) for fair comparison")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/baselines/centralized_lstm_results.json"),
    )
    args = ap.parse_args()

    if not args.parquet.exists():
        print(f"parquet not found: {args.parquet}", file=sys.stderr)
        return 1

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    import torch
    from sklearn.metrics import f1_score, roc_auc_score
    from fl_oran.utils.seed import seed_everything
    from fl_oran.data_v2.encoders import (
        FeatureSchema,
        fit_continuous_scaler,
    )
    from fl_oran.data_v2.features import engineer_features
    from fl_oran.data_v2.split import ood_split_by_tr
    from fl_oran.data_v2.sequences import build_run_sequences
    from fl_oran.training.centralized_v3 import (
        V3_CATEGORICAL,
        V3_CAT_SIZES,
        V3_CONTINUOUS,
    )
    from fl_oran.models.forecaster_v2 import ForecasterV2

    import pandas as pd

    seed_everything(args.seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; CUDA available: {torch.cuda.is_available()}")

    print(f"Loading parquet {args.parquet} …")
    df = pd.read_parquet(args.parquet)
    df = engineer_features(df)
    split = ood_split_by_tr(df)
    print(f"Train rows {len(split.train):,}; test rows {len(split.test):,}")

    schema = FeatureSchema(
        categorical=V3_CATEGORICAL,
        categorical_sizes=V3_CAT_SIZES,
        continuous=V3_CONTINUOUS,
    )
    feat_cols = V3_CATEGORICAL + V3_CONTINUOUS

    # Fit scaler on train only (matches FL leak-free protocol per §3.5)
    train_arr = split.train[feat_cols].to_numpy(dtype=np.float32)
    scaler = fit_continuous_scaler({0: train_arr}, schema)

    # Build train + test sequences
    print("Building sequences …")
    Xtr, Ytr = build_run_sequences(
        split.train, feat_cols, ["y_sla_violation_next"], seq_len=5,
    )
    Xte, Yte = build_run_sequences(
        split.test, feat_cols, ["y_sla_violation_next"], seq_len=5,
    )
    n_cat = schema.n_categorical
    cat_tr = Xtr[..., :n_cat].astype(np.int64)
    cont_tr = (Xtr[..., n_cat:].astype(np.float32) - scaler.mean) / scaler.std
    y_tr = Ytr.squeeze(-1).astype(np.float32)
    cat_te = Xte[..., :n_cat].astype(np.int64)
    cont_te = (Xte[..., n_cat:].astype(np.float32) - scaler.mean) / scaler.std
    y_te = Yte.squeeze(-1).astype(np.float32)

    print(f"Train sequences: {len(y_tr):,}; test sequences: {len(y_te):,}")
    pos_rate = float(y_tr.mean())
    pos_weight = (1.0 - pos_rate) / pos_rate
    print(f"Train pos_rate {pos_rate:.4f}; pos_weight {pos_weight:.4f}")

    cat_tr_t = torch.from_numpy(cat_tr).to(device)
    cont_tr_t = torch.from_numpy(cont_tr).to(device)
    y_tr_t = torch.from_numpy(y_tr).to(device).unsqueeze(-1)
    cat_te_t = torch.from_numpy(cat_te).to(device)
    cont_te_t = torch.from_numpy(cont_te).to(device)

    n_train = len(y_tr)
    results_per_epoch_count = {}

    # Train fresh model per epoch_count to keep cells independent
    for n_epochs in args.epochs:
        seed_everything(args.seed, deterministic=True)
        model = ForecasterV2(schema=schema, task="classification", seq_len=5).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=device)
        )

        steps_per_epoch = (n_train + args.batch_size - 1) // args.batch_size
        total_steps = n_epochs * steps_per_epoch

        print(f"\n=== Centralized LSTM, {n_epochs} epoch(s), {total_steps:,} total steps ===")

        # Index permutation for shuffling
        rng = np.random.default_rng(args.seed)
        step = 0
        for epoch in range(n_epochs):
            perm = rng.permutation(n_train)
            for batch_start in range(0, n_train, args.batch_size):
                idx = perm[batch_start:batch_start + args.batch_size]
                idx_t = torch.from_numpy(idx).to(device)
                cat_b = cat_tr_t.index_select(0, idx_t)
                cont_b = cont_tr_t.index_select(0, idx_t)
                y_b = y_tr_t.index_select(0, idx_t)
                opt.zero_grad()
                logit = model(cat_b, cont_b)
                loss = loss_fn(logit, y_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                step += 1
                if step % 5000 == 0:
                    print(f"  step {step:,}/{total_steps:,} loss={float(loss):.4f}")

        # Eval
        model.eval()
        with torch.no_grad():
            logits = []
            for i in range(0, len(y_te), 4096):
                out = model(cat_te_t[i:i + 4096], cont_te_t[i:i + 4096])
                logits.append(out.cpu().numpy())
            logits = np.concatenate(logits).reshape(-1)
        test_auc = float(roc_auc_score(y_te, logits))
        test_f1 = float(f1_score(y_te, (logits > 0).astype(np.int32)))

        results_per_epoch_count[f"{n_epochs}_epoch"] = {
            "test_auc": test_auc,
            "test_f1": test_f1,
            "n_total_steps": total_steps,
            "n_train_seqs": int(n_train),
            "n_test_seqs": int(len(y_te)),
        }
        print(f"  test_auc = {test_auc:.4f}; test_f1 = {test_f1:.4f}")

    payload = {
        "centralized_lstm_results": results_per_epoch_count,
        "config": {
            "lr": args.lr,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "feature_count": len(V3_CONTINUOUS),
        },
        "phase5_fl_reference": {
            "lstm_fedavg_natural_test_auc": 0.9159,  # paper §6.2
            "comment": "FL reference for ML-lift / federation-cost decomposition",
        },
        "computed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
