"""P1.1-GREEN: Logistic regression baseline on V3_CONTINUOUS features.

Trains sklearn LogisticRegression on the 17 continuous features at the
same time-step granularity as last-BLER persistence (no sequence info).
Comparing to LSTM/Mamba/Spiking AUC quantifies the marginal value of
sequence modelling over a linear last-step classifier.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np


def fit_logreg(X_train: np.ndarray, y_train: np.ndarray, max_iter: int = 1000):
    """Fit a sklearn LogisticRegression with class-balanced weighting
    (matches the FL trainer's globally-pooled positive-class weighting,
    paper §7.1)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X_train)
    Xs = scaler.transform(X_train)
    lr = LogisticRegression(
        max_iter=max_iter,
        class_weight="balanced",
        random_state=0,
    )
    lr.fit(Xs, y_train)

    # Wrap so the returned estimator transforms inputs automatically.
    class _ScaledLR:
        def __init__(self, scaler, lr):
            self.scaler, self.lr = scaler, lr
        def predict_proba(self, X):
            return self.lr.predict_proba(self.scaler.transform(X))
        def predict(self, X):
            return self.lr.predict(self.scaler.transform(X))
    return _ScaledLR(scaler, lr)


def _merge_results_json(out_path: Path, payload: dict) -> dict:
    if out_path.exists():
        existing = json.loads(out_path.read_text())
    else:
        existing = {}
    existing.update(payload)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(existing, indent=2, sort_keys=True))
    return existing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--parquet",
        type=Path,
        default=Path("/home/thc1006/dev/colosseum-oran-federated-slicing/data/coloran_raw_unified.parquet"),
    )
    ap.add_argument("--max-iter", type=int, default=1000)
    ap.add_argument(
        "--include-categorical",
        action="store_true",
        help="Also one-hot encode V3_CATEGORICAL (slice_id, sched, tr) and "
             "concat with V3_CONTINUOUS (R2 reviewer #13 suggestion: report "
             "logreg-with-categorical alongside logreg-continuous-only).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/baselines/naive_results.json"),
    )
    args = ap.parse_args()

    if not args.parquet.exists():
        print(f"parquet not found: {args.parquet}", file=sys.stderr)
        return 1

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    import pandas as pd
    from sklearn.metrics import f1_score, roc_auc_score
    from fl_oran.data_v2.features import engineer_features
    from fl_oran.data_v2.split import ood_split_by_tr
    from fl_oran.training.centralized_v3 import V3_CATEGORICAL, V3_CONTINUOUS

    print(f"Loading parquet: {args.parquet}")
    df = pd.read_parquet(args.parquet)
    df = engineer_features(df)
    split = ood_split_by_tr(df)

    feature_cols = [c for c in V3_CONTINUOUS if c in split.train.columns]
    if len(feature_cols) != len(V3_CONTINUOUS):
        missing = set(V3_CONTINUOUS) - set(feature_cols)
        print(f"WARNING: {len(missing)} V3_CONTINUOUS features missing from data: {missing}",
              file=sys.stderr)

    X_train = split.train[feature_cols].values.astype(np.float32)
    y_train = split.train["y_sla_violation_next"].values.astype(np.int32)
    X_test = split.test[feature_cols].values.astype(np.float32)
    y_test = split.test["y_sla_violation_next"].values.astype(np.int32)

    if args.include_categorical:
        # R2 reviewer #13 suggestion. One-hot encode V3_CATEGORICAL using
        # train-set categories only; test-time categories not in train fall
        # to all-zero (OOD-safe). For tr ∈ {25..27}, all 3 test-tr OHE
        # columns are all-zero — matches the nn.Embedding(30, 8) random-init
        # behaviour at test time documented in §7.1.6.
        cat_cols = [c for c in V3_CATEGORICAL if c in split.train.columns]
        train_cats: dict[str, np.ndarray] = {}
        for c in cat_cols:
            train_cats[c] = np.sort(split.train[c].dropna().unique())
        def _ohe(df_split, c, cats):
            arr = df_split[c].to_numpy()
            out = np.zeros((len(arr), len(cats)), dtype=np.float32)
            for j, v in enumerate(cats):
                out[:, j] = (arr == v).astype(np.float32)
            return out
        ohe_train = [_ohe(split.train, c, train_cats[c]) for c in cat_cols]
        ohe_test  = [_ohe(split.test,  c, train_cats[c]) for c in cat_cols]
        X_train = np.hstack([X_train] + ohe_train).astype(np.float32)
        X_test  = np.hstack([X_test]  + ohe_test ).astype(np.float32)
        n_cat_dims = sum(len(train_cats[c]) for c in cat_cols)
        print(f"+categorical OHE: {len(cat_cols)} cat cols → {n_cat_dims} dims; "
              f"total feature_dim = {X_train.shape[1]}")

    print(f"Train rows: {len(X_train):,}; test rows: {len(X_test):,}")
    print(f"Feature count: {X_train.shape[1]} (V3_CONTINUOUS expected 17{'+ V3_CATEGORICAL OHE' if args.include_categorical else ''})")

    print("\nFitting LogisticRegression …")
    est = fit_logreg(X_train, y_train, max_iter=args.max_iter)
    test_proba = est.predict_proba(X_test)[:, 1]
    test_pred = est.predict(X_test)
    test_auc = float(roc_auc_score(y_test, test_proba))
    test_f1 = float(f1_score(y_test, test_pred))

    print()
    print("=== Logistic regression baseline (test split, tr ∈ {25..27}) ===")
    print(f"  feature count:               {len(feature_cols)}")
    print(f"  test rows:                   {len(X_test):,}")
    print(f"  ROC-AUC:                     {test_auc:.4f}")
    print(f"  F1:                          {test_f1:.4f}")

    if args.include_categorical:
        payload = {
            "logreg_plus_cat_test_auc": test_auc,
            "logreg_plus_cat_test_f1": test_f1,
            "logreg_plus_cat_n_features": int(X_train.shape[1]),
            "logreg_plus_cat_n_continuous": int(len(feature_cols)),
            "logreg_plus_cat_computed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
    else:
        payload = {
            "logreg_test_auc": test_auc,
            "logreg_test_f1": test_f1,
            "logreg_n_features": int(len(feature_cols)),
            "logreg_computed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
    # Reuse positive_rate/n_test_rows if last_bler script already populated them
    merged = _merge_results_json(args.out, payload)
    print(f"\nWrote {args.out} (keys: {sorted(merged.keys())})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
