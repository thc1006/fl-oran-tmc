"""P1.1-GREEN: Last-BLER persistence baseline.

Reviewer MC2 highlights that the paper's FL benchmark lacks a naive
baseline. Since `ul_bler[t]` is in the input features and the target is
`1[ul_bler[t+1] > 0.10]`, the trivial persistence predictor
`y_pred = 1[ul_bler[t] > 0.10]` (or, for AUC, score = ul_bler[t]) is the
canonical baseline FL methods must beat by a meaningful margin.

Output: artifacts/baselines/naive_results.json (also written by
baseline_logreg.py — they share the file by merging).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np


def predict_last_bler(ul_bler: np.ndarray, threshold: float = 0.10) -> np.ndarray:
    """Persistence prediction: y_t+1 = 1[ul_bler_t > threshold]."""
    return (np.asarray(ul_bler) > threshold).astype(np.int32)


def score_last_bler(ul_bler: np.ndarray) -> np.ndarray:
    """For AUC: use raw ul_bler as discriminative score (higher = more
    likely to be in violation next step). This makes ROC-AUC meaningful;
    threshold-based 0/1 prediction collapses to balanced accuracy."""
    return np.asarray(ul_bler).astype(np.float32)


def score_smoothed_bler(
    ul_bler_seq: np.ndarray, window: int = 5
) -> np.ndarray:
    """Smoothed persistence: score = trailing mean of ul_bler over the
    last `window` seconds within the same (run, slice) group. Addresses
    the issue that raw ul_bler[t] can be noisy at 1-s granularity even if
    short-term mean is informative."""
    import pandas as pd
    s = pd.Series(np.asarray(ul_bler_seq))
    return s.rolling(window=window, min_periods=1).mean().to_numpy().astype(np.float32)


def _merge_results_json(out_path: Path, payload: dict) -> dict:
    """Merge new keys into existing naive_results.json (so persistence
    + logreg can share the file)."""
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
    ap.add_argument("--threshold", type=float, default=0.10)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/baselines/naive_results.json"),
    )
    args = ap.parse_args()

    if not args.parquet.exists():
        print(f"parquet not found: {args.parquet}", file=sys.stderr)
        return 1

    # Local imports (heavy)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    import pandas as pd
    from sklearn.metrics import f1_score, roc_auc_score
    from fl_oran.data_v2.features import engineer_features, SLA_BLER_THRESHOLD
    from fl_oran.data_v2.split import ood_split_by_tr

    print(f"Loading parquet: {args.parquet}")
    df = pd.read_parquet(args.parquet)
    print(f"Raw rows: {len(df):,}")

    print("engineer_features (adds y_sla_violation_next + drops NaN) …")
    df = engineer_features(df)
    print(f"After engineering: {len(df):,}")

    split = ood_split_by_tr(df)
    test = split.test
    print(f"Test rows: {len(test):,}; train rows: {len(split.train):,}")

    # Per-(run_id, slice_id) Pearson auto-correlation diagnostic — distinguishes
    # "BLER is white noise" from "BLER is correlated but threshold crossings
    # are unpredictable". My previous interpretation conflated AUC<0.55 with
    # "near-zero autoregressive correlation"; this measures it directly.
    test_sorted = test.sort_values(["run_id", "slice_id", "step_idx"])
    pearson_corr = float(
        test_sorted.groupby(["run_id", "slice_id"], observed=True)["ul_bler"]
        .apply(lambda s: s.autocorr(lag=1) if len(s) > 1 else np.nan)
        .dropna().mean()
    )

    # Raw persistence baseline
    ul_bler_test = test["ul_bler"].values
    y_test = test["y_sla_violation_next"].values
    score = score_last_bler(ul_bler_test)
    pred = predict_last_bler(ul_bler_test, threshold=args.threshold)
    test_auc = float(roc_auc_score(y_test, score))
    test_f1 = float(f1_score(y_test, pred))
    pos_rate = float(y_test.mean())

    # Smoothed persistence baseline (per-group rolling mean over 5 seconds)
    smoothed_score = (
        test_sorted.groupby(["run_id", "slice_id"], observed=True)["ul_bler"]
        .transform(lambda s: s.rolling(window=5, min_periods=1).mean())
    )
    # re-align to original test order (test_sorted re-sorts; rebuild AUC on the
    # sorted index then map back)
    smoothed_auc = float(roc_auc_score(
        test_sorted["y_sla_violation_next"].values, smoothed_score.values
    ))

    print()
    print("=== Last-BLER persistence baseline (test split, tr ∈ {25..27}) ===")
    print(f"  threshold:                       {args.threshold}")
    print(f"  test rows:                       {len(test):,}")
    print(f"  positive rate:                   {pos_rate:.4f}")
    print(f"  per-group lag-1 Pearson(ul_bler): {pearson_corr:.4f}")
    print(f"  Raw  persistence ROC-AUC:        {test_auc:.4f}")
    print(f"  Raw  persistence F1@threshold:   {test_f1:.4f}")
    print(f"  Smoothed-5s persistence ROC-AUC: {smoothed_auc:.4f}")

    payload = {
        "last_bler_test_auc": test_auc,
        "last_bler_test_f1": test_f1,
        "smoothed_5s_persistence_test_auc": smoothed_auc,
        "ul_bler_lag1_pearson_per_group_mean": pearson_corr,
        "n_test_rows": int(len(test)),
        "positive_rate": pos_rate,
        "threshold": args.threshold,
        "computed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    merged = _merge_results_json(args.out, payload)
    print(f"\nWrote {args.out} (keys: {sorted(merged.keys())})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
