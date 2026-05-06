"""P2.1 LOTO cluster bootstrap (MC5 reviewer concern).

Reviewer MC5: "10-seed paired bootstrap captures init RNG only; cluster
bootstrap over tr/BS would expose external uncertainty."

Strategy:
1. Load existing Phase 5 LSTM × FedAvg × natural-by-BS checkpoints (10
   seeds, inference-only — no retraining)
2. Run inference on test set; partition predictions by tr ∈ {25, 26, 27}
3. Compute per-cell per-tr AUC (10 cells × 3 tr = 30 values)
4. Decompose:
   - σ_init  = std across 10 seeds (current paper methodology)
   - σ_tr    = mean per-cell std across 3 tr configs (external uncertainty)
   - LOTO CI = combined cluster bootstrap (resample tr's + seeds)

If σ_tr >> σ_init, the standard paired-bootstrap CI95 is too tight —
§4.5 + §8 L13 should add caveat about external uncertainty.

Output: artifacts/p2_loto/results.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent


def _build_test_tensors_with_tr(parquet_path: Path):
    """Load test parquet → engineer → split → encode → sequences,
    AND return the tr value for each sequence."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import pandas as pd
    from fl_oran.data_v2.features import engineer_features
    from fl_oran.data_v2.split import ood_split_by_tr
    from fl_oran.data_v2.sequences import build_run_sequences
    from fl_oran.data_v2.encoders import FeatureSchema, fit_continuous_scaler
    from fl_oran.training.centralized_v3 import (
        V3_CATEGORICAL,
        V3_CAT_SIZES,
        V3_CONTINUOUS,
    )

    df = pd.read_parquet(parquet_path)
    df = engineer_features(df)
    split = ood_split_by_tr(df)
    schema = FeatureSchema(
        categorical=V3_CATEGORICAL,
        categorical_sizes=V3_CAT_SIZES,
        continuous=V3_CONTINUOUS,
    )
    feat_cols = V3_CATEGORICAL + V3_CONTINUOUS
    train_arr = split.train[feat_cols].to_numpy(dtype=np.float32)
    scaler = fit_continuous_scaler({0: train_arr}, schema)

    Xte, Yte = build_run_sequences(
        split.test, feat_cols, ["y_sla_violation_next"], seq_len=5,
    )
    n_cat = schema.n_categorical
    cat = Xte[..., :n_cat].astype(np.int64)
    cont = (Xte[..., n_cat:].astype(np.float32) - scaler.mean) / scaler.std
    y = Yte.squeeze(-1).astype(np.float32)

    # Extract tr for each sequence: tr is at the END of each window
    # (matches build_run_sequences alignment — label at last position)
    tr_idx_in_cat = V3_CATEGORICAL.index("tr")
    tr_per_seq = cat[:, -1, tr_idx_in_cat].astype(np.int32)

    return cat, cont, y, tr_per_seq, schema


@torch.no_grad()
def _eval_per_tr(model, cat, cont, y, tr_per_seq, device, batch_size=4096):
    """Inference + per-tr-config AUC."""
    from sklearn.metrics import roc_auc_score
    model = model.to(device).eval()
    cat_t = torch.from_numpy(cat).to(device)
    cont_t = torch.from_numpy(cont).to(device)
    n = len(y)
    logits = []
    for i in range(0, n, batch_size):
        out = model(cat_t[i:i + batch_size], cont_t[i:i + batch_size])
        logits.append(out.detach().cpu().numpy())
    logits = np.concatenate(logits).reshape(-1)

    overall_auc = float(roc_auc_score(y, logits))
    per_tr = {}
    for tr in sorted(np.unique(tr_per_seq)):
        mask = tr_per_seq == tr
        if mask.sum() < 10 or len(np.unique(y[mask])) < 2:
            continue
        per_tr[int(tr)] = {
            "test_auc": float(roc_auc_score(y[mask], logits[mask])),
            "n_seqs": int(mask.sum()),
            "pos_rate": float(y[mask].mean()),
        }
    return overall_auc, per_tr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--parquet",
        type=Path,
        default=Path("/home/thc1006/dev/colosseum-oran-federated-slicing/data/coloran_raw_unified.parquet"),
    )
    ap.add_argument(
        "--ckpt-root", type=Path,
        default=REPO_ROOT / "artifacts" / "v7_stage2_full",
    )
    ap.add_argument("--arch", default="lstm",
                    choices=["lstm", "mamba", "spiking_expand2"])
    ap.add_argument("--algo", default="fedavg")
    ap.add_argument("--partition", default="iid")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    ap.add_argument("--n-boot", type=int, default=10000,
                    help="Cluster bootstrap iterations")
    ap.add_argument("--bootstrap-seed", type=int, default=2026,
                    help="Match Section 4.5 base seed for algorithm-pair sweep")
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "artifacts" / "p2_loto" / "results.json",
    )
    args = ap.parse_args()

    if not args.parquet.exists():
        print(f"parquet not found: {args.parquet}", file=sys.stderr)
        return 1

    sys.path.insert(0, str(REPO_ROOT / "src"))
    import importlib
    from fl_oran.utils.seed import seed_everything

    arch_registry = {
        "lstm": ("fl_oran.models.forecaster_v2", "ForecasterV2", {}),
        "mamba": ("fl_oran.models.mamba_forecaster", "MambaForecaster", {}),
        "spiking_expand2": (
            "fl_oran.models.spiking_forecaster", "SpikingForecaster",
            {"backbone_d_model": 56, "backbone_expand": 2},
        ),
    }
    module_path, cls_name, extra_kwargs = arch_registry[args.arch]
    cls = getattr(importlib.import_module(module_path), cls_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; arch: {args.arch}; algo: {args.algo}; partition: {args.partition}")

    print(f"Loading test data from {args.parquet} …")
    cat, cont, y, tr_per_seq, schema = _build_test_tensors_with_tr(args.parquet)
    test_tr_values = sorted(np.unique(tr_per_seq).tolist())
    print(f"Test sequences: {len(y):,}; test tr values: {test_tr_values}")

    per_cell_results = []
    for seed in args.seeds:
        ckpt = args.ckpt_root / f"v7_{args.arch}_{args.algo}_{args.partition}_n7_s{seed}" / "best.pt"
        if not ckpt.exists():
            print(f"  SKIP missing {ckpt}")
            continue
        seed_everything(0, deterministic=True)
        model = cls(schema=schema, task="classification", seq_len=5, **extra_kwargs)
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        cleaned = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        model.load_state_dict(cleaned, strict=True)
        overall_auc, per_tr = _eval_per_tr(model, cat, cont, y, tr_per_seq, device)
        per_cell_results.append({"seed": seed, "overall_test_auc": overall_auc, "per_tr": per_tr})
        per_tr_str = ", ".join(f"tr{t}={per_tr[t]['test_auc']:.4f}" for t in sorted(per_tr))
        print(f"  s{seed:<3}  overall={overall_auc:.4f}  | {per_tr_str}")

    if not per_cell_results:
        print("ERROR: no checkpoints loaded", file=sys.stderr)
        return 1

    # Variance decomposition
    overall_per_seed = [r["overall_test_auc"] for r in per_cell_results]
    sigma_init = stdev(overall_per_seed) if len(overall_per_seed) > 1 else 0.0
    print()
    print(f"=== Variance decomposition (n={len(per_cell_results)} seeds) ===")
    print(f"  σ_init  (across seeds, current paper methodology):  {sigma_init:.6f}")

    # Per-cell per-tr std (the "external" component within a single cell)
    per_cell_tr_std = []
    for r in per_cell_results:
        aucs = [r["per_tr"][t]["test_auc"] for t in sorted(r["per_tr"])]
        if len(aucs) > 1:
            per_cell_tr_std.append(stdev(aucs))
    sigma_tr_within = mean(per_cell_tr_std) if per_cell_tr_std else 0.0
    print(f"  σ_tr (within-cell, mean across seeds, MC5 external): {sigma_tr_within:.6f}")
    print(f"  ratio σ_tr / σ_init: {sigma_tr_within / sigma_init if sigma_init else float('inf'):.2f}x")

    # Cluster bootstrap over (seed, tr) pairs: resample tr-configs
    # WITH REPLACEMENT for each iteration, average across all seeds.
    rng = np.random.default_rng(args.bootstrap_seed)
    boot_means = []
    n_seeds = len(per_cell_results)
    for _ in range(args.n_boot):
        # Resample 3 tr configs with replacement
        tr_sample = rng.choice(test_tr_values, size=len(test_tr_values), replace=True)
        # Average per-(seed, tr) AUCs across the resampled tr's then across seeds
        seed_means = []
        for r in per_cell_results:
            tr_aucs = [r["per_tr"].get(int(t), {}).get("test_auc")
                       for t in tr_sample]
            tr_aucs = [a for a in tr_aucs if a is not None]
            if tr_aucs:
                seed_means.append(mean(tr_aucs))
        if seed_means:
            boot_means.append(mean(seed_means))
    boot_mean = float(np.mean(boot_means))
    boot_ci_lo = float(np.percentile(boot_means, 2.5))
    boot_ci_hi = float(np.percentile(boot_means, 97.5))
    print(f"  Cluster-bootstrap-over-tr CI95: [{boot_ci_lo:.4f}, {boot_ci_hi:.4f}]  (mean {boot_mean:.4f})")
    standard_ci_lo = mean(overall_per_seed) - 1.96 * sigma_init / (n_seeds ** 0.5)
    standard_ci_hi = mean(overall_per_seed) + 1.96 * sigma_init / (n_seeds ** 0.5)
    print(f"  Standard seed-paired CI95:      [{standard_ci_lo:.4f}, {standard_ci_hi:.4f}]")
    cluster_width = boot_ci_hi - boot_ci_lo
    standard_width = standard_ci_hi - standard_ci_lo
    print(f"  Width ratio (cluster / standard): {cluster_width / standard_width if standard_width else float('inf'):.2f}x")

    payload = {
        "description": f"P2.1 LOTO cluster bootstrap on {args.arch}×{args.algo}×{args.partition} (n={n_seeds} seeds × 3 test tr configs)",
        "per_cell": per_cell_results,
        "test_tr_values": test_tr_values,
        "n_seeds": n_seeds,
        "sigma_init_across_seeds": sigma_init,
        "sigma_tr_within_cell_mean": sigma_tr_within,
        "sigma_ratio_tr_over_init": sigma_tr_within / sigma_init if sigma_init else None,
        "cluster_bootstrap_ci95_lo": boot_ci_lo,
        "cluster_bootstrap_ci95_hi": boot_ci_hi,
        "cluster_bootstrap_mean": boot_mean,
        "cluster_bootstrap_n_iters": args.n_boot,
        "cluster_bootstrap_seed": args.bootstrap_seed,
        "standard_seed_paired_ci95_lo": standard_ci_lo,
        "standard_seed_paired_ci95_hi": standard_ci_hi,
        "ci_width_ratio_cluster_over_standard": (boot_ci_hi - boot_ci_lo) / (standard_ci_hi - standard_ci_lo) if (standard_ci_hi - standard_ci_lo) else None,
        "computed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
