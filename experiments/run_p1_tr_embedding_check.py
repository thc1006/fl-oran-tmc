"""P1.2-GREEN: tr embedding bug-quantification experiment.

Inference-only on existing Phase 5 LSTM checkpoints. Tests the hypothesis
"natural-by-BS dominance is partly tr-embedding-bug-driven" by comparing
test AUC with random-init test_tr embedding rows (the bug) vs with those
rows replaced by mean-of-trained-rows (the fix).

For each (partition, seed) pair:
  AUC_normal:  inference with bit-identical-to-init test_tr rows
  AUC_meanfix: inference with test_tr rows replaced by mean of trained rows

Compute:
  gap_normal  = AUC(natural-by-BS, normal)  - AUC(Dirichlet α=0.05, normal)
  gap_meanfix = AUC(natural-by-BS, meanfix) - AUC(Dirichlet α=0.05, meanfix)
  gap_shrinkage_fraction = (gap_normal - gap_meanfix) / gap_normal
  residual_natural_minus_dirichlet_auc = gap_meanfix

Per preregistered predictions_p1_2_tr_embedding.yaml:
  H1.2.B (gap_shrinkage_fraction < 0.50): bug doesn't explain >50% of gap
  H1.2.C (residual >= 0.05): natural-by-BS lead survives the fix
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent


def freeze_test_tr_rows(
    weight: torch.Tensor | np.ndarray,
    train_tr_indices: list[int],
    mode: str = "mean",
) -> torch.Tensor:
    """Replace embedding rows for tr indices NOT in train_tr_indices.

    `mode='mean'`: replace with mean of trained rows (neutral but in-distribution)
    `mode='zero'`: replace with zero vector
    """
    if isinstance(weight, np.ndarray):
        weight = torch.from_numpy(weight.copy())
    fixed = weight.clone().detach()
    train_set = set(train_tr_indices)
    test_rows = [i for i in range(weight.shape[0]) if i not in train_set]
    if mode == "zero":
        fixed[test_rows] = 0.0
    elif mode == "mean":
        train_mean = weight[list(train_set)].mean(dim=0, keepdim=False)
        for r in test_rows:
            fixed[r] = train_mean
    else:
        raise ValueError(f"unknown mode {mode!r}; expected 'mean' or 'zero'")
    return fixed


# ---------- inference pipeline ----------

def _build_test_tensors(parquet_path: Path):
    """Load test parquet → engineer → split → encode → sequences."""
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

    # Re-fit scaler on train (to apply to test)
    schema = FeatureSchema(
        categorical=V3_CATEGORICAL,
        categorical_sizes=V3_CAT_SIZES,
        continuous=V3_CONTINUOUS,
    )
    feat_cols = V3_CATEGORICAL + V3_CONTINUOUS
    train_arr = split.train[feat_cols].to_numpy(dtype=np.float32)
    scaler = fit_continuous_scaler({0: train_arr}, schema)

    # Build test sequences
    Xte, Yte = build_run_sequences(
        split.test, feat_cols, ["y_sla_violation_next"], seq_len=5,
    )
    # Apply scaler: split (cat, cont) and standardize cont
    n_cat = schema.n_categorical
    cat = Xte[..., :n_cat].astype(np.int64)
    cont = (Xte[..., n_cat:].astype(np.float32) - scaler.mean) / scaler.std
    return cat, cont, Yte.squeeze(-1).astype(np.float32), schema


_ARCH_REGISTRY = {
    "lstm": ("fl_oran.models.forecaster_v2", "ForecasterV2", {}),
    "mamba": ("fl_oran.models.mamba_forecaster", "MambaForecaster", {}),
    "spiking_expand2": (
        "fl_oran.models.spiking_forecaster", "SpikingForecaster",
        {"backbone_d_model": 56, "backbone_expand": 2},
    ),
}


def _load_arch_model(arch: str, ckpt_path: Path, schema):
    """Build the model class for `arch` and load checkpoint state_dict.
    Strips the `_orig_mod.` prefix added by torch.compile."""
    import importlib
    from fl_oran.utils.seed import seed_everything

    if arch not in _ARCH_REGISTRY:
        raise ValueError(f"unknown arch={arch!r}; expected {sorted(_ARCH_REGISTRY)}")
    module_path, cls_name, extra_kwargs = _ARCH_REGISTRY[arch]
    cls = getattr(importlib.import_module(module_path), cls_name)

    seed_everything(0, deterministic=True)
    model = cls(schema=schema, task="classification", seq_len=5, **extra_kwargs)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(cleaned, strict=True)
    model.eval()
    return model


def _load_lstm(ckpt_path: Path, schema):
    """Backward-compat alias for the LSTM-only call signature."""
    return _load_arch_model("lstm", ckpt_path, schema)


@torch.no_grad()
def _eval_auc(model, cat, cont, y, device, batch_size=4096) -> float:
    """Run model on test, return ROC-AUC. ForecasterV2.forward returns
    a single Tensor (verified 2026-05-06; the dict-fallback I had
    initially was wrong about the API — single-task models return a
    plain Tensor of shape [B, 1])."""
    from sklearn.metrics import roc_auc_score
    model = model.to(device)
    cat_t = torch.from_numpy(cat).to(device)
    cont_t = torch.from_numpy(cont).to(device)
    n = len(y)
    logits = []
    for i in range(0, n, batch_size):
        out = model(cat_t[i:i + batch_size], cont_t[i:i + batch_size])
        logits.append(out.detach().cpu().numpy())
    logits = np.concatenate(logits).reshape(-1)
    return float(roc_auc_score(y, logits))


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
    ap.add_argument("--quick", action="store_true",
                    help="Only 2 checkpoints (1 natural + 1 dirichlet, seed 0)")
    ap.add_argument("--archs", type=str, nargs="*", default=["lstm"],
                    help="Architectures to evaluate (default: lstm). "
                         "Supports lstm, mamba, spiking_expand2.")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(range(10)))
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "artifacts" / "p1_tr_embedding" / "results.json",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} (CUDA available: {torch.cuda.is_available()})")
    seeds = [0] if args.quick else args.seeds
    train_tr = list(range(22))

    print(f"Architectures: {args.archs}")
    print(f"Loading test data from {args.parquet} …")
    cat, cont, y, schema = _build_test_tensors(args.parquet)
    print(f"Test: {len(y):,} sequences, n_cat={cat.shape[-1]}, n_cont={cont.shape[-1]}")

    rows = []
    for arch in args.archs:
        cells = [("iid", s) for s in seeds] + [("dirichlet_a0p05", s) for s in seeds]
        for partition, seed in cells:
            ckpt_dir = args.ckpt_root / f"v7_{arch}_fedavg_{partition}_n7_s{seed}"
            ckpt_path = ckpt_dir / "best.pt"
            if not ckpt_path.exists():
                print(f"  SKIP missing checkpoint: {ckpt_path}")
                continue

            model = _load_arch_model(arch, ckpt_path, schema)
            auc_normal = _eval_auc(model, cat, cont, y, device)
            with torch.no_grad():
                orig_w = model.embeddings["tr"].weight.data.clone()
                fixed = freeze_test_tr_rows(orig_w, train_tr, mode="mean")
                model.embeddings["tr"].weight.data.copy_(fixed)
            auc_meanfix = _eval_auc(model, cat, cont, y, device)

            delta = auc_meanfix - auc_normal
            print(f"  {arch:18s} {partition:18s} s{seed:<3}  normal={auc_normal:.4f}  meanfix={auc_meanfix:.4f}  Δ={delta:+.4f}")
            rows.append({
                "arch": arch, "partition": partition, "seed": seed,
                "auc_normal": auc_normal, "auc_meanfix": auc_meanfix,
                "delta_meanfix_minus_normal": delta,
            })

    if not rows:
        print("ERROR: no checkpoints loaded", file=sys.stderr)
        return 1

    def _mean(rs, key): return float(np.mean([r[key] for r in rs])) if rs else float("nan")

    # Per-arch aggregates
    per_arch = {}
    for arch in args.archs:
        iid = [r for r in rows if r["arch"] == arch and r["partition"] == "iid"]
        dir_ = [r for r in rows if r["arch"] == arch and r["partition"] == "dirichlet_a0p05"]
        if not iid or not dir_:
            continue
        nn_ = _mean(iid, "auc_normal")
        nm = _mean(iid, "auc_meanfix")
        dn = _mean(dir_, "auc_normal")
        dm = _mean(dir_, "auc_meanfix")
        gap_n = nn_ - dn
        gap_m = nm - dm
        shrink = (gap_n - gap_m) / gap_n if abs(gap_n) > 1e-6 else float("nan")
        per_arch[arch] = {
            "natural_normal": nn_, "natural_meanfix": nm,
            "dirichlet_normal": dn, "dirichlet_meanfix": dm,
            "gap_normal": gap_n, "gap_meanfix": gap_m,
            "gap_shrinkage_fraction": shrink, "n_seeds": len(iid),
        }

    # Backward-compat top-level aggregate (LSTM if present, else first arch)
    primary = "lstm" if "lstm" in per_arch else (next(iter(per_arch)) if per_arch else None)
    payload = {
        "per_arch": per_arch,
        "per_cell": rows,
        "computed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    if primary is not None:
        a = per_arch[primary]
        payload.update({
            "primary_arch": primary,
            "natural_by_bs_normal_auc_mean": a["natural_normal"],
            "natural_by_bs_meanfix_auc_mean": a["natural_meanfix"],
            "natural_by_bs_frozen_auc_mean": a["natural_meanfix"],
            "dirichlet_a005_normal_auc_mean": a["dirichlet_normal"],
            "dirichlet_a005_meanfix_auc_mean": a["dirichlet_meanfix"],
            "dirichlet_a005_frozen_auc_mean": a["dirichlet_meanfix"],
            "gap_normal": a["gap_normal"],
            "gap_meanfix": a["gap_meanfix"],
            "gap_shrinkage_fraction": a["gap_shrinkage_fraction"],
            "residual_natural_minus_dirichlet_auc": a["gap_meanfix"],
            "n_seeds": a["n_seeds"],
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True))

    print()
    print("=== Per-arch summary ===")
    for arch, a in per_arch.items():
        print(f"  [{arch:18s}] gap_normal={a['gap_normal']:+.4f}  gap_meanfix={a['gap_meanfix']:+.4f}  "
              f"shrinkage={a['gap_shrinkage_fraction']:+.4f}  residual={a['gap_meanfix']:+.4f}")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
