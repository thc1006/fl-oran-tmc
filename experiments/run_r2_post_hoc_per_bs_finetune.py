"""R2 C3: post-hoc per-BS fine-tune (reviewer R2 #5 / A2 — FedBN spirit).

Reviewer R2 #5: our paper's mechanism story attributes the natural-by-BS
dominance to per-BS feature shift (CQI / MCS distribution differences).
FedBN was literally designed for feature-shift non-IID; we argued FedBN
reduces to FedAvg on no-BN backbones (true) but that argument satisfies
the *letter* not the *spirit* of MC3.

This script fills the spirit: take a global Phase-5 FedAvg checkpoint,
fine-tune it on each BS's local train data, and compare the resulting
per-BS personalised AUC against the global model on the per-BS test split.

Per-BS Δ_personalised = AUC_personalised − AUC_global
Mean Δ across (arch, seed, BS) tells us:
  Δ < +0.005  → "feature-shift personalisation gives little" → strengthens §8 L2
  Δ ≈ 0      → personalisation neutral
  Δ > +0.01  → personalisation helps; paper needs a new caveat

Hardware: V100 cluster (4 cards × ~4 concurrent cells/card oversubscribed
per artifacts/audit/r2_gpu_design.md). FP16 for V100 BF16-emulation
avoidance. Each cell needs ~5 GiB VRAM and ~3 min wall.

Output (per-cell): artifacts/r2_post_hoc_per_bs_finetune/<cell_id>.json
Aggregator (later): artifacts/r2_post_hoc_per_bs_finetune/aggregated.json

Run on a single GPU (pure CLI, no spec yaml):
  python experiments/run_r2_post_hoc_per_bs_finetune.py \\
    --cells "lstm:s0,lstm:s1,..." \\
    --device cuda:0 \\
    --finetune-steps 200 \\
    --batch-size 64 \\
    --mixed-precision fp16 \\
    --out artifacts/r2_post_hoc_per_bs_finetune

The launcher scripts/v100_r2_c3_launcher.sh distributes the 105 cells
(7 BS × 3 archs × 5 seeds — BS expanded internally, so cells = arch:seed
combos) across the 4 V100 cards.
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

# Mirror the registry from run_p2_inference_latency.py for consistency.
ARCH_REGISTRY = {
    "lstm": ("fl_oran.models.forecaster_v2", "ForecasterV2", {}),
    "mamba": ("fl_oran.models.mamba_forecaster", "MambaForecaster", {}),
    "spiking_expand2": (
        "fl_oran.models.spiking_forecaster", "SpikingForecaster",
        {"backbone_d_model": 56, "backbone_expand": 2},
    ),
}

# Phase 5 LSTM × FedAvg × natural-by-BS × seed=<S> checkpoint dir convention.
# v7_stage2_full / v7_<arch>_fedavg_iid_n7_s<seed> / best.pt
PHASE5_DIR = REPO_ROOT / "artifacts" / "v7_stage2_full"


def _ckpt_path(arch: str, seed: int) -> Path:
    return PHASE5_DIR / f"v7_{arch}_fedavg_iid_n7_s{seed}" / "best.pt"


def _build_model(arch: str):
    """Build a fresh model + return (model, schema). Caller loads
    state_dict separately so we can refresh weights between BS cells."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import importlib
    from fl_oran.utils.seed import seed_everything
    from fl_oran.data_v2.encoders import FeatureSchema
    from fl_oran.training.centralized_v3 import (
        V3_CATEGORICAL,
        V3_CAT_SIZES,
        V3_CONTINUOUS,
    )
    if arch not in ARCH_REGISTRY:
        raise ValueError(f"unknown arch={arch!r}")
    module_path, cls_name, extra_kwargs = ARCH_REGISTRY[arch]
    cls = getattr(importlib.import_module(module_path), cls_name)
    schema = FeatureSchema(
        categorical=V3_CATEGORICAL,
        categorical_sizes=V3_CAT_SIZES,
        continuous=V3_CONTINUOUS,
    )
    seed_everything(0, deterministic=True)
    model = cls(schema=schema, task="classification", seq_len=5, **extra_kwargs)
    return model, schema


def _load_ckpt_into(model, ckpt_path: Path) -> None:
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(cleaned, strict=True)


def _build_per_bs_tensors(
    parquet_path: Path, schema, device,
):
    """Load the unified parquet, engineer features, build sequences,
    return per-BS train+test tensors as a dict keyed by bs_id.

    Reuses the same data pipeline as run_p1_centralized_lstm.py so the
    train/test split + standardisation are identical to Phase 5."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from fl_oran.data_v2.features import engineer_features
    from fl_oran.data_v2.split import ood_split_by_tr
    from fl_oran.data_v2.sequences import build_run_sequences
    from fl_oran.data_v2.encoders import fit_continuous_scaler
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    df = engineer_features(df)
    split = ood_split_by_tr(df)

    feat_cols = list(schema.categorical) + list(schema.continuous)
    train_arr = split.train[feat_cols].to_numpy(dtype=np.float32)
    scaler = fit_continuous_scaler({0: train_arr}, schema)
    n_cat = schema.n_categorical

    per_bs: dict[int, dict] = {}
    for bs_id in sorted(split.train["bs_id"].unique()):
        tr_bs = split.train[split.train["bs_id"] == bs_id]
        te_bs = split.test[split.test["bs_id"] == bs_id]
        if len(tr_bs) == 0 or len(te_bs) == 0:
            continue
        Xtr, Ytr = build_run_sequences(
            tr_bs, feat_cols, ["y_sla_violation_next"], seq_len=5,
        )
        Xte, Yte = build_run_sequences(
            te_bs, feat_cols, ["y_sla_violation_next"], seq_len=5,
        )
        if len(Ytr) == 0 or len(Yte) == 0:
            continue
        cat_tr = Xtr[..., :n_cat].astype(np.int64)
        cont_tr = (Xtr[..., n_cat:].astype(np.float32) - scaler.mean) / scaler.std
        y_tr = Ytr.squeeze(-1).astype(np.float32)
        cat_te = Xte[..., :n_cat].astype(np.int64)
        cont_te = (Xte[..., n_cat:].astype(np.float32) - scaler.mean) / scaler.std
        y_te = Yte.squeeze(-1).astype(np.float32)
        per_bs[int(bs_id)] = dict(
            cat_tr=torch.from_numpy(cat_tr).to(device),
            cont_tr=torch.from_numpy(cont_tr).to(device),
            y_tr=torch.from_numpy(y_tr).to(device).unsqueeze(-1),
            cat_te=torch.from_numpy(cat_te).to(device),
            cont_te=torch.from_numpy(cont_te).to(device),
            y_te=torch.from_numpy(y_te).to(device).unsqueeze(-1),
        )
    return per_bs


def _eval_auc(model, cat_te, cont_te, y_te, batch=4096) -> float:
    from sklearn.metrics import roc_auc_score
    model.eval()
    logits = []
    with torch.no_grad():
        for i in range(0, len(y_te), batch):
            out = model(cat_te[i:i + batch], cont_te[i:i + batch])
            logits.append(out.cpu().numpy())
    y_np = y_te.cpu().numpy().reshape(-1)
    logits = np.concatenate(logits).reshape(-1)
    if len(np.unique(y_np)) < 2:
        return float("nan")
    return float(roc_auc_score(y_np, logits))


def _finetune_per_bs(
    model, cat_tr, cont_tr, y_tr,
    n_steps: int, batch_size: int, lr: float,
    device, mixed_precision: str,
) -> None:
    """Adam fine-tune in-place on the per-BS train data for n_steps."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    n_train = len(y_tr)
    rng = np.random.default_rng(seed=0)
    use_amp = mixed_precision in ("fp16", "bf16") and device.type == "cuda"
    amp_dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda") if mixed_precision == "fp16" else None
    for step in range(n_steps):
        idx = rng.choice(n_train, size=batch_size, replace=(n_train < batch_size))
        idx_t = torch.from_numpy(idx).to(device)
        cat_b = cat_tr.index_select(0, idx_t)
        cont_b = cont_tr.index_select(0, idx_t)
        y_b = y_tr.index_select(0, idx_t)
        opt.zero_grad()
        if use_amp:
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                logit = model(cat_b, cont_b)
                loss = loss_fn(logit, y_b)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        else:
            logit = model(cat_b, cont_b)
            loss = loss_fn(logit, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


def run_one_cell(
    arch: str, seed: int, parquet_path: Path,
    finetune_steps: int, batch_size: int, lr: float,
    device, mixed_precision: str,
) -> dict:
    """Fine-tune per BS, eval personalised vs global per-BS AUC.

    Returns a dict with per-BS results suitable for JSON serialisation.
    """
    ckpt = _ckpt_path(arch, seed)
    if not ckpt.exists():
        raise FileNotFoundError(f"missing Phase 5 checkpoint: {ckpt}")
    print(f"[{arch} s{seed}] loading checkpoint {ckpt}")
    model, schema = _build_model(arch)
    _load_ckpt_into(model, ckpt)
    model.to(device)

    per_bs_tensors = _build_per_bs_tensors(parquet_path, schema, device)

    # First pass: GLOBAL (no fine-tune) per-BS AUC.
    global_auc: dict[int, float] = {}
    for bs_id, t in per_bs_tensors.items():
        global_auc[bs_id] = _eval_auc(model, t["cat_te"], t["cont_te"], t["y_te"])
    print(f"[{arch} s{seed}] global per-BS AUC: "
          f"{{{', '.join(f'{k}={v:.4f}' for k, v in global_auc.items())}}}")

    # Second pass: PERSONALISED per-BS AUC. Per BS, reload fresh global
    # weights then fine-tune on that BS's train data only.
    personalised_auc: dict[int, float] = {}
    for bs_id, t in per_bs_tensors.items():
        # Refresh weights from the global checkpoint
        _load_ckpt_into(model, ckpt)
        model.to(device)
        _finetune_per_bs(
            model, t["cat_tr"], t["cont_tr"], t["y_tr"],
            n_steps=finetune_steps, batch_size=batch_size, lr=lr,
            device=device, mixed_precision=mixed_precision,
        )
        personalised_auc[bs_id] = _eval_auc(
            model, t["cat_te"], t["cont_te"], t["y_te"],
        )
        delta = personalised_auc[bs_id] - global_auc[bs_id]
        print(f"  bs={bs_id}: personalised={personalised_auc[bs_id]:.4f} "
              f"global={global_auc[bs_id]:.4f} Δ={delta:+.4f}")

    deltas = {bs: personalised_auc[bs] - global_auc[bs] for bs in global_auc}
    payload = {
        "arch": arch,
        "seed": seed,
        "finetune_steps": finetune_steps,
        "batch_size": batch_size,
        "lr": lr,
        "mixed_precision": mixed_precision,
        "ckpt": str(ckpt),
        "per_bs": {
            int(bs): {
                "global_auc": float(global_auc[bs]),
                "personalised_auc": float(personalised_auc[bs]),
                "delta_personalised_minus_global": float(deltas[bs]),
                "n_train": int(len(per_bs_tensors[bs]["y_tr"])),
                "n_test": int(len(per_bs_tensors[bs]["y_te"])),
            }
            for bs in sorted(global_auc.keys())
        },
        "summary": {
            "mean_delta": float(np.mean(list(deltas.values()))),
            "std_delta": float(np.std(list(deltas.values()), ddof=1))
            if len(deltas) > 1 else 0.0,
            "n_bs": len(deltas),
        },
        "computed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--parquet", type=Path,
        default=Path("/home/thc1006/dev/colosseum-oran-federated-slicing/data/coloran_raw_unified.parquet"),
    )
    ap.add_argument(
        "--cells", type=str, required=True,
        help="Comma-separated 'arch:s<seed>' cells, e.g. 'lstm:s0,lstm:s42,mamba:s0'",
    )
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--finetune-steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--mixed-precision", type=str, default="fp16",
                    choices=["fp16", "bf16", "fp32"])
    ap.add_argument(
        "--out", type=Path,
        default=Path("artifacts/r2_post_hoc_per_bs_finetune"),
    )
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if not args.parquet.exists():
        print(f"parquet not found: {args.parquet}", file=sys.stderr)
        return 1

    device = torch.device(args.device)

    cell_specs = []
    for spec in args.cells.split(","):
        spec = spec.strip()
        if not spec:
            continue
        arch, seed_part = spec.split(":")
        if not seed_part.startswith("s"):
            raise ValueError(f"cell spec must be 'arch:s<seed>', got {spec!r}")
        seed = int(seed_part[1:])
        cell_specs.append((arch, seed))

    n_done = 0
    n_failed = 0
    for arch, seed in cell_specs:
        cell_id = f"{arch}_s{seed}"
        out_path = args.out / f"{cell_id}.json"
        if out_path.exists():
            print(f"[skip] {cell_id} (already at {out_path})")
            continue
        try:
            t0 = time.time()
            payload = run_one_cell(
                arch=arch, seed=seed, parquet_path=args.parquet,
                finetune_steps=args.finetune_steps, batch_size=args.batch_size,
                lr=args.lr, device=device, mixed_precision=args.mixed_precision,
            )
            payload["wall_time_s"] = time.time() - t0
            out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
            print(f"[done] {cell_id} → {out_path}  wall={payload['wall_time_s']:.1f}s "
                  f"mean_Δ={payload['summary']['mean_delta']:+.4f}")
            n_done += 1
        except Exception as e:
            print(f"[FAIL] {cell_id}: {e!r}")
            n_failed += 1

    print(f"\nFinished: {n_done} done, {n_failed} failed.")
    return 0 if n_failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
