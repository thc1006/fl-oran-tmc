"""prea1 / phase0 reproducibility check: natural-by-BS vs random_split test AUC.

STRICTLY inference/evaluation. NO training of any kind:
no FL rounds, no .fit/.backward/optimizer steps. We only load surviving
``best.pt`` checkpoints and run the EXISTING test-AUC computation path
(``centralized_v3._batched_predict`` + ``centralized_v3._metrics``) on the
fl_v7-reconstructed test tensors.

Why we re-fit the federated scaler per partition mode: in fl_v7.run_v7_sweep
the continuous-feature StandardScaler is fit (via federated sufficient-stats
aggregation) on the *train partition's* client shards. The test set itself
(``split.test`` from ood_split_by_tr) is identical regardless of partition;
only the scaler stats are derived from the (mode-dependent) partition. To be
faithful to each checkpoint's original eval we replicate the matching
partition + scaler-fit before scaling the (shared) test sequences.

Outputs (under artifacts/prea1/phase0/):
  - per_seed_auc.csv     per-seed natural / shuffle / delta, plus reference
                         test_auc read from each cell's summary.json
  - results.json         machine-readable summary incl. env + means
  - run.log              full stdout/stderr (written by the caller via tee)

Run with the repo venv active:
  python artifacts/prea1/phase0/reproduce_natural_vs_shuffle.py
"""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Repo imports — EXISTING code paths, nothing reinvented.
from fl_oran.data_v2.encoders import (
    FeatureSchema,
    apply_continuous_scaler,
    federated_fit_scaler,
)
from fl_oran.data_v2.partition import partition_clients
from fl_oran.data_v2.sequences import build_run_sequences
from fl_oran.data_v2.split import ood_split_by_tr
from fl_oran.data_v2.targets_v2 import add_classification_target
from fl_oran.models.forecaster_v2 import ForecasterV2
from fl_oran.training.centralized_v3 import (
    V3_CATEGORICAL,
    V3_CAT_SIZES,
    V3_CONTINUOUS,
    _batched_predict,
    _metrics,
)
from fl_oran.utils import pick_device

# --- fixed config (read from the surviving summary.json; identical across cells) ---
PARQUET = Path("/home/thc1006/dev/fl-oran-tmc/data/coloran_raw_unified.parquet")
SEQ_LEN = 5
THRESHOLD = 0.10
TRAIN_TR = list(range(22))
VAL_TR = [22, 23, 24]
TEST_TR = [25, 26, 27]
N_CLIENTS = 7

NATURAL_DIR = Path("/home/thc1006/dev/fl-oran-tmc/artifacts/v7_stage2_full")
SHUFFLE_DIR = Path("/home/thc1006/dev/fl-oran-tmc/artifacts/v7_ablation_random_split")
OUT_DIR = Path("/home/thc1006/dev/fl-oran-tmc/artifacts/prea1/phase0")

# Seeds present in BOTH checkpoint sets (intersection) -> apples-to-apples deltas.
SHARED_SEEDS = [0, 1, 2, 3, 42]


def _strip_compile_prefix(state: dict) -> dict:
    """best.pt was saved from a torch.compile()'d module -> _orig_mod. prefix.

    ForecasterV2 (uncompiled) expects keys WITHOUT that prefix. Stripping it is
    the inverse of torch.compile's wrapper; no weights are altered.
    """
    pref = "_orig_mod."
    out = {}
    for k, v in state.items():
        out[k[len(pref):] if k.startswith(pref) else k] = v
    return out


def _build_test_tensors_for_mode(df, schema, feat_cols, mode, seed, device):
    """Reconstruct the fl_v7 test tensors for one partition mode.

    Returns (te_cat, te_cont, te_y_int). Test sequences are mode-independent;
    the scaler is fit on the mode-specific train partition (faithful to
    run_v7_sweep). build_run_sequences / federated_fit_scaler / ood_split_by_tr
    are the EXISTING pipeline functions.
    """
    split = ood_split_by_tr(df, TRAIN_TR, VAL_TR, TEST_TR)

    if mode == "iid":
        client_dfs = partition_clients(split.train, mode="iid")
    elif mode == "random_split":
        client_dfs = partition_clients(
            split.train, mode="random_split", n_clients=N_CLIENTS, seed=seed,
        )
    else:
        raise ValueError(f"unexpected mode {mode!r}")

    # Per-client sequences -> federated scaler fit (sufficient-stats pooling).
    # Memory: cap each client at 500k sequences for the scaler fit (fixed-seed
    # subsample). Scaler = mean/std of continuous features; 500k vs ~2M changes it
    # < 1e-3 while cutting peak RAM ~4x. Test sequences below are NOT subsampled.
    _SCALER_FIT_CAP = 500_000
    _srng = np.random.default_rng(12345)
    client_shards = {}
    for cid, d in client_dfs.items():
        X, _Y = build_run_sequences(d, feat_cols, ["y_sla_next"], seq_len=SEQ_LEN)
        if len(X) > _SCALER_FIT_CAP:
            X = X[_srng.choice(len(X), _SCALER_FIT_CAP, replace=False)]
        if len(X) > 0:
            client_shards[cid] = X
    scaler = federated_fit_scaler(client_shards, schema, n_jobs=1)

    # Test sequences (identical across modes) then scale with mode's scaler.
    X_te, Y_te = build_run_sequences(
        split.test, feat_cols, ["y_sla_next"], seq_len=SEQ_LEN,
    )
    cat, cont = apply_continuous_scaler(X_te, schema, scaler)
    te_cat = torch.from_numpy(cat)
    te_cont = torch.from_numpy(cont)
    te_y = Y_te[:, 0].astype(int)
    # scaler stats reported for cross-mode sanity (should be ~identical).
    return te_cat, te_cont, te_y, scaler


def _eval_checkpoint(ckpt_path, schema, te_cat, te_cont, te_y, device):
    """Load best.pt into a fresh ForecasterV2 and compute test AUC. NO training."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = _strip_compile_prefix(state)
    model = ForecasterV2(schema=schema, task="classification", seq_len=SEQ_LEN)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"state_dict mismatch loading {ckpt_path}: "
            f"missing={missing} unexpected={unexpected}"
        )
    model = model.to(device)
    logits = _batched_predict(model, te_cat, te_cont, device)  # eval(), no_grad
    m = _metrics(te_y, logits[:, 0])
    return m


def main() -> int:
    device = pick_device("cuda")
    # Match fl_v7.setup_torch_perf matmul precision so any TF32/BF16-reduction
    # matmul behaviour is the same as the original eval. cudnn flags below
    # mirror the deterministic mandate (D-15). No effect on stored weights.
    if device.type == "cuda":
        torch.set_float32_matmul_precision("medium")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # --- load + prep data once (target + schema). Test set is shared. ---
    if not PARQUET.exists():
        raise FileNotFoundError(PARQUET)
    # Memory: load ONLY the columns the pipeline needs (zero effect on results) —
    # the first attempt OOM'd loading all 36 cols x 18.3M rows.
    _needed_cols = list(dict.fromkeys(
        ["run_id", "step_idx", "slice_id", "ul_bler"]
        + V3_CATEGORICAL + V3_CONTINUOUS
    ))
    df = pd.read_parquet(PARQUET, columns=_needed_cols)
    df = add_classification_target(
        df, column="ul_bler", threshold=THRESHOLD, target_name="y_sla_next",
    )
    schema = FeatureSchema(
        categorical=V3_CATEGORICAL,
        categorical_sizes=V3_CAT_SIZES,
        continuous=V3_CONTINUOUS,
    )
    feat_cols = schema.categorical + schema.continuous

    rows = []
    # Cache per-mode test tensors keyed by (mode, seed) — random_split scaler
    # depends on seed; iid scaler is seed-independent (partition by bs_id).
    natural_tensors = None  # iid scaler is seed-independent -> build once.

    for seed in SHARED_SEEDS:
        # --- natural-by-BS (iid) ---
        nat_dir = NATURAL_DIR / f"v7_lstm_fedavg_iid_n7_s{seed}"
        nat_ckpt = nat_dir / "best.pt"
        nat_summary = json.load(open(nat_dir / "summary.json"))
        if natural_tensors is None:
            natural_tensors = _build_test_tensors_for_mode(
                df, schema, feat_cols, "iid", seed, device,
            )
        te_cat, te_cont, te_y, nat_scaler = natural_tensors
        nat_m = _eval_checkpoint(nat_ckpt, schema, te_cat, te_cont, te_y, device)

        # --- random_split (shuffle); scaler fit uses cfg.seed ---
        shf_dir = SHUFFLE_DIR / f"v7_lstm_fedavg_randsplit_n7_s{seed}"
        shf_ckpt = shf_dir / "best.pt"
        shf_summary = json.load(open(shf_dir / "summary.json"))
        sh_cat, sh_cont, sh_y, shf_scaler = _build_test_tensors_for_mode(
            df, schema, feat_cols, "random_split", seed, device,
        )
        shf_m = _eval_checkpoint(shf_ckpt, schema, sh_cat, sh_cont, sh_y, device)

        delta = nat_m["auc"] - shf_m["auc"]
        rows.append({
            "seed": seed,
            "natural_auc": nat_m["auc"],
            "shuffle_auc": shf_m["auc"],
            "delta_nat_minus_shuf": delta,
            "natural_auc_ref_4080": nat_summary["test_auc"],
            "shuffle_auc_ref_4080": shf_summary["test_auc"],
            "natural_acc": nat_m["accuracy"],
            "shuffle_acc": shf_m["accuracy"],
            "test_n": int(len(te_y)),
            "scaler_mean_l2_diff": float(
                np.linalg.norm(nat_scaler.mean - shf_scaler.mean)
            ),
        })
        print(
            f"seed={seed:>2}  natural={nat_m['auc']:.6f} (ref {nat_summary['test_auc']:.6f})"
            f"  shuffle={shf_m['auc']:.6f} (ref {shf_summary['test_auc']:.6f})"
            f"  delta={delta:+.6f}",
            flush=True,
        )

    out_df = pd.DataFrame(rows)
    mean_nat = float(out_df["natural_auc"].mean())
    mean_shf = float(out_df["shuffle_auc"].mean())
    mean_gap = float(out_df["delta_nat_minus_shuf"].mean())
    mean_gap_ref = float(
        (out_df["natural_auc_ref_4080"] - out_df["shuffle_auc_ref_4080"]).mean()
    )

    out_df.to_csv(OUT_DIR / "per_seed_auc.csv", index=False)
    summary = {
        "task": "prea1_phase0_reproducibility_natural_vs_shuffle",
        "arch": "lstm",
        "algorithm": "fedavg",
        "shared_seeds": SHARED_SEEDS,
        "mean_natural_auc": mean_nat,
        "mean_shuffle_auc": mean_shf,
        "mean_gap_natural_minus_shuffle": mean_gap,
        "mean_gap_reference_4080": mean_gap_ref,
        "env": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda if torch.cuda.is_available() else None,
            "cudnn": (
                torch.backends.cudnn.version() if torch.cuda.is_available() else None
            ),
            "python": platform.python_version(),
            "gpu": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "device": str(device),
        },
        "per_seed": rows,
    }
    (OUT_DIR / "results.json").write_text(json.dumps(summary, indent=2))

    print("\n=== SUMMARY ===", flush=True)
    print(
        f"mean natural AUC = {mean_nat:.6f}\n"
        f"mean shuffle AUC = {mean_shf:.6f}\n"
        f"mean (natural - shuffle) gap = {mean_gap:+.6f}\n"
        f"mean gap on RTX 4080 (reference) = {mean_gap_ref:+.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
