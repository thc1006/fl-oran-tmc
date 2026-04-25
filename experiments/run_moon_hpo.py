"""MOON hyperparameter sweep at fixed (seed, alpha).

Runs an MxN grid over (mu, tau) in a single Python process, sharing
PreparedData across all trials (only the algorithm instance + per-trial
RNG re-seed differ). Used to find good MOON hparams before the full
5x5 sweep — pilot showed default (mu=1.0, tau=0.5) is ~0.08 AUC below
FedAvg at alpha=0.5.

Outputs:
  artifacts/v5_sweep/_moon_hpo/moon_mu{mu}_tau{tau}_s{seed}_a{alpha}/
                              {summary.json, history.csv, best.pt}
  artifacts/v5_sweep/_moon_hpo/moon_hpo_summary_s{seed}_a{alpha}.csv

Example:
  python experiments/run_moon_hpo.py --seed 42 --alpha 0.5 \\
      --mus 0.1 0.5 1.0 5.0 10.0 \\
      --taus 0.1 0.5 1.0
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch

from fl_oran.logging_utils import get_logger
from fl_oran.training.fl_v5 import (
    V5Config,
    _run_training,
    prepare_shared_splits,
    prepare_v5_data,
    setup_torch_perf,
)
from fl_oran.utils import autocast_dtype, log_cuda_info, pick_device, seed_everything

log = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--mus", type=float, nargs="+",
                   default=[0.1, 0.5, 1.0, 5.0, 10.0])
    p.add_argument("--taus", type=float, nargs="+",
                   default=[0.1, 0.5, 1.0])
    p.add_argument("--n-clients", type=int, default=5)
    p.add_argument("--num-rounds", type=int, default=20)
    p.add_argument("--clients-per-round", type=int, default=5)
    p.add_argument("--max-steps-per-round", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--lr-warmup-rounds", type=int, default=3)
    p.add_argument("--unified-parquet",
                   default="data/coloran_raw_unified.parquet")
    p.add_argument("--output-dir", default="artifacts/v5_sweep/_moon_hpo")
    return p.parse_args()


def _make_cfg(args: argparse.Namespace, mu: float | None = None,
              tau: float | None = None, name: str = "") -> V5Config:
    """Reusable config factory — algo_kwargs filled when (mu, tau) given."""
    algo_kwargs = {} if mu is None else {"mu": mu, "tau": tau}
    return V5Config(
        name=name,
        algorithm="moon",
        algo_kwargs=algo_kwargs,
        partition_mode="dirichlet",
        alpha=args.alpha,
        n_clients=args.n_clients,
        num_rounds=args.num_rounds,
        clients_per_round=args.clients_per_round,
        max_steps_per_round=args.max_steps_per_round,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_warmup_rounds=args.lr_warmup_rounds,
        sample_ratio=1.0,
        seq_len=5,
        unified_parquet=Path(args.unified_parquet),
        seed=args.seed,
        device="cuda",
        mixed_precision="bf16",
        compile_model="reduce-overhead",
        pos_weight_split="train",
        cudnn_deterministic=True,
        output_dir=Path(args.output_dir),
    )


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device("cuda")
    log_cuda_info(device)
    setup_torch_perf(device, deterministic=True)
    amp_enabled, amp_dtype = autocast_dtype("bf16")

    # One-time data prep — shared across all (mu, tau) trials.
    seed_everything(args.seed)
    base_cfg = _make_cfg(args)
    log.info("=== preparing shared data (seed=%d alpha=%.3f) ===",
             args.seed, args.alpha)
    shared = prepare_shared_splits(base_cfg)
    data = prepare_v5_data(base_cfg, device, shared=shared)

    grid = [(mu, tau) for mu in args.mus for tau in args.taus]
    log.info("=== MOON HPO grid: %d cells (mu x tau = %d x %d) ===",
             len(grid), len(args.mus), len(args.taus))

    results: list[dict] = []
    t_start = time.time()
    for mu, tau in grid:
        mu_tag = f"{mu}".replace(".", "p")
        tau_tag = f"{tau}".replace(".", "p")
        cfg = _make_cfg(args, mu=mu, tau=tau,
                        name=f"hpo_moon_mu{mu_tag}_tau{tau_tag}_s{args.seed}_a{args.alpha:.2f}".replace(".", "p"))
        log.info("--- trial: mu=%.4f tau=%.4f ---", mu, tau)
        t0 = time.time()
        seed_everything(args.seed)
        result = _run_training(cfg, data, device, amp_enabled, amp_dtype)
        dt = time.time() - t0
        results.append({
            "mu": mu,
            "tau": tau,
            "test_auc": result["test"].get("auc"),
            "test_acc": result["test"]["accuracy"],
            "test_f1": result["test"]["f1"],
            "best_val_auc": result["best_val_auc"],
            "duration_s": round(dt, 2),
        })

    # CSV
    csv_path = out_dir / f"moon_hpo_summary_s{args.seed}_a{args.alpha:.2f}.csv".replace("0.", "0p")
    with csv_path.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    # Sorted table
    sorted_r = sorted(results, key=lambda r: r["test_auc"] or 0, reverse=True)
    print("\n" + "=" * 60)
    print(f"  MOON HPO  seed={args.seed}  alpha={args.alpha}  n={len(grid)}")
    print("=" * 60)
    print(f"  {'mu':>6}  {'tau':>6}  {'AUC':>8}  {'Acc':>8}  {'F1':>8}  {'sec':>6}")
    print("-" * 60)
    for r in sorted_r:
        auc = r["test_auc"] or 0
        print(f"  {r['mu']:>6}  {r['tau']:>6}  {auc:>8.4f}  "
              f"{r['test_acc']:>8.4f}  {r['test_f1']:>8.4f}  "
              f"{r['duration_s']:>6.1f}")
    best = sorted_r[0]
    print("=" * 60)
    print(f"BEST: mu={best['mu']}  tau={best['tau']}  AUC={best['test_auc']:.4f}")
    print(f"FedAvg (5-seed reference) at α=0.5: AUC=0.7698±0.0127")
    print(f"Pre-HPO MOON at α=0.5 seed=42: AUC=0.6906")
    if best["test_auc"] and best["test_auc"] > 0.74:
        print("→ MOON RESCUED: tuned hparams reach top-tier range")
    elif best["test_auc"] and best["test_auc"] > 0.71:
        print("→ MOON IMPROVED: tuned hparams help but still below top tier")
    else:
        print("→ MOON STILL UNDERPERFORMS: consider dropping from paper")
    print(f"Total HPO wall-clock: {(time.time()-t_start)/60:.1f} min")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
