"""Multi-cell v5 sweep driver — shares data prep across the algorithm row.

For full TMC matrices (6 algorithms x N alphas x M seeds = up to 150 cells)
the single-cell CLI wastes ~55 s per cell re-reading + re-sequencing the
same ColO-RAN parquet. This driver batches a row of the matrix into a
single Python process so that ``prepare_v5_data`` runs once per
(seed, alpha) combination and feeds ``_run_training`` for every
algorithm.

Example (pilot row — 6 algorithms, 1 seed, 1 alpha):

    python experiments/run_v5_sweep_matrix.py \\
        --seeds 42 --alphas 0.5 --n-clients 5 \\
        --num-rounds 5 --max-steps-per-round 50 --batch-size 256 \\
        --lr 5e-4 --sample-ratio 1.0 --seq-len 5 \\
        --device cuda --mixed-precision bf16 --compile-model reduce-overhead \\
        --algo-spec 'fedavg:{}' \\
        --algo-spec 'fedprox:{"mu": 0.01}' \\
        --algo-spec 'fedadam:{"server_lr": 0.01}' \\
        --algo-spec 'scaffold:{}' \\
        --algo-spec 'feddyn:{"alpha": 0.01}' \\
        --algo-spec 'moon:{"mu": 1.0, "tau": 0.5}'

Each cell writes its own ``artifacts/v5_sweep/<auto_name>/`` directory
exactly as the single-cell CLI does; a top-level ``_matrix_summary.csv``
is also emitted alongside them.
"""
from __future__ import annotations

import argparse
import csv
import json
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
    p = argparse.ArgumentParser(
        description=("Matrix driver for v5 sweeps. Shares data prep across a "
                     "row (fixed seed/alpha, varying algorithm)."),
    )
    p.add_argument("--algo-spec", action="append", required=True,
                   help=("Algorithm specification in the form "
                         "'name:<json_kwargs>'. Supply once per algorithm; "
                         "may be repeated."))
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--alphas", type=float, nargs="+", default=[0.5])
    p.add_argument("--partition-mode", default="dirichlet",
                   choices=["dirichlet", "iid"])
    p.add_argument("--n-clients", type=int, default=5)
    p.add_argument("--num-rounds", type=int, default=20)
    p.add_argument("--clients-per-round", type=int, default=5)
    p.add_argument("--max-steps-per-round", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--lr-warmup-rounds", type=int, default=3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--unified-parquet",
                   default="data/coloran_raw_unified.parquet")
    p.add_argument("--sample-ratio", type=float, default=1.0)
    p.add_argument("--threshold", type=float, default=0.10)
    p.add_argument("--seq-len", type=int, default=5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--mixed-precision", default="bf16")
    p.add_argument("--compile-model", default=None,
                   choices=[None, "default", "reduce-overhead", "max-autotune"])
    p.add_argument("--output-dir", default="artifacts/v5_sweep")
    return p.parse_args()


def _parse_algo_spec(spec: str) -> tuple[str, dict]:
    """Parse 'algoname:{json...}' into ``(name, kwargs)``."""
    if ":" not in spec:
        raise ValueError(f"algo-spec must be 'name:<json>': got {spec!r}")
    name, raw = spec.split(":", 1)
    kwargs = json.loads(raw) if raw.strip() else {}
    return name.strip(), kwargs


def _base_cfg_for(seed: int, alpha: float, args: argparse.Namespace) -> V5Config:
    """V5Config carrying only shared fields — algorithm is filled per-cell."""
    return V5Config(
        algorithm="fedavg",  # placeholder; per-cell override below
        algo_kwargs={},
        partition_mode=args.partition_mode,
        alpha=alpha,
        n_clients=args.n_clients,
        num_rounds=args.num_rounds,
        clients_per_round=args.clients_per_round,
        max_steps_per_round=args.max_steps_per_round,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_warmup_rounds=args.lr_warmup_rounds,
        grad_clip=args.grad_clip,
        unified_parquet=Path(args.unified_parquet),
        sample_ratio=args.sample_ratio,
        threshold=args.threshold,
        seq_len=args.seq_len,
        seed=seed,
        device=args.device,
        mixed_precision=args.mixed_precision,
        compile_model=args.compile_model,
        output_dir=Path(args.output_dir),
    )


def main() -> None:
    args = _parse_args()
    specs = [_parse_algo_spec(s) for s in args.algo_spec]

    # One-time: device + perf switches (shared across all cells).
    device = pick_device(args.device)
    log_cuda_info(device)
    # Matrix driver inherits deterministic from the first cell's default (True).
    setup_torch_perf(device, deterministic=True)
    amp_enabled, amp_dtype = autocast_dtype(args.mixed_precision)

    summary_rows: list[dict] = []
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Timestamped filename so repeated sweeps don't clobber each other.
    ts = time.strftime("%Y%m%d_%H%M%S")
    summary_path = out_dir / f"_matrix_summary_{ts}.csv"
    latest_link = out_dir / "_matrix_summary_latest.csv"

    t_matrix_start = time.time()
    # Build (seed, alpha)-invariant parts ONCE. With sample_ratio=1.0 this
    # is truly shared across the entire matrix. With sample_ratio<1.0 the
    # parquet sample is seed-dependent, so we warn and fall back to
    # per-combo prep inside the loop.
    base_cfg_for_shared = _base_cfg_for(args.seeds[0], args.alphas[0], args)
    use_shared = args.sample_ratio >= 1.0
    if use_shared:
        log.info("=== building shared splits (one-time) ===")
        seed_everything(args.seeds[0])
        shared = prepare_shared_splits(base_cfg_for_shared)
    else:
        log.warning("sample_ratio=%.3f < 1.0 — disabling SharedSplits "
                    "(sample is seed-dependent)", args.sample_ratio)
        shared = None

    for seed in args.seeds:
        for alpha in args.alphas:
            # Re-seed per (seed, alpha) so partition + scaler are deterministic.
            seed_everything(seed)
            base = _base_cfg_for(seed, alpha, args)
            log.info("=== per-combo prep for seed=%d alpha=%.3f ===", seed, alpha)
            t_prep = time.time()
            data = prepare_v5_data(base, device, shared=shared)
            log.info("per-combo prep took %.1fs", time.time() - t_prep)

            for algo_name, algo_kwargs in specs:
                # Re-seed before each training cell so that two cells with
                # the same algorithm/alpha/seed are reproducible.
                seed_everything(seed)
                cfg = _base_cfg_for(seed, alpha, args)
                cfg.algorithm = algo_name
                cfg.algo_kwargs = algo_kwargs
                cfg.name = ""
                cfg.__post_init__()  # regenerate auto-name
                log.info("--- cell: algo=%s alpha=%.3f seed=%d ---",
                         algo_name, alpha, seed)
                t_cell = time.time()
                result = _run_training(cfg, data, device, amp_enabled, amp_dtype)
                dt = time.time() - t_cell
                summary_rows.append({
                    "algorithm": algo_name,
                    "alpha": alpha,
                    "seed": seed,
                    "test_auc": result["test"].get("auc", float("nan")),
                    "test_acc": result["test"]["accuracy"],
                    "test_f1": result["test"]["f1"],
                    "best_val_auc": result["best_val_auc"],
                    "duration_s": round(dt, 2),
                })

            # Free PreparedData tensors before next (seed, alpha) iteration
            # so we don't pile up gigabytes of pinned memory.
            del data
            if device.type == "cuda":
                torch.cuda.empty_cache()

    with summary_path.open("w", newline="") as fp:
        w = csv.DictWriter(
            fp, fieldnames=list(summary_rows[0].keys()) if summary_rows else [
                "algorithm", "alpha", "seed", "test_auc", "test_acc",
                "test_f1", "best_val_auc", "duration_s",
            ],
        )
        w.writeheader()
        w.writerows(summary_rows)
    # Maintain a stable "latest" pointer for scripts that want the most
    # recent sweep without knowing the timestamp.
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    try:
        latest_link.symlink_to(summary_path.name)
    except OSError:
        # Fallback: plain copy on filesystems that don't support symlinks.
        latest_link.write_text(summary_path.read_text())
    log.info("matrix done in %.1fs  cells=%d  summary=%s",
             time.time() - t_matrix_start, len(summary_rows), summary_path)


if __name__ == "__main__":
    main()
