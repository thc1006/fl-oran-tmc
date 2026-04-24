"""CLI wrapper for a single v5 sweep cell.

Usage:

    python experiments/run_v5_algorithm_sweep.py \\
        --algorithm fedprox --algo-kwargs '{"mu": 0.01}' \\
        --alpha 0.5 --n-clients 5 --seed 42 \\
        --num-rounds 30 --sample-ratio 0.2

Each invocation runs ONE cell of the paper matrix (one algorithm, one
alpha, one seed). An outer driver (shell script or another Python file)
is responsible for sweeping over the 6 x N_alpha x N_seeds cells. This
separation keeps CLI complexity low and composes cleanly with job-array
schedulers.

Training is only triggered when this script is invoked directly — the
library function ``fl_oran.training.fl_v5.run_v5_sweep`` is
import-safe and has no module-level side effects.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fl_oran.training.fl_v5 import V5Config, run_v5_sweep


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=("Run one (algorithm, alpha, seed) cell of the v5 sweep. "
                     "Writes artifacts/v5_sweep/<name>/{summary.json, history.csv, best.pt}."),
    )
    # Algorithm.
    p.add_argument("--algorithm", required=True,
                   choices=["fedavg", "fedprox", "fedadam",
                            "scaffold", "feddyn", "moon"])
    p.add_argument("--algo-kwargs", default="{}",
                   help='JSON dict of algorithm-specific kwargs, e.g. \'{"mu": 0.01}\'')
    # Partition.
    p.add_argument("--partition-mode", default="dirichlet",
                   choices=["dirichlet", "iid"])
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Dirichlet concentration (ignored if partition-mode=iid)")
    p.add_argument("--n-clients", type=int, default=5)
    # Training.
    p.add_argument("--num-rounds", type=int, default=20)
    p.add_argument("--clients-per-round", type=int, default=5)
    p.add_argument("--max-steps-per-round", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--lr-warmup-rounds", type=int, default=3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    # Data.
    p.add_argument("--unified-parquet",
                   default="data/coloran_raw_unified.parquet")
    p.add_argument("--sample-ratio", type=float, default=1.0)
    p.add_argument("--threshold", type=float, default=0.10)
    p.add_argument("--seq-len", type=int, default=5)
    # System.
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--mixed-precision", default="bf16")
    p.add_argument("--compile-model", default=None,
                   choices=[None, "default", "reduce-overhead", "max-autotune"],
                   help=("torch.compile mode for local_model (CUDA only). "
                         "None=eager. 'reduce-overhead' uses CUDA graphs "
                         "— fastest for small static-shape models."))
    p.add_argument("--pos-weight-split", default="train",
                   choices=["train", "test"],
                   help="Which split drives pos_weight for BCEWithLogitsLoss.")
    p.add_argument("--cudnn-nondeterministic", action="store_true",
                   help="Disable cudnn.deterministic (recover ~5-15% speed).")
    p.add_argument("--output-dir", default="artifacts/v5_sweep")
    p.add_argument("--name", default="",
                   help="Run name (auto-generated if empty)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    algo_kwargs = json.loads(args.algo_kwargs)
    cfg = V5Config(
        name=args.name,
        algorithm=args.algorithm,
        algo_kwargs=algo_kwargs,
        partition_mode=args.partition_mode,
        alpha=args.alpha,
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
        seed=args.seed,
        device=args.device,
        mixed_precision=args.mixed_precision,
        compile_model=args.compile_model,
        pos_weight_split=args.pos_weight_split,
        cudnn_deterministic=not args.cudnn_nondeterministic,
        output_dir=Path(args.output_dir),
    )
    run_v5_sweep(cfg)


if __name__ == "__main__":
    main()
