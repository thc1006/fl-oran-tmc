"""CLI wrapper for a single v7 sweep cell (Phase 1.5c per ADR D-22).

Usage::

    python experiments/run_v7_fl_arch_sweep.py \\
        --arch spiking_expand2 --algorithm fedprox \\
        --algo-kwargs '{"mu": 0.01}' \\
        --alpha 0.5 --n-clients 7 --seed 42 \\
        --num-rounds 100 --sample-ratio 1.0

Each invocation runs ONE cell of the Stage 2 paper matrix
(one arch, one algorithm, one alpha, one seed). For multi-cell sweeps
with shared splits caching see ``experiments/run_v7_fl_arch_sweep_matrix.py``
(Phase 1.5e) — that driver gives ~4× speedup vs sequential single-cell
calls of this wrapper.

Library function ``fl_oran.training.fl_v7.run_v7_sweep`` is import-safe
and has no module-level side effects; training only triggers when this
script is invoked directly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fl_oran.training.fl_v7 import V7Config, run_v7_sweep


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run one (arch, algorithm, alpha, seed) cell of the v7 sweep. "
            "Writes artifacts/v7_fl_arch_sweep/<name>/{summary.json, "
            "history.csv, best.pt}."
        ),
    )
    # Architecture (NEW vs v5 CLI).
    p.add_argument(
        "--arch", required=True,
        choices=["lstm", "mamba", "mamba_expand2", "spiking", "spiking_expand2"],
        help="Architecture key into ARCH_REGISTRY (run_v6_arch_sweep.py).",
    )
    p.add_argument(
        "--arch-kwargs", default="{}",
        help='JSON dict of arch-specific kwargs override, e.g. \'{"dropout": 0.1}\'',
    )
    # Algorithm (MOON not supported in Phase 1.5; D-16 deferred).
    p.add_argument(
        "--algorithm", required=True,
        choices=["fedavg", "fedprox", "fedadam", "scaffold", "feddyn"],
        help="FL algorithm. MOON deferred to Phase 2 polish (D-16).",
    )
    p.add_argument(
        "--algo-kwargs", default="{}",
        help='JSON dict of algorithm-specific kwargs, e.g. \'{"mu": 0.01}\'',
    )
    # Partition.
    p.add_argument(
        "--partition-mode", default="dirichlet",
        choices=["dirichlet", "iid"],
    )
    p.add_argument(
        "--alpha", type=float, default=0.5,
        help="Dirichlet concentration (ignored if partition-mode=iid)",
    )
    p.add_argument(
        "--n-clients", type=int, default=7,
        help="Number of FL clients (ignored in iid mode — uses bs_id partition).",
    )
    # Training.
    p.add_argument("--num-rounds", type=int, default=100)
    p.add_argument("--clients-per-round", type=int, default=5)
    p.add_argument("--max-steps-per-round", type=int, default=36)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--lr-warmup-rounds", type=int, default=3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    # Data.
    p.add_argument(
        "--unified-parquet", default="data/coloran_raw_unified.parquet",
    )
    p.add_argument("--sample-ratio", type=float, default=1.0)
    p.add_argument("--threshold", type=float, default=0.10)
    p.add_argument("--seq-len", type=int, default=5)
    # System (perf inheritance from M5).
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--mixed-precision", default="bf16")
    p.add_argument(
        "--compile-model", default=None,
        choices=[None, "default", "reduce-overhead", "max-autotune"],
        help=(
            "torch.compile mode override. None lets fl_v7 pick "
            "arch-conditional default (None for spiking, 'reduce-overhead' "
            "for dense)."
        ),
    )
    p.add_argument(
        "--pos-weight-split", default="train", choices=["train", "test"],
        help="D-12 contract: train (default) avoids leakage; test matches v4.",
    )
    p.add_argument(
        "--cudnn-nondeterministic", action="store_true",
        help="Disable cudnn.deterministic (recover ~5-15% speed; off by default per D-15).",
    )
    p.add_argument("--output-dir", default="artifacts/v7_fl_arch_sweep")
    p.add_argument(
        "--name", default="",
        help="Run name (auto-generated as v7_<arch>_<algo>_a<alpha>_s<seed> if empty)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    algo_kwargs = json.loads(args.algo_kwargs)
    arch_kwargs = json.loads(args.arch_kwargs)
    cfg = V7Config(
        name=args.name,
        arch=args.arch,
        arch_kwargs=arch_kwargs,
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
    run_v7_sweep(cfg)


if __name__ == "__main__":
    main()
