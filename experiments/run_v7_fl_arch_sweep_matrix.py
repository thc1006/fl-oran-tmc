"""Multi-cell v7 sweep driver — runs an arch × algo × alpha × seed matrix.

Phase 1.5e per ADR D-22. For full Stage 2 sweep (3 archs × 7 algos × 5
alphas × 10 seeds = 1050 cells, ~14-26 hr GPU) the single-cell CLI
``run_v7_fl_arch_sweep.py`` would orchestrate one Python process per
cell — wasteful boot time. This driver batches a configurable matrix
into a single process.

Example (Phase 2 minimum-viable smoke per ADR D-22 — 36 cells):

    python experiments/run_v7_fl_arch_sweep_matrix.py \\
        --archs lstm,mamba,spiking_expand2 \\
        --algo-spec 'fedavg:{}' --algo-spec 'fedprox:{"mu": 0.01}' \\
        --alphas 0.5 \\
        --seeds 42 0 1 \\
        --partition-mode dirichlet \\
        --num-rounds 100 --max-steps-per-round 36 \\
        --sample-ratio 1.0 --device cuda

Each cell writes its own ``artifacts/v7_fl_arch_sweep/<auto_name>/``
directory. A top-level ``_matrix_summary_<timestamp>.csv`` is emitted
under ``--output-dir`` for cross-cell comparison.

**Performance note (deferred SharedSplits refactor)**: this driver
runs cells SEQUENTIALLY, calling ``run_v7_sweep`` per cell. On single
GPU we can't parallelize the training itself; the M5-style 4× speedup
from ``SharedSplits`` (sharing parquet read + OOD split + val/test
sequences across cells of the same (seed, alpha)) is NOT applied here
to keep Phase 1.5e under the ADR D-22 ~1 hr budget. For 36-cell smoke
the missed saving is ~7.5 min / 180 min = ~4% — acceptable. For
ADR's full 1050-cell Stage 2 sweep the missed saving is ~18% (~2.5 hr
out of 14 hr) — at that scale, refactor fl_v7 to expose
``prepare_shared_splits_v7`` + ``prepare_v7_data`` + ``_run_training_v7``
mirroring fl_v5's split, then matrix driver caches SharedSplits.
"""
from __future__ import annotations

# Performance env vars (must be set BEFORE torch import). Per ADR D-22
# perf checklist: expandable_segments lets the CUDA caching allocator
# return memory back to the OS to prevent fragmentation across long
# multi-cell sweeps. OMP/MKL_NUM_THREADS prevents PyTorch CPU ops from
# oversubscribing when joblib is also threading inside
# federated_fit_scaler / build_run_sequences.
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import argparse
import csv
import json
import time
import traceback
from itertools import product
from pathlib import Path

import torch

from fl_oran.logging_utils import get_logger
from fl_oran.training.fl_v7 import V7Config, run_v7_sweep

log = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Matrix driver for v7 FL × architecture sweeps. Runs the "
            "Cartesian product of --archs × --algo-spec × --alphas × "
            "--seeds sequentially on a single GPU."
        ),
    )
    p.add_argument(
        "--archs", required=True,
        help=("Comma-separated arch keys, e.g. "
              "'lstm,mamba,spiking_expand2'."),
    )
    p.add_argument(
        "--algo-spec", action="append", required=True,
        help=("Algorithm specification 'name:<json_kwargs>'. Repeatable. "
              "e.g. --algo-spec 'fedavg:{}' --algo-spec 'fedprox:{\"mu\": 0.01}'"),
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--alphas", type=float, nargs="+", default=[0.5])
    p.add_argument(
        "--partition-mode", default="dirichlet",
        choices=["dirichlet", "iid"],
    )
    p.add_argument(
        "--n-clients", type=int, default=7,
        help="Ignored under iid mode (uses bs_id = 7 ColO-RAN gNBs).",
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
    # System.
    p.add_argument("--device", default="cuda")
    p.add_argument("--mixed-precision", default="bf16")
    p.add_argument(
        "--compile-model", default=None,
        choices=[None, "default", "reduce-overhead", "max-autotune"],
        help="Override fl_v7's arch-conditional default.",
    )
    p.add_argument(
        "--cudnn-nondeterministic", action="store_true",
        help="Disable cudnn.deterministic (recover ~5-15% speed; off by default).",
    )
    p.add_argument("--output-dir", default="artifacts/v7_fl_arch_sweep")
    p.add_argument(
        "--continue-on-cell-failure", action="store_true",
        help=("If a cell crashes (e.g. CUDA OOM on one arch), log and "
              "continue with subsequent cells rather than aborting."),
    )
    return p.parse_args()


def _parse_algo_spec(spec: str) -> tuple[str, dict]:
    """Parse 'algoname:{json...}' into ``(name, kwargs)``."""
    if ":" not in spec:
        raise ValueError(f"--algo-spec must be 'name:<json>': got {spec!r}")
    name, raw = spec.split(":", 1)
    kwargs = json.loads(raw) if raw.strip() else {}
    return name.strip(), kwargs


def _build_cfg(arch: str, algo_name: str, algo_kwargs: dict,
               alpha: float, seed: int, args: argparse.Namespace) -> V7Config:
    return V7Config(
        arch=arch,
        algorithm=algo_name,
        algo_kwargs=algo_kwargs,
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
        cudnn_deterministic=not args.cudnn_nondeterministic,
        output_dir=Path(args.output_dir),
    )


def main() -> None:
    args = _parse_args()
    archs = [a.strip() for a in args.archs.split(",") if a.strip()]
    specs = [_parse_algo_spec(s) for s in args.algo_spec]

    cells = list(product(archs, specs, args.alphas, args.seeds))
    log.info(
        "v7 matrix sweep: %d cells (archs=%s × algos=%s × alphas=%s × seeds=%s)",
        len(cells), archs, [n for n, _ in specs], args.alphas, args.seeds,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    summary_path = out_dir / f"_matrix_summary_{ts}.csv"
    latest_link = out_dir / "_matrix_summary_latest.csv"

    summary_rows: list[dict] = []
    n_failed = 0
    t_matrix_start = time.time()

    for arch, (algo_name, algo_kwargs), alpha, seed in cells:
        cfg = _build_cfg(arch, algo_name, algo_kwargs, alpha, seed, args)
        log.info(
            "--- cell: arch=%s algo=%s alpha=%.3f seed=%d (cfg.name=%s) ---",
            arch, algo_name, alpha, seed, cfg.name,
        )
        t_cell = time.time()
        try:
            result = run_v7_sweep(cfg)
            dt = time.time() - t_cell
            test_m = result.get("test", {})
            summary_rows.append({
                "arch": arch,
                "algorithm": algo_name,
                "alpha": alpha,
                "seed": seed,
                "test_auc": test_m.get("auc", float("nan")),
                "test_acc": test_m.get("accuracy", float("nan")),
                "test_f1": test_m.get("f1", float("nan")),
                "best_val_auc": result.get("best_val_auc", float("nan")),
                "duration_s": round(dt, 2),
                "name": cfg.name,
                "status": "ok",
            })
        except Exception as exc:
            n_failed += 1
            dt = time.time() - t_cell
            log.error(
                "cell FAILED arch=%s algo=%s alpha=%.3f seed=%d: %s\n%s",
                arch, algo_name, alpha, seed, exc, traceback.format_exc(),
            )
            summary_rows.append({
                "arch": arch,
                "algorithm": algo_name,
                "alpha": alpha,
                "seed": seed,
                "test_auc": float("nan"),
                "test_acc": float("nan"),
                "test_f1": float("nan"),
                "best_val_auc": float("nan"),
                "duration_s": round(dt, 2),
                "name": cfg.name,
                "status": f"failed: {type(exc).__name__}: {exc}",
            })
            if not args.continue_on_cell_failure:
                log.error("aborting matrix sweep (use --continue-on-cell-failure to skip).")
                break

        # Free VRAM between cells; matters for Spiking expand2 → LSTM transitions.
        if args.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    if summary_rows:
        with summary_path.open("w", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        try:
            latest_link.symlink_to(summary_path.name)
        except OSError:
            latest_link.write_text(summary_path.read_text())

    total_s = time.time() - t_matrix_start
    log.info(
        "matrix done: %d/%d cells succeeded in %.1fs (%.1f min); summary=%s",
        len(cells) - n_failed, len(cells), total_s, total_s / 60, summary_path,
    )
    if n_failed > 0 and not args.continue_on_cell_failure:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
