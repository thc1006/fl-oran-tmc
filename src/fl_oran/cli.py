"""CLI entry point: `python -m fl_oran` or the installed `fl-oran` script."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import DPConfig, DataConfig, ExperimentConfig, FedConfig, TrainingConfig
from .logging_utils import setup_logging, get_logger
from .training import run_experiment


def _add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--variant", choices=["v106", "v107_1", "v107_2"], required=False,
                   help="Which notebook variant to run.")
    p.add_argument("--config", type=Path, help="YAML config to load (overrides CLI args for matching keys).")
    p.add_argument("--name", default=None)
    p.add_argument("--parquet", type=Path, default=Path("data/coloran_processed_features.parquet"))
    p.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    p.add_argument("--samples-per-client", type=int, default=200_000)
    p.add_argument("--sample-ratio", type=float, default=None,
                   help="Sub-sample the raw parquet to this fraction before anything else. Good for smoke tests.")
    p.add_argument("--num-rounds", type=int, default=30)
    p.add_argument("--clients-per-round", type=int, default=5)
    p.add_argument("--local-epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seq-len", type=int, default=5, help="Temporal window (v107_1 only).")
    p.add_argument("--early-stopping-patience", type=int, default=8)
    p.add_argument("--dp", action="store_true", help="Enable DP (Gaussian mechanism on server updates).")
    p.add_argument("--dp-clip", type=float, default=1.0)
    p.add_argument("--dp-noise", type=float, default=0.1)
    p.add_argument("--dp-target-epsilon", type=float, default=10.0,
                   help="Privacy budget; training stops when cumulative ε exceeds this value.")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--amp", choices=["off", "fp16", "bf16"], default="bf16")
    p.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--seed", type=int, default=42)


def _build_cfg_from_args(args: argparse.Namespace) -> ExperimentConfig:
    if args.config is not None:
        cfg = ExperimentConfig.from_yaml(args.config)
    else:
        cfg = ExperimentConfig(
            name=args.name or f"{args.variant}_run",
            variant=args.variant,
        )
    # CLI overrides (both when config was loaded and not).
    if args.variant:
        cfg.variant = args.variant
    if args.name:
        cfg.name = args.name
    cfg.data = DataConfig(
        parquet_path=args.parquet,
        samples_per_client=args.samples_per_client,
        sample_ratio=args.sample_ratio,
        sequence_length=args.seq_len,
    )
    cfg.fed = FedConfig(
        num_total_clients=7,
        num_rounds=args.num_rounds,
        clients_per_round=args.clients_per_round,
        local_epochs=args.local_epochs,
        client_lr=args.lr,
        batch_size=args.batch_size,
        random_state=args.seed,
        early_stopping_patience=args.early_stopping_patience,
    )
    cfg.dp = DPConfig(
        enabled=args.dp,
        l2_norm_clip=args.dp_clip,
        noise_multiplier=args.dp_noise,
        target_epsilon=args.dp_target_epsilon,
    )
    cfg.training = TrainingConfig(
        device=args.device,
        mixed_precision=args.amp,
        compile_model=not args.no_compile,
        num_workers=args.num_workers,
    )
    cfg.output_dir = args.output_dir
    return cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fl-oran", description="Local PyTorch FL for ColO-RAN slicing.")
    _add_arguments(parser)
    args = parser.parse_args(argv)
    cfg = _build_cfg_from_args(args)
    setup_logging(level=args.log_level, run_name=cfg.name)
    log = get_logger("fl_oran.cli")
    log.info("Running experiment '%s' (variant=%s)", cfg.name, cfg.variant)
    log.info("Config: %s", cfg.to_dict())
    hist = run_experiment(cfg)
    log.info("Done. %d rounds recorded. Final test loss=%.6f", len(hist), float(hist["test_loss"].iloc[-1]) if len(hist) else float("nan"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
