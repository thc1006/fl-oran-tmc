"""Centralized LSTM baseline — scientific gate for v3."""
from __future__ import annotations

from pathlib import Path

from fl_oran.logging_utils import setup_logging
from fl_oran.training.centralized_v3 import V3Config, run_centralized


def main() -> int:
    cfg = V3Config(
        name="v3_centralized",
        unified_parquet=Path("data/coloran_raw_unified.parquet"),
        sample_ratio=0.2,
        seq_len=5,
        threshold=0.10,
        centralized_epochs=3,
        batch_size=256,
        lr=1e-3,
        mixed_precision="bf16",
    )
    setup_logging(level="INFO", run_name=cfg.name)
    run_centralized(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
