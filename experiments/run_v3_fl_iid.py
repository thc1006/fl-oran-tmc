"""FL IID run."""
from __future__ import annotations

from pathlib import Path

from fl_oran.logging_utils import setup_logging
from fl_oran.training.centralized_v3 import V3Config
from fl_oran.training.fl_v3 import run_federated


def main() -> int:
    cfg = V3Config(
        name="v3_fl_iid",
        unified_parquet=Path("data/coloran_raw_unified.parquet"),
        sample_ratio=0.2,
        seq_len=5,
        threshold=0.10,
        num_rounds=20,
        clients_per_round=5,
        max_steps_per_round=500,
        batch_size=256,
        lr=1e-3,
        lr_warmup_rounds=2,
        grad_clip=1.0,
        mixed_precision="bf16",
    )
    setup_logging(level="INFO", run_name=cfg.name)
    run_federated(cfg, partition_mode="iid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
