"""CLI to build the unified parquet from the raw ColO-RAN tree.

Usage:
    python -m fl_oran.data_raw.cli --raw-root raw/colosseum-oran-coloran-dataset-master \
                                   --out-path data/coloran_raw_unified.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..logging_utils import get_logger, setup_logging
from .merge import build_unified_parquet


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build unified parquet from raw ColO-RAN CSVs.")
    parser.add_argument("--raw-root", type=Path, required=True,
                        help="Directory containing rome_static_medium/sched*/tr*/exp*/bs*/")
    parser.add_argument("--out-path", type=Path,
                        default=Path("data/coloran_raw_unified.parquet"))
    parser.add_argument("--limit-runs", type=int, default=None,
                        help="Limit number of runs processed (for smoke tests)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logging(level=args.log_level, run_name="build_unified")
    log = get_logger("fl_oran.data_raw.cli")
    log.info("raw_root=%s out_path=%s limit_runs=%s",
             args.raw_root, args.out_path, args.limit_runs)
    build_unified_parquet(args.raw_root, args.out_path, limit_runs=args.limit_runs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
