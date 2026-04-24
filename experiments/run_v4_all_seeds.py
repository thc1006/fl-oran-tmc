"""v4: run all 3 experiments × 3 seeds = 9 runs, aggregate to mean ± std.

- Federated scaler (sufficient-stats aggregation)
- Equalised gradient budget (50k steps)
- Seeds 42, 123, 456
"""
from __future__ import annotations

import json
import statistics as stats
import time
from pathlib import Path

from fl_oran.logging_utils import setup_logging, get_logger
from fl_oran.training.centralized_v3 import V3Config, run_centralized
from fl_oran.training.fl_v3 import run_federated


NONIID_MAP = {
    1: [0], 2: [0],   # eMBB specialists
    3: [1], 4: [1],   # MTC specialists
    5: [2], 6: [2],   # URLLC specialists
    7: [0, 1, 2],     # generalist
}

SEEDS = [42, 123, 456]


def _base_cfg(seed: int, name: str) -> V3Config:
    return V3Config(
        name=name,
        unified_parquet=Path("data/coloran_raw_unified.parquet"),
        sample_ratio=0.2,
        seq_len=5,
        threshold=0.10,
        seed=seed,
        total_gradient_steps=50_000,
        num_rounds=20,
        clients_per_round=5,
        max_steps_per_round=500,
        batch_size=256,
        lr=1e-3,
        lr_warmup_rounds=2,
        grad_clip=1.0,
        mixed_precision="bf16",
    )


def _extract_test_metrics(result: dict) -> dict:
    """Extract the comparable test metrics from either centralized or FL result."""
    for k in ("centralized_lstm", "fl_lstm_test"):
        if k in result:
            m = result[k]
            return {"auc": m.get("auc", 0), "acc": m["accuracy"],
                    "f1": m["f1"], "pos_rate_pred": m["positive_rate_pred"]}
    raise ValueError("no recognised metric key in result")


def main() -> int:
    setup_logging(level="INFO", run_name="v4_multi_seed")
    log = get_logger("v4_runner")

    baselines = {"majority": None, "persistence": None}
    records: dict[str, dict[int, dict]] = {
        "centralized": {},
        "fl_iid": {},
        "fl_noniid": {},
    }

    for seed in SEEDS:
        log.info("=" * 70)
        log.info("SEED %d  (1/3 experiments: centralized)", seed)
        log.info("=" * 70)
        t0 = time.time()
        r = run_centralized(_base_cfg(seed, f"v4_cen_s{seed}"))
        records["centralized"][seed] = _extract_test_metrics(r)
        if baselines["majority"] is None:
            baselines["majority"] = r["majority_baseline"]
            baselines["persistence"] = r["persistence_classifier"]
        log.info("centralized seed=%d done in %.1fs", seed, time.time() - t0)

        log.info("=" * 70)
        log.info("SEED %d  (2/3 experiments: FL IID)", seed)
        log.info("=" * 70)
        t0 = time.time()
        r = run_federated(_base_cfg(seed, f"v4_iid_s{seed}"), partition_mode="iid")
        records["fl_iid"][seed] = _extract_test_metrics(r)
        log.info("fl_iid seed=%d done in %.1fs", seed, time.time() - t0)

        log.info("=" * 70)
        log.info("SEED %d  (3/3 experiments: FL Non-IID)", seed)
        log.info("=" * 70)
        t0 = time.time()
        r = run_federated(_base_cfg(seed, f"v4_noniid_s{seed}"),
                          partition_mode="noniid_slice",
                          client_slice_map=NONIID_MAP)
        records["fl_noniid"][seed] = _extract_test_metrics(r)
        log.info("fl_noniid seed=%d done in %.1fs", seed, time.time() - t0)

    # Aggregate mean ± std.
    summary = {
        "seeds": SEEDS,
        "baselines": baselines,
        "by_experiment": {},
    }
    for exp_name, seed_results in records.items():
        agg = {}
        for metric in ("auc", "acc", "f1", "pos_rate_pred"):
            values = [seed_results[s][metric] for s in SEEDS]
            agg[metric] = {
                "mean": float(stats.fmean(values)),
                "std": float(stats.stdev(values)) if len(values) > 1 else 0.0,
                "per_seed": values,
            }
        summary["by_experiment"][exp_name] = agg

    out_path = Path("artifacts/logs/v4_multi_seed_summary.json")
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    log.info("wrote %s", out_path)

    # Pretty table.
    log.info("=" * 80)
    log.info("v4 FINAL (3 seeds, mean ± std)")
    log.info("  %-22s | %-15s | %-15s | %-15s",
             "Model", "AUC", "Accuracy", "F1")
    log.info("  " + "-" * 78)
    maj = baselines["majority"]
    per = baselines["persistence"]
    log.info("  %-22s | %-15s | %.3f           | %.3f",
             "Majority baseline", "(const)", maj["accuracy"], 0.0)
    log.info("  %-22s | %.3f           | %.3f           | %.3f",
             "Persistence classifier", per["auc"], per["accuracy"], per["f1"])
    for exp_name in ("centralized", "fl_iid", "fl_noniid"):
        a = summary["by_experiment"][exp_name]
        label = {"centralized": "Centralized LSTM",
                 "fl_iid": "FL IID LSTM",
                 "fl_noniid": "FL Non-IID LSTM"}[exp_name]
        log.info("  %-22s | %.3f ± %.3f   | %.3f ± %.3f   | %.3f ± %.3f",
                 label,
                 a["auc"]["mean"], a["auc"]["std"],
                 a["acc"]["mean"], a["acc"]["std"],
                 a["f1"]["mean"], a["f1"]["std"])
    log.info("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
