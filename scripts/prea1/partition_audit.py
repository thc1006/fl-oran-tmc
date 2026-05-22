"""Auditable partition-structure table for the sequence-integrity evidence pack.

For each partition mode, partition split.train into 7 clients and report, per client:
  rows, windows (sequences), runs touched, distinct bs / slice / tr (+ Shannon
  entropy of the bs and slice distributions), and the fragmentation score
  (= fraction of per-client sliding windows whose seq_len rows have contiguous
  step_idx; 1.0 = no fragmentation).

Run caged:
  systemd-run --user --scope -p MemoryMax=14G -p MemorySwapMax=0 -- \
    .venv/bin/python artifacts/prea1/partition_audit.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fl_oran.data_v2.partition import partition_clients
from fl_oran.data_v2.split import ood_split_by_tr

REPO = Path("/home/thc1006/dev/fl-oran-tmc")
PARQUET = REPO / "data/coloran_raw_unified.parquet"
OUT = REPO / "artifacts/prea1/partition_audit.json"
TRAIN_TR, VAL_TR, TEST_TR = list(range(22)), [22, 23, 24], [25, 26, 27]
SEQ_LEN, N_CLIENTS, SEED = 5, 7, 0
# (label, mode, kwargs). Dirichlet modes use alpha=1.0 (representative non-IID point).
MODES = [
    ("natural (iid by-BS)", "iid", {}),
    ("row-random (random_split)", "random_split", {"n_clients": N_CLIENTS, "seed": SEED}),
    ("row-Dirichlet (a=1.0)", "dirichlet", {"n_clients": N_CLIENTS, "seed": SEED, "alpha": 1.0}),
    ("run_random", "run_random", {"n_clients": N_CLIENTS, "seed": SEED}),
    ("run_dirichlet (a=1.0)", "run_dirichlet", {"n_clients": N_CLIENTS, "seed": SEED, "alpha": 1.0}),
]


def _entropy(counts: np.ndarray) -> float:
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def _client_stats(s: pd.DataFrame) -> dict:
    n_win = 0
    n_contig = 0
    runs = set()
    for key, g in s.groupby(["run_id", "slice_id"], observed=True):
        runs.add(key[0])
        steps = np.sort(g["step_idx"].to_numpy())
        n = len(steps)
        if n < SEQ_LEN:
            continue
        spans = steps[SEQ_LEN - 1:] - steps[:n - SEQ_LEN + 1]
        n_win += len(spans)
        n_contig += int((spans == SEQ_LEN - 1).sum())
    bs_counts = s["bs_id"].value_counts().to_numpy()
    sl_counts = s["slice_id"].value_counts().to_numpy()
    return {
        "rows": int(len(s)),
        "windows": int(n_win),
        "runs": int(len(runs)),
        "distinct_bs": int(s["bs_id"].nunique()),
        "bs_entropy_bits": round(_entropy(bs_counts), 3),
        "distinct_slice": int(s["slice_id"].nunique()),
        "slice_entropy_bits": round(_entropy(sl_counts), 3),
        "distinct_tr": int(s["tr"].nunique()),
        "contig_frac": round(n_contig / n_win, 4) if n_win else float("nan"),
        "bs_dist": {int(k): int(v) for k, v in s["bs_id"].value_counts().sort_index().items()},
        "slice_dist": {int(k): int(v) for k, v in s["slice_id"].value_counts().sort_index().items()},
        "tr_dist": {int(k): int(v) for k, v in s["tr"].value_counts().sort_index().items()},
    }


def main() -> None:
    df = pd.read_parquet(PARQUET, columns=["run_id", "slice_id", "step_idx", "tr", "bs_id"])
    train = ood_split_by_tr(df, TRAIN_TR, VAL_TR, TEST_TR).train
    out = {"seq_len": SEQ_LEN, "n_clients": N_CLIENTS, "seed": SEED, "modes": {}}
    hdr = f"{'mode':>26} | {'cli':>3} | {'rows':>9} | {'windows':>9} | {'runs':>6} | {'bs':>2} | {'bsH':>5} | {'sl':>2} | {'tr':>2} | {'contig':>6}"
    print(hdr)
    print("-" * len(hdr))
    for label, mode, kw in MODES:
        shards = partition_clients(train, mode=mode, **kw)
        per_client = {}
        tot_w = tot_c = 0
        for cid in sorted(shards):
            st = _client_stats(shards[cid])
            per_client[str(cid)] = st
            tot_w += st["windows"]
            tot_c += int(round(st["contig_frac"] * st["windows"])) if st["windows"] else 0
            print(f"{label:>26} | {str(cid):>3} | {st['rows']:>9,} | {st['windows']:>9,} | "
                  f"{st['runs']:>6} | {st['distinct_bs']:>2} | {st['bs_entropy_bits']:>5} | "
                  f"{st['distinct_slice']:>2} | {st['distinct_tr']:>2} | {st['contig_frac']:>6.3f}")
        agg = {
            "n_clients_nonempty": len(per_client),
            "total_windows": tot_w,
            "fragmentation_score_overall": round(tot_c / tot_w, 4) if tot_w else float("nan"),
        }
        out["modes"][label] = {"mode": mode, "kwargs": kw, "aggregate": agg, "per_client": per_client}
        print(f"{label:>26} | AGG | overall contig_frac (fragmentation score) = {agg['fragmentation_score_overall']}")
        print("-" * len(hdr))
    OUT.write_text(json.dumps(out, indent=2))
    print(f"WROTE {OUT}")


if __name__ == "__main__":
    main()
