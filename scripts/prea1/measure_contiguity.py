"""Back the §6.2 claim that the row-Dirichlet α-curve is fragmentation severity.

NO training. For each Dirichlet alpha, build the row-level partition on split.train
and measure the fraction of sliding windows (5 consecutive rows within a
(run_id,slice_id) group, sorted by step_idx) whose step_idx are CONTIGUOUS
(span == seq_len-1). Prediction (§6.2): contiguity fraction rises as alpha falls
(concentration -> denser per-client coverage of each run -> less fragmentation),
mirroring the AUC curve.

Run caged: systemd-run --user --scope -p MemoryMax=14G -p MemorySwapMax=0 -- \
    .venv/bin/python artifacts/prea1/fragmentation/measure_contiguity.py
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
OUT = REPO / "artifacts/prea1/fragmentation"
TRAIN_TR, VAL_TR, TEST_TR = list(range(22)), [22, 23, 24], [25, 26, 27]
SEQ_LEN, N_CLIENTS = 5, 7
ALPHAS = [0.05, 0.1, 0.5, 1.0, 5.0]


def _contig_fraction(shards: dict) -> tuple[int, int]:
    """(#contiguous windows, #windows) over all clients' (run,slice) groups."""
    win = 0
    contig = 0
    for _cid, s in shards.items():
        for _key, g in s.groupby(["run_id", "slice_id"], observed=True):
            steps = np.sort(g["step_idx"].to_numpy())
            n = len(steps)
            if n < SEQ_LEN:
                continue
            # window i covers steps[i..i+SEQ_LEN-1]; contiguous iff span == SEQ_LEN-1
            spans = steps[SEQ_LEN - 1:] - steps[:n - SEQ_LEN + 1]
            win += len(spans)
            contig += int((spans == SEQ_LEN - 1).sum())
    return contig, win


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PARQUET, columns=["run_id", "slice_id", "step_idx", "tr", "bs_id"])
    train = ood_split_by_tr(df, TRAIN_TR, VAL_TR, TEST_TR).train
    res = {"seq_len": SEQ_LEN, "alphas": {}}
    print(f"{'alpha':>6} | {'contig_windows':>15} | {'total_windows':>14} | {'contig_frac':>11}")
    for a in ALPHAS:
        sh = partition_clients(train, mode="dirichlet", alpha=a, n_clients=N_CLIENTS, seed=0)
        c, w = _contig_fraction(sh)
        frac = c / w if w else float("nan")
        res["alphas"][a] = {"contiguous": c, "windows": w, "contig_fraction": round(frac, 4)}
        print(f"{a:>6} | {c:>15,} | {w:>14,} | {frac:>11.4f}")
    # natural-by-BS reference (intact -> ~1.0 expected)
    sh_nat = partition_clients(train, mode="iid")
    c, w = _contig_fraction(sh_nat)
    res["natural_by_bs"] = {"contiguous": c, "windows": w, "contig_fraction": round(c / w, 4)}
    print(f"{'natural':>6} | {c:>15,} | {w:>14,} | {c/w:>11.4f}")
    (OUT / "contiguity_vs_alpha.json").write_text(json.dumps(res, indent=2))
    print(f"WROTE {OUT / 'contiguity_vs_alpha.json'}")


if __name__ == "__main__":
    main()
