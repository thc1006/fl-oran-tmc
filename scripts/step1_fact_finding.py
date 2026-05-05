"""Step 1A — Fact-finding for the X3/C25/C26 BLOCKER triad.

Pure-CPU dataset inspection that resolves four open questions about
the Phase 5 sweep before any rewrite of PAPER_DRAFT.md mechanism
text:

  * **Q1 (T-H X9):** how many continuous features does the unified
    parquet actually expose, vs the §3.1 wording's "29"?
  * **Q2 (T-H slice_id / bs_id / sched / tr):** confirm cardinality
    of every categorical column.
  * **Q3 (T-H global pos rate):** measure the global train-set
    positive rate so §7.1 (a)'s vague "8-12 %" can be replaced with
    a specific number.
  * **Q4 (T-A C26 structural-correlation):** compute per-bs_id slice
    distribution, then fit a Dirichlet to those 7 distributions to
    quantify the "structural correlation" that natural-by-BS partition
    preserves and Dirichlet partition destroys.

All work is CPU-only. Wall-clock budget: ≤ 2 minutes on the unified
parquet's 18 M rows.

Output:
  artifacts/step1_factfinding.json   — machine-readable summary
  artifacts/step1_factfinding.md     — human-readable narrative
  /tmp/step1_factfinding.log         — progress log

Usage:
    python scripts/step1_fact_finding.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PARQUET = Path("/home/thc1006/dev/colosseum-oran-federated-slicing/data/coloran_raw_unified.parquet")
OUT_JSON = ROOT / "artifacts" / "step1_factfinding.json"
OUT_MD = ROOT / "artifacts" / "step1_factfinding.md"
LOG = Path("/tmp/step1_factfinding.log")


def _log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def main() -> int:
    LOG.write_text("")  # truncate
    _log(f"Step 1 fact-finding starting; parquet={PARQUET}")
    if not PARQUET.exists():
        _log(f"FATAL: parquet not found")
        return 1

    t0 = time.time()
    _log("loading parquet (this may take ~30s)...")
    df = pd.read_parquet(PARQUET)
    _log(f"loaded {len(df):,} rows in {time.time() - t0:.1f}s; columns={len(df.columns)}")

    # ====== Q1: continuous feature count ======
    # Definition matches src/fl_oran/training/centralized_v3.py V3_CONTINUOUS:
    V3_CONTINUOUS = [
        "num_ues", "slice_prb",
        "sum_requested_prbs", "sum_granted_prbs",
        "tx_brate_dl_Mbps", "rx_brate_ul_Mbps",
        "tx_pkts_dl", "rx_pkts_ul",
        "dl_buffer_bytes", "ul_buffer_bytes",
        "dl_bler", "ul_bler",
        "dl_mcs", "ul_mcs", "dl_cqi", "ul_sinr", "ul_rssi",
    ]
    n_continuous = len(V3_CONTINUOUS)
    missing_cont = [c for c in V3_CONTINUOUS if c not in df.columns]
    _log(f"Q1: V3_CONTINUOUS has {n_continuous} features; missing in parquet: {missing_cont}")

    # ====== Q2: categorical cardinality ======
    cat_cols = ["bs_id", "slice_id", "sched", "tr"]
    cat_summary = {}
    for c in cat_cols:
        if c not in df.columns:
            cat_summary[c] = {"present": False}
            continue
        uniq = sorted(int(v) for v in df[c].dropna().unique().tolist())
        cat_summary[c] = {"present": True, "n_unique": len(uniq), "values": uniq}
        _log(f"Q2 {c}: {len(uniq)} unique values, range [{uniq[0]}..{uniq[-1]}]")

    # ====== Q3: global train pos rate ======
    # Phase 5 setup: train tr ∈ {0..21}, val {22..24}, test {25..27}
    # Threshold = 0.10 on ul_bler (next-second). For consistency with
    # add_classification_target's behaviour we replicate the next-second
    # shift here on the train subset only.
    train_df = df[df["tr"].between(0, 21)].copy()
    _log(f"Q3: train subset = {len(train_df):,} rows")
    if "ul_bler" in train_df.columns:
        # add_classification_target shifts by -1 within each (bs_id, slice_id) group
        # to make the target "next second's BLER".
        train_df = train_df.sort_values(["bs_id", "slice_id"]).copy()
        # Use a per-group shift via groupby to keep semantics identical to
        # data_v2.targets_v2.add_classification_target (next-second target).
        train_df["y_sla_next"] = (
            train_df.groupby(["bs_id", "slice_id"], observed=True)["ul_bler"].shift(-1)
        )
        # Drop tail rows of each group where shift produces NaN
        train_df = train_df.dropna(subset=["y_sla_next"])
        positives = (train_df["y_sla_next"] > 0.10).astype(int)
        global_pos_rate = float(positives.mean())
        _log(f"Q3: global train pos rate = {global_pos_rate:.4f} ({global_pos_rate*100:.2f}%)")
        # Per-slice positive rate
        per_slice_pos = (
            train_df.assign(y=positives)
                    .groupby("slice_id", observed=True)["y"]
                    .agg(["mean", "size"])
                    .rename(columns={"mean": "pos_rate", "size": "n_rows"})
        )
        per_slice_pos_dict = {
            int(s): {"pos_rate": float(r["pos_rate"]), "n_rows": int(r["n_rows"])}
            for s, r in per_slice_pos.iterrows()
        }
        for s, r in per_slice_pos_dict.items():
            _log(f"Q3 per-slice {s}: pos_rate={r['pos_rate']:.4f} n={r['n_rows']:,}")
    else:
        global_pos_rate = None
        per_slice_pos_dict = {}
        _log("Q3: ul_bler column missing; skipping pos rate computation")

    # ====== Q4: per-bs_id slice distribution ======
    # Compute the empirical slice mixture for each base station, then
    # quantify how concentrated those mixtures are. If natural-by-BS
    # is to be defended as a "structural correlation" partition, we
    # should see substantively non-uniform per-bs slice distributions.
    train_only = df[df["tr"].between(0, 21)]
    bs_slice_counts = (
        train_only.groupby(["bs_id", "slice_id"], observed=True)
                  .size()
                  .unstack(fill_value=0)
                  .sort_index()
    )
    # Row-normalise: each row is a 4-vector summing to 1 (the per-bs slice mixture)
    bs_slice_props = bs_slice_counts.div(bs_slice_counts.sum(axis=1), axis=0)
    _log(f"Q4: per-bs slice mixture (rows=bs_id, cols=slice_id):")
    for bs_id, row in bs_slice_props.iterrows():
        _log(f"  bs_id={bs_id}: " + " ".join(f"s{s}={p:.3f}" for s, p in row.items()))

    # Distance from uniform: KL divergence to uniform 1/n_slice
    n_slice = bs_slice_props.shape[1]
    uniform = 1.0 / n_slice
    eps = 1e-12
    per_bs_kl_to_uniform = {
        int(bs): float(np.sum(row.values * np.log((row.values + eps) / uniform)))
        for bs, row in bs_slice_props.iterrows()
    }
    avg_kl = float(np.mean(list(per_bs_kl_to_uniform.values())))
    _log(f"Q4: avg KL(per-bs slice mix || uniform) = {avg_kl:.4f}")
    _log(f"Q4: per-bs KL: {per_bs_kl_to_uniform}")

    # Also fit a Dirichlet to the 7 per-bs distributions; the MoM
    # estimator gives an α that quantifies "how concentrated" the
    # 7 BS distributions are. This α is the empirical Dirichlet-α
    # equivalent of natural-by-BS — useful for §7.1's reframing.
    # MoM Dirichlet: α_k = mean_k * (mean_k(1-mean_k) / var_k - 1) summed appropriately.
    #
    # Simpler: Cramer-style estimator α̂_total = (1 - sum_k mean_k^2) / (sum_k var_k).
    means = bs_slice_props.mean(axis=0).values
    vars_ = bs_slice_props.var(axis=0, ddof=0).values
    if np.all(vars_ > 0):
        sum_var = float(np.sum(vars_))
        alpha_hat_total = float((1.0 - np.sum(means ** 2)) / sum_var) - 1.0
        # Per-component:
        alpha_hat_per = (means * alpha_hat_total).tolist()
        _log(f"Q4: empirical Dirichlet α̂_total = {alpha_hat_total:.3f}")
        _log(f"Q4: per-component α̂ = {[f'{a:.3f}' for a in alpha_hat_per]}")
    else:
        alpha_hat_total = None
        alpha_hat_per = None
        _log(f"Q4: per-bs slice variances are zero — cannot fit Dirichlet")

    # ====== Final summary ======
    summary = {
        "parquet_path": str(PARQUET),
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "Q1_n_continuous_features": n_continuous,
        "Q1_continuous_list": V3_CONTINUOUS,
        "Q1_missing_in_parquet": missing_cont,
        "Q2_categorical_summary": cat_summary,
        "Q3_global_train_pos_rate": global_pos_rate,
        "Q3_per_slice_pos_rate": per_slice_pos_dict,
        "Q4_per_bs_slice_proportions": {
            int(bs): {int(s): float(p) for s, p in row.items()}
            for bs, row in bs_slice_props.iterrows()
        },
        "Q4_per_bs_kl_to_uniform": per_bs_kl_to_uniform,
        "Q4_avg_kl_to_uniform": avg_kl,
        "Q4_dirichlet_alpha_hat_total": alpha_hat_total,
        "Q4_dirichlet_alpha_hat_per_component": alpha_hat_per,
        "wall_clock_s": time.time() - t0,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    _log(f"wrote {OUT_JSON}")

    md_lines = [
        "# Step 1 fact-finding summary\n",
        f"Generated by `scripts/step1_fact_finding.py` in {summary['wall_clock_s']:.1f}s.\n",
        f"## Parquet shape\n",
        f"- {summary['n_rows']:,} rows × {summary['n_columns']} columns\n",
        f"## Q1 — continuous features\n",
        f"- V3_CONTINUOUS has **{n_continuous}** features (paper §3.1 currently says 29 — wrong)\n",
        f"## Q2 — categorical cardinality\n",
    ]
    for c, info in cat_summary.items():
        if info.get("present"):
            md_lines.append(f"- `{c}`: {info['n_unique']} unique values, range [{info['values'][0]}..{info['values'][-1]}]\n")
    md_lines.append(f"## Q3 — global train positive rate\n")
    if global_pos_rate is not None:
        md_lines.append(f"- **{global_pos_rate*100:.2f}%** (paper §7.1 (a) currently says \"8-12 %\" — replace with this)\n")
        md_lines.append(f"- Per-slice positive rate:\n")
        for s, r in per_slice_pos_dict.items():
            md_lines.append(f"  - slice {s}: {r['pos_rate']*100:.2f}% (n={r['n_rows']:,})\n")
    md_lines.append(f"## Q4 — structural correlation (per-bs slice mix)\n")
    md_lines.append(f"- Avg KL(per-bs slice mix || uniform) = **{avg_kl:.4f}** (higher = more structurally distinct per-bs traffic)\n")
    if alpha_hat_total is not None:
        md_lines.append(f"- Empirical Dirichlet α̂_total = **{alpha_hat_total:.3f}** (Phase 5 swept α∈[0.05, 5.0]; natural-by-BS sits at this fitted α)\n")
    md_lines.append(f"- Per-bs slice mixture (rows=bs_id, cols=slice_id):\n\n")
    md_lines.append(f"| bs_id | s0 | s1 | s2 | s3 | KL→uniform |\n")
    md_lines.append(f"|---|---|---|---|---|---|\n")
    for bs_id, row in bs_slice_props.iterrows():
        kl = per_bs_kl_to_uniform.get(int(bs_id), 0.0)
        md_lines.append(f"| {bs_id} | {row.get(0, 0):.3f} | {row.get(1, 0):.3f} | {row.get(2, 0):.3f} | {row.get(3, 0):.3f} | {kl:.4f} |\n")
    OUT_MD.write_text("".join(md_lines))
    _log(f"wrote {OUT_MD}")
    _log(f"DONE in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
