"""Step 2 — Mechanism candidate search after Step 1 destroyed both
prior hypotheses.

Step 1 (artifacts/step1_factfinding.json) showed:
  * Per-bs slice mixture is uniform (KL≈0, fitted Dirichlet α̂=234k)
    → "bs↔slice structural correlation" mechanism is FALSE
  * Global train pos rate is 30.9% (not 8-12%)
    → "sparse-positive-class specialist" mechanism is FALSE

But the empirical finding remains: natural-by-BS uniformly outperforms
every Dirichlet α across all 90 (arch, algo, partition) groups in
Phase 5. Step 2 searches for what the real mechanism is.

Three candidates tested here, all CPU-only:

  C1: **bs_id is a strong predictor.** If per-bs continuous-feature
      distributions differ substantially, then natural-by-BS
      (one client per bs_id) lets each client specialise on its
      bs's signal regime, while Dirichlet partition mixes bs's
      together and dilutes the specialisation.
      → Test: KL divergence between per-bs marginal distributions
        of every continuous feature. Large KL → bs differentiation.

  C2: **Per-client sample-size collapse under Dirichlet α<<1.**
      Dirichlet partition divides each slice's rows by Dirichlet
      proportions; with α=0.05, some clients get ≈0 rows of a slice
      → effective per-client training data is small + concentrated.
      → Test: simulate Dirichlet partition (partition.py logic) for
        each α∈{0.05, 0.10, 0.50, 1.00, 5.00} × seed, report per-
        client min/median/max row counts.

  C3: **bs_id embedding norm — does the model actually use bs_id?**
      If bs_id embedding is the dominant categorical feature, then
      partitioning ON bs_id (natural-by-BS) preserves that feature's
      learning, while partitioning across bs_id (Dirichlet) dilutes
      it. Read a saved Phase 5 LSTM checkpoint and compare per-
      categorical-feature embedding norms.
      → Test: walk artifacts/v7_stage2_full/v7_lstm_fedavg_iid_*/
        for any best_state.pt, load its embedding weights, compute
        L2 norm per categorical feature.

Wall-clock budget: ≤ 5 minutes on the unified parquet's 18 M rows.

Output:
  artifacts/step2_mechanism_search.json
  artifacts/step2_mechanism_search.md
  /tmp/step2_mechanism.log
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PARQUET = Path("/home/thc1006/dev/colosseum-oran-federated-slicing/data/coloran_raw_unified.parquet")
OUT_JSON = ROOT / "artifacts" / "step2_mechanism_search.json"
OUT_MD = ROOT / "artifacts" / "step2_mechanism_search.md"
LOG = Path("/tmp/step2_mechanism.log")

V3_CONTINUOUS = [
    "num_ues", "slice_prb",
    "sum_requested_prbs", "sum_granted_prbs",
    "tx_brate_dl_Mbps", "rx_brate_ul_Mbps",
    "tx_pkts_dl", "rx_pkts_ul",
    "dl_buffer_bytes", "ul_buffer_bytes",
    "dl_bler", "ul_bler",
    "dl_mcs", "ul_mcs", "dl_cqi", "ul_sinr", "ul_rssi",
]


def _log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def _bin_kl(a: np.ndarray, b: np.ndarray, n_bins: int = 50) -> float:
    """KL(a || b) on shared histogram bins. Add eps to avoid div by 0."""
    lo = float(min(a.min(), b.min()))
    hi = float(max(a.max(), b.max()))
    if hi == lo:
        return 0.0
    edges = np.linspace(lo, hi, n_bins + 1)
    pa, _ = np.histogram(a, bins=edges, density=False)
    pb, _ = np.histogram(b, bins=edges, density=False)
    pa = pa.astype(float) + 1.0
    pb = pb.astype(float) + 1.0
    pa /= pa.sum()
    pb /= pb.sum()
    return float(np.sum(pa * np.log(pa / pb)))


def main() -> int:
    LOG.write_text("")
    _log("Step 2 mechanism search starting")
    t0 = time.time()

    _log(f"loading parquet from {PARQUET}")
    df = pd.read_parquet(PARQUET)
    train = df[df["tr"].between(0, 21)].copy()
    _log(f"train subset = {len(train):,} rows")

    # =========================================================================
    # C1: bs_id discriminative power on continuous features
    # =========================================================================
    _log("=" * 60)
    _log("C1: per-bs continuous-feature distribution KL divergences")
    _log("=" * 60)
    bs_ids = sorted(int(b) for b in train["bs_id"].unique())
    _log(f"bs_ids = {bs_ids}")

    # For each continuous feature, compute pairwise KL between bs's,
    # subsampling to keep things fast.
    sample_per_bs = 50_000
    feature_kl_summary = {}
    for feat in V3_CONTINUOUS:
        if feat not in train.columns:
            _log(f"  feature {feat} missing in parquet; skipping")
            continue
        per_bs_samples = {}
        for bs in bs_ids:
            sub = train.loc[train["bs_id"] == bs, feat].dropna().to_numpy()
            if len(sub) > sample_per_bs:
                rng = np.random.default_rng(42)
                sub = rng.choice(sub, size=sample_per_bs, replace=False)
            per_bs_samples[bs] = sub
        # Pairwise KLs
        kls = []
        for i, bs_a in enumerate(bs_ids):
            for bs_b in bs_ids[i + 1:]:
                kl = _bin_kl(per_bs_samples[bs_a], per_bs_samples[bs_b])
                kls.append(kl)
        kls = np.array(kls)
        feature_kl_summary[feat] = {
            "kl_mean": float(kls.mean()),
            "kl_max": float(kls.max()),
            "kl_min": float(kls.min()),
            "n_pairs": int(len(kls)),
        }
        _log(f"  {feat:24s}: KL mean={kls.mean():.4f} max={kls.max():.4f} min={kls.min():.4f}")

    # Top discriminative features
    sorted_feats = sorted(
        feature_kl_summary.items(), key=lambda kv: kv[1]["kl_mean"], reverse=True
    )
    _log("C1 verdict — top 5 most-discriminating features by mean pairwise bs-KL:")
    for feat, info in sorted_feats[:5]:
        _log(f"   {feat}: mean KL = {info['kl_mean']:.4f}")

    # =========================================================================
    # C2: per-client sample-size distribution under Dirichlet partition
    # =========================================================================
    _log("=" * 60)
    _log("C2: simulated per-client sample sizes under Dirichlet partition")
    _log("=" * 60)
    n_clients = 7
    alpha_sweep = [0.05, 0.10, 0.50, 1.00, 5.00]
    seed = 42
    sample_size_summary: dict[float, dict] = {}
    for alpha in alpha_sweep:
        rng = np.random.default_rng(seed)
        client_sizes = np.zeros(n_clients, dtype=int)
        for slice_id, g in train.groupby("slice_id", observed=True):
            n_rows = len(g)
            proportions = rng.dirichlet([alpha] * n_clients)
            splits = (np.cumsum(proportions) * n_rows).astype(int)
            client_chunks = np.diff(np.concatenate([[0], splits]))
            client_sizes += client_chunks
        # Stats
        s = client_sizes.astype(float)
        sample_size_summary[alpha] = {
            "min": int(s.min()),
            "max": int(s.max()),
            "mean": float(s.mean()),
            "std": float(s.std(ddof=0)),
            "min_over_max_ratio": float(s.min() / max(s.max(), 1)),
            "min_over_mean_ratio": float(s.min() / max(s.mean(), 1)),
            "per_client": s.astype(int).tolist(),
        }
        _log(f"  α={alpha}: min={int(s.min()):>10d}  max={int(s.max()):>10d}  mean={s.mean():>10.0f}  min/max={s.min()/max(s.max(),1):.4f}")

    # Natural-by-BS sample sizes
    nat_sizes = train.groupby("bs_id", observed=True).size().sort_index().to_dict()
    nat_sizes_int = {int(k): int(v) for k, v in nat_sizes.items()}
    _log(f"  natural-by-BS: per-client (bs_id) sizes = {nat_sizes_int}")
    nat_arr = np.array(list(nat_sizes_int.values()), dtype=float)
    _log(
        f"  natural-by-BS: min={int(nat_arr.min())}  max={int(nat_arr.max())}  "
        f"min/max={nat_arr.min()/nat_arr.max():.4f}"
    )

    # =========================================================================
    # C3: scan for saved best_state.pt files in Phase 5 cells
    # =========================================================================
    _log("=" * 60)
    _log("C3: looking for saved Phase 5 LSTM checkpoints")
    _log("=" * 60)
    art_root = ROOT / "artifacts" / "v7_stage2_full"
    candidate_dirs = sorted(art_root.glob("v7_lstm_fedavg_iid_*"))
    _log(f"  found {len(candidate_dirs)} v7_lstm_fedavg_iid_* dirs")
    pt_paths = []
    for d in candidate_dirs[:5]:
        pt = d / "best.pt"
        if pt.exists():
            pt_paths.append(str(pt))
    _log(f"  best.pt files in first 5 dirs: {len(pt_paths)}")
    embedding_norm_summary: dict | None = None
    if pt_paths:
        try:
            import torch
            sd = torch.load(pt_paths[0], map_location="cpu", weights_only=True)
            _log(f"  loaded {pt_paths[0]} keys={len(sd)}")
            cat_norms = {}
            cat_keys = ["bs_id", "slice_id", "sched", "tr"]
            for k, v in sd.items():
                for cat in cat_keys:
                    if cat in k.lower() and "embed" in k.lower():
                        cat_norms.setdefault(cat, []).append(
                            (k, float(v.norm().item()), tuple(v.shape))
                        )
            for cat, entries in cat_norms.items():
                for k, n, sh in entries:
                    _log(f"  {cat:10s} norm={n:.4f} key={k} shape={sh}")
            embedding_norm_summary = {
                cat: [{"key": k, "norm": n, "shape": list(sh)} for (k, n, sh) in entries]
                for cat, entries in cat_norms.items()
            }
        except Exception as exc:
            _log(f"  could not inspect checkpoint: {exc}")
            embedding_norm_summary = {"error": str(exc)}
    else:
        _log("  no checkpoints found; C3 inconclusive")
        embedding_norm_summary = None

    # =========================================================================
    # Final summary
    # =========================================================================
    summary = {
        "wall_clock_s": time.time() - t0,
        "C1_per_feature_pairwise_bs_KL": feature_kl_summary,
        "C1_top5_discriminating_features": [
            {"feature": f, **info} for f, info in sorted_feats[:5]
        ],
        "C2_dirichlet_sample_sizes": {str(k): v for k, v in sample_size_summary.items()},
        "C2_natural_by_bs_sample_sizes": nat_sizes_int,
        "C3_lstm_iid_checkpoint_embedding_norms": embedding_norm_summary,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    _log(f"wrote {OUT_JSON}")

    # Markdown narrative
    md = ["# Step 2 mechanism search\n",
          f"Generated by `scripts/step2_mechanism_search.py` in {summary['wall_clock_s']:.1f}s.\n\n",
          "## C1 — bs-discriminative power of continuous features\n",
          "Mean pairwise KL (bs_a vs bs_b) per continuous feature. "
          "Higher = bs's have more distinct distributions on this feature → "
          "natural-by-BS preserves bs-specific signal.\n\n",
          "| feature | mean KL | max KL |\n|---|---|---|\n"]
    for f, info in sorted_feats:
        md.append(f"| {f} | {info['kl_mean']:.4f} | {info['kl_max']:.4f} |\n")
    md.append("\n## C2 — per-client sample sizes under Dirichlet partition\n\n")
    md.append("| α | min | max | mean | min/max ratio |\n|---|---|---|---|---|\n")
    for a, info in sample_size_summary.items():
        md.append(f"| {a} | {info['min']:,} | {info['max']:,} | {info['mean']:.0f} | {info['min_over_max_ratio']:.4f} |\n")
    md.append(f"| natural-by-BS | {int(nat_arr.min()):,} | {int(nat_arr.max()):,} | {nat_arr.mean():.0f} | {nat_arr.min()/nat_arr.max():.4f} |\n\n")
    md.append("## C3 — categorical embedding norms in saved Phase 5 checkpoint\n")
    if embedding_norm_summary and "error" not in embedding_norm_summary:
        md.append("| categorical | norm | shape |\n|---|---|---|\n")
        for cat, entries in embedding_norm_summary.items():
            for e in entries:
                md.append(f"| {cat} | {e['norm']:.4f} | {e['shape']} |\n")
    else:
        md.append("- No checkpoint inspection (no best.pt found, or load failed)\n")
    OUT_MD.write_text("".join(md))
    _log(f"wrote {OUT_MD}")
    _log(f"DONE in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
