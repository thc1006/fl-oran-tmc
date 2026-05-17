"""Path D aggregator — paper-grade Δ AUC table vs Phase 5 baselines.

Reads V100 Path D sweep cells from ``artifacts/v7_sam_family/`` and pairs
them against matched Phase 5 baselines in ``artifacts/v7_stage2_full/``
to compute mean Δ AUC + paired-bootstrap CI95 + Wilcoxon p per
(arch, algo, partition) group.

Pairing rules per paper §2.6 mechanism:

* ``fedscam`` / ``fedmoswa`` → ``fedavg``  (both FedAvg-class + extra step)
* ``fedgmt``                → ``fedadam``  (FedGMT is adaptive)

This script is **safe to re-run any time during the sweep** — it
gracefully handles partial completion. Groups with < 3 paired seeds
get ``n=X (preliminary)`` markers; the script never errors on missing
cells.

Outputs:

* ``docs/RESULTS_V7_PATH_D.md`` — paper-grade table sorted winners-first
* ``artifacts/v7_sam_family/_aggregate_path_d.json`` — machine-readable

Usage::

    python scripts/aggregate_path_d.py \\
        --sweep-dir artifacts/v7_sam_family \\
        --phase5-dir artifacts/v7_stage2_full \\
        --output docs/RESULTS_V7_PATH_D.md \\
        [--n-boot 10000]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


# Re-use load_cells + paired_bootstrap_delta from the existing aggregator.
# Direct import is fragile because aggregate_v7_results imports heavy deps
# at module-load; use importlib for cleaner isolation.
_AGG_PATH = Path(__file__).resolve().parent / "aggregate_v7_results.py"
_spec = importlib.util.spec_from_file_location("_agg_v7", _AGG_PATH)
_agg = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("_agg_v7", _agg)
_spec.loader.exec_module(_agg)
load_cells = _agg.load_cells
paired_bootstrap_delta = _agg.paired_bootstrap_delta


# Path D scope
PATH_D_ARCHS = ("lstm", "mamba", "spiking_expand2")
PATH_D_ALGOS = ("fedscam", "fedgmt", "fedmoswa")
PATH_D_PARTITIONS = [
    ("iid", None),
    ("dirichlet", 0.05),
    ("dirichlet", 0.10),
    ("dirichlet", 0.50),
    ("dirichlet", 1.00),
    ("dirichlet", 5.00),
]
PATH_D_SEEDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 42)

# Pairing rule
ALGO_TO_BASELINE = {
    "fedscam":  "fedavg",
    "fedmoswa": "fedavg",
    "fedgmt":   "fedadam",
}

ARCH_DISPLAY = {
    "lstm":             "LSTM",
    "mamba":            "Mamba",
    "spiking_expand2":  "Spiking",
}
ALGO_DISPLAY = {
    "fedscam":  "FedSCAM",
    "fedgmt":   "FedGMT",
    "fedmoswa": "FedMoSWA",
    "fedavg":   "FedAvg",
    "fedadam":  "FedAdam",
}


def partition_tag(pmode: str, alpha) -> str:
    """Human-readable partition label."""
    if pmode == "iid":
        return "IID"
    if alpha is None:
        return f"{pmode} (α=?)"
    return f"Dir α={alpha:.2f}".rstrip("0").rstrip(".")


def partition_sort_key(pmode_alpha: tuple) -> tuple:
    """IID first, then dirichlet by descending heterogeneity (lower α first)."""
    pmode, alpha = pmode_alpha
    if pmode == "iid":
        return (0, 0.0)
    return (1, alpha if alpha is not None else 99.0)


def select_path_d_cells(cells: dict) -> dict:
    """Subset cells to Path D scope (3 archs × 3 algos × 6 partitions × 10 seeds)."""
    out = {}
    for key, summary in cells.items():
        arch, algo, pmode, alpha, seed = key
        if arch not in PATH_D_ARCHS:
            continue
        if algo not in PATH_D_ALGOS:
            continue
        if seed not in PATH_D_SEEDS:
            continue
        if (pmode, alpha) not in PATH_D_PARTITIONS:
            continue
        out[key] = summary
    return out


def select_baseline_cells(cells: dict, algos: tuple = ("fedavg", "fedadam")) -> dict:
    """Subset to Phase 5 baselines under Path D scope (same archs/partitions/seeds)."""
    out = {}
    for key, summary in cells.items():
        arch, algo, pmode, alpha, seed = key
        if arch not in PATH_D_ARCHS:
            continue
        if algo not in algos:
            continue
        if seed not in PATH_D_SEEDS:
            continue
        if (pmode, alpha) not in PATH_D_PARTITIONS:
            continue
        out[key] = summary
    return out


def compute_group_stats(merged: dict, arch: str, algo: str,
                        pmode: str, alpha, n_boot: int) -> dict:
    """Compute mean Δ + paired-bootstrap CI95 + Wilcoxon p for one group.

    Returns a dict with keys: n, path_d_mean, path_d_std, baseline_mean,
    baseline_std, delta_mean, ci_lo, ci_hi, wilcoxon_p, sig.
    """
    base_algo = ALGO_TO_BASELINE[algo]
    a_axes = dict(arch=arch, algorithm=algo, partition_mode=pmode, alpha=alpha)
    b_axes = dict(arch=arch, algorithm=base_algo,
                  partition_mode=pmode, alpha=alpha)
    delta_result = paired_bootstrap_delta(
        merged, a=a_axes, b=b_axes, n_boot=n_boot,
    )
    # Path D auc descriptors
    path_d_aucs = [
        s["test"]["auc"] for k, s in merged.items()
        if k[:4] == (arch, algo, pmode, alpha)
    ]
    base_aucs = [
        s["test"]["auc"] for k, s in merged.items()
        if k[:4] == (arch, base_algo, pmode, alpha)
    ]
    n_path = len(path_d_aucs)
    n_base = len(base_aucs)

    out = {
        "arch": arch, "algo": algo, "partition_mode": pmode, "alpha": alpha,
        "base_algo": base_algo,
        "n_path_d": n_path, "n_baseline": n_base,
        "n_paired": delta_result.get("n_paired_seeds", 0),
        "path_d_mean": float(np.mean(path_d_aucs)) if path_d_aucs else None,
        "path_d_std":  float(np.std(path_d_aucs, ddof=1)) if n_path >= 2 else None,
        "baseline_mean": float(np.mean(base_aucs)) if base_aucs else None,
        "baseline_std":  float(np.std(base_aucs, ddof=1)) if n_base >= 2 else None,
        "delta_mean": delta_result.get("delta_mean"),
        "ci_lo": delta_result.get("ci_lo"),
        "ci_hi": delta_result.get("ci_hi"),
        "wilcoxon_p": delta_result.get("wilcoxon_p"),
        "warning": delta_result.get("warning"),
        "paired_seeds": delta_result.get("seeds", []),
    }
    # Significance verdict
    if out["ci_lo"] is None or out["ci_hi"] is None:
        out["sig"] = f"prelim n={out['n_paired']}"
    elif out["ci_lo"] > 0:
        out["sig"] = "★ 顯著優"
    elif out["ci_hi"] < 0:
        out["sig"] = "↓ 顯著劣"
    else:
        out["sig"] = "— 無顯著"
    return out


def render_markdown(stats: list[dict], n_path_d_cells: int, n_phase5_cells: int) -> str:
    """Render the full markdown report."""
    lines = []
    p = lines.append

    p("# Path D — SAM-family multi-arch sweep aggregate")
    p("")
    p(f"**Generated**: {datetime.now():%Y-%m-%d %H:%M:%S}")
    p(f"**Path D cells discovered**: {n_path_d_cells}")
    p(f"**Phase 5 baseline cells**: {n_phase5_cells}")
    p("")
    p("Pairing rule (per paper §2.6):")
    p("- `fedscam`, `fedmoswa` → `fedavg` (FedAvg-class + extra step)")
    p("- `fedgmt` → `fedadam` (adaptive)")
    p("")
    p("Significance: ★ = paired-bootstrap CI95 excludes 0 positively; "
      "↓ = CI95 excludes 0 negatively; — = CI95 straddles 0; "
      "`prelim` = n < 3 paired seeds (CI not computed).")
    p("")

    # One table per arch
    for arch in PATH_D_ARCHS:
        arch_stats = [s for s in stats if s["arch"] == arch]
        if not any(s["n_paired"] > 0 for s in arch_stats):
            p(f"## {ARCH_DISPLAY[arch]} (no paired data yet)")
            p("")
            continue
        p(f"## {ARCH_DISPLAY[arch]}")
        p("")
        p("| Algo | Partition | n paired | Path D AUC | Baseline AUC | Δ AUC | 95% CI | Wilcoxon p | sig |")
        p("|---|---|---:|---|---|---:|---|---:|---|")
        # Sort by partition then algo for stable reading order
        arch_stats.sort(key=lambda s: (
            partition_sort_key((s["partition_mode"], s["alpha"])),
            s["algo"],
        ))
        for s in arch_stats:
            algo_d = ALGO_DISPLAY[s["algo"]]
            part_d = partition_tag(s["partition_mode"], s["alpha"])
            n_pair = s["n_paired"]
            pd_m = s["path_d_mean"]
            pd_s = s["path_d_std"]
            bs_m = s["baseline_mean"]
            bs_s = s["baseline_std"]
            d_m = s["delta_mean"]
            ci_lo = s["ci_lo"]
            ci_hi = s["ci_hi"]
            wp = s["wilcoxon_p"]
            sig = s["sig"]
            pd_str = (
                f"{pd_m:.4f} ± {pd_s:.4f}" if pd_s is not None
                else (f"{pd_m:.4f}" if pd_m is not None else "—")
            )
            bs_str = (
                f"{bs_m:.4f} ± {bs_s:.4f}" if bs_s is not None
                else (f"{bs_m:.4f}" if bs_m is not None else "—")
            )
            d_str = f"{d_m:+.4f}" if d_m is not None else "—"
            ci_str = (
                f"[{ci_lo:+.4f}, {ci_hi:+.4f}]"
                if (ci_lo is not None and ci_hi is not None) else "—"
            )
            wp_str = f"{wp:.4f}" if wp is not None else "—"
            p(f"| {algo_d} | {part_d} | {n_pair} | {pd_str} | {bs_str} | {d_str} | {ci_str} | {wp_str} | {sig} |")
        p("")

    # Aggregate summary
    n_win = sum(1 for s in stats if s["sig"] == "★ 顯著優")
    n_lose = sum(1 for s in stats if s["sig"] == "↓ 顯著劣")
    n_null = sum(1 for s in stats if s["sig"] == "— 無顯著")
    n_prelim = sum(1 for s in stats if s["sig"].startswith("prelim"))
    p("## Aggregate verdict")
    p("")
    p(f"- Significant wins (★): **{n_win}**")
    p(f"- Significant losses (↓): **{n_lose}**")
    p(f"- No significant difference (—): **{n_null}**")
    p(f"- Preliminary (n<3 paired): **{n_prelim}**")
    p("")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", type=Path,
                    default=Path("artifacts/v7_sam_family"))
    ap.add_argument("--phase5-dir", type=Path,
                    default=Path("artifacts/v7_stage2_full"))
    ap.add_argument("--output", type=Path,
                    default=Path("docs/RESULTS_V7_PATH_D.md"))
    ap.add_argument("--out-json", type=Path,
                    default=Path("artifacts/v7_sam_family/_aggregate_path_d.json"))
    ap.add_argument("--n-boot", type=int, default=10_000)
    args = ap.parse_args()

    # Discover cells from both dirs
    path_d_raw = load_cells(args.sweep_dir) if args.sweep_dir.exists() else {}
    phase5_raw = load_cells(args.phase5_dir) if args.phase5_dir.exists() else {}

    path_d_cells = select_path_d_cells(path_d_raw)
    baseline_cells = select_baseline_cells(phase5_raw)

    # Build merged dict for paired_bootstrap_delta consumption
    merged = {}
    merged.update(path_d_cells)
    merged.update(baseline_cells)

    print(f"Path D cells (in scope): {len(path_d_cells)}", file=sys.stderr)
    print(f"Phase 5 baseline cells:  {len(baseline_cells)}", file=sys.stderr)

    # Compute stats per (arch, algo, partition) group
    stats = []
    for arch in PATH_D_ARCHS:
        for algo in PATH_D_ALGOS:
            for pmode, alpha in PATH_D_PARTITIONS:
                stat = compute_group_stats(merged, arch, algo, pmode, alpha,
                                           n_boot=args.n_boot)
                stats.append(stat)

    md = render_markdown(stats, len(path_d_cells), len(baseline_cells))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md)
    print(f"wrote {args.output}", file=sys.stderr)

    # JSON output (machine-readable)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    out_payload = {
        "generated": datetime.now().isoformat(),
        "path_d_cells_in_scope": len(path_d_cells),
        "phase5_baseline_cells": len(baseline_cells),
        "groups": stats,
    }
    args.out_json.write_text(json.dumps(out_payload, indent=2, default=str))
    print(f"wrote {args.out_json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
