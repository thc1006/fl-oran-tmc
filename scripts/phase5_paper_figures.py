"""Phase 5 paper-figure generators.

Reads ``artifacts/v7_stage2_full/v7_*/summary.json`` (NVML energy +
phase timings + best_val_auc) and emits the three paper-side
artefacts that motivated this Phase 5 sweep:

  * ``pareto.png`` — Energy vs AUC scatter. Each point = one
    ``(arch, algo, partition)`` group; x = mean model-attributable
    energy, y = mean best_val_auc, error bars = std across seeds.
    Color encodes arch (LSTM/Mamba/Spiking), marker encodes algo
    family. The visual headline of paper Tier-2 (energy-Pareto).

  * ``interaction_heatmap.png`` — 3-row heatmap (one row per arch).
    Each cell = mean best_val_auc across seeds for an
    ``(algo, partition)`` pair, annotated with mean ± std and
    ``n=<seeds>``. Tier-1 contribution: does the algorithm ranking
    flip across architectures?

  * ``algo_ranking.png`` — Per-arch bar chart of mean Dirichlet-only
    AUC (IID excluded as the upper-bound reference). Quickly
    answers "which algo dominates on this arch under stress".

  * ``results_table.csv`` — long-form group table (one row per
    ``(arch, algo, partition_mode, alpha)``) with seeds collapsed
    to mean/std. Drop-in for paper Table 1.

The script reads only summary.json (never history.csv) so it is
safe to run while the sweep is in progress; partial data renders
empty cells in the heatmap rather than failing.

Usage:
    python scripts/phase5_paper_figures.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts" / "v7_stage2_full"
# GH#9: figures must include Phase 6 + T-ABLATION cells alongside the
# Phase 5 main sweep. Each sweep lives in its own artifact directory;
# load_cells walks all of them and yields a unified DataFrame.
SWEEP_DIRS = [
    ART,
    ROOT / "artifacts" / "v7_phase6_per_bs_dirichlet",
    ROOT / "artifacts" / "v7_phase6_threshold",
    ROOT / "artifacts" / "v7_ablation_random_split",
]
DEFAULT_OUT_DIR = ROOT / "artifacts" / "figures"

ARCH_COLORS = {
    "lstm": "#1f77b4",
    "mamba": "#ff7f0e",
    "spiking_expand2": "#d62728",
}
ARCH_LABELS = {
    "lstm": "LSTM",
    "mamba": "Mamba",
    "spiking_expand2": "Spiking-SSM",
}
ALGO_MARKERS = {
    "fedavg":   "o",
    "fedprox":  "s",
    "fedadam":  "^",
    "scaffold": "D",
    "feddyn":   "v",
}
ALGO_LABELS = {
    "fedavg":   "FedAvg",
    "fedprox":  "FedProx",
    "fedadam":  "FedAdam",
    "scaffold": "SCAFFOLD",
    "feddyn":   "FedDyn",
}
# Order: IID first (upper bound), then dirichlet from least to most
# heterogeneous (large alpha = closer to IID; small alpha = stress).
PARTITION_ORDER = [
    ("iid",       None),
    ("dirichlet", 5.00),
    ("dirichlet", 1.00),
    ("dirichlet", 0.50),
    ("dirichlet", 0.10),
    ("dirichlet", 0.05),
]
PARTITION_LABELS = {
    ("iid",       None):  "natural\n(by-BS)",
    ("dirichlet", 5.00):  r"$\alpha$=5.0",
    ("dirichlet", 1.00):  r"$\alpha$=1.0",
    ("dirichlet", 0.50):  r"$\alpha$=0.5",
    ("dirichlet", 0.10):  r"$\alpha$=0.1",
    ("dirichlet", 0.05):  r"$\alpha$=0.05",
}


def parse_cell_name(name: str) -> dict | None:
    """Decode v7_<arch>_<algo>_<part>[_a<alpha>]_n<N>_s<seed>.

    Returns None for malformed names. Architecture token must be
    matched longest-first so ``spiking_expand2`` is not split into
    ``spiking`` + ``expand2``.
    """
    if not name.startswith("v7_"):
        return None
    s = name[len("v7_"):]

    arch = None
    for a in ("spiking_expand2", "mamba", "lstm"):
        if s.startswith(a + "_"):
            arch = a
            s = s[len(a) + 1:]
            break
    if arch is None:
        return None

    algo = None
    for a in ALGO_MARKERS:
        if s.startswith(a + "_"):
            algo = a
            s = s[len(a) + 1:]
            break
    if algo is None:
        return None

    # GH#9: branch on partition_mode prefix; Phase 6 introduced
    # ``perbsdir`` (Rank 3 per-BS Dirichlet, no n_clients in name) and
    # ``randsplit`` (T-ABLATION, no alpha). Standard ``iid`` /
    # ``dirichlet`` cells may also carry an optional ``_t<thr>`` suffix
    # (Rank 1 threshold-sensitivity ablation, e.g. _t05 = 5% BLER).
    threshold: float | None = None
    if s.startswith("perbsdir_"):
        partition_mode = "per_bs_dirichlet"
        s = s[len("perbsdir_"):]
        if not s.startswith("a"):
            return None
        apart, _, s = s.partition("_")
        try:
            alpha = float(apart[1:].replace("p", "."))
        except ValueError:
            return None
        # No n_clients token in this format; default to 7 (the per-BS
        # partition implies the natural base-station count).
        n_clients = 7
    elif s.startswith("randsplit_"):
        partition_mode = "random_split"
        alpha = None
        s = s[len("randsplit_"):]
        if not s.startswith("n"):
            return None
        npart, _, s = s.partition("_")
        try:
            n_clients = int(npart[1:])
        except ValueError:
            return None
    elif s.startswith("iid_"):
        partition_mode = "iid"
        alpha = None
        s = s[len("iid_"):]
        if not s.startswith("n"):
            return None
        npart, _, s = s.partition("_")
        try:
            n_clients = int(npart[1:])
        except ValueError:
            return None
    elif s.startswith("dirichlet_"):
        partition_mode = "dirichlet"
        s = s[len("dirichlet_"):]
        if not s.startswith("a"):
            return None
        apart, _, s = s.partition("_")
        try:
            alpha = float(apart[1:].replace("p", "."))
        except ValueError:
            return None
        if not s.startswith("n"):
            return None
        npart, _, s = s.partition("_")
        try:
            n_clients = int(npart[1:])
        except ValueError:
            return None
    else:
        return None

    if not s.startswith("s"):
        return None
    # Seed segment may be plain ``s<seed>`` or ``s<seed>_t<thr>`` (the
    # Rank 1 threshold suffix encodes the BLER gate as percent ×10:
    # _t05 → 0.05, _t15 → 0.15, _t20 → 0.20).
    seed_tail = s[1:]
    if "_t" in seed_tail:
        seed_str, thr_str = seed_tail.split("_t", 1)
        try:
            threshold = int(thr_str) / 100.0
        except ValueError:
            return None
    else:
        seed_str = seed_tail
    try:
        seed = int(seed_str)
    except ValueError:
        return None
    return {
        "arch": arch, "algo": algo,
        "partition_mode": partition_mode, "alpha": alpha,
        "n_clients": n_clients, "seed": seed,
        "threshold": threshold,
    }


def load_cells() -> pd.DataFrame:
    """Walk every directory in ``SWEEP_DIRS`` and return one row per
    completed cell (any sweep) with metadata, AUC, energy, timing.
    Skips in-flight cells (no summary.json) and missing sweep dirs.

    GH#9: combining Phase 5 main sweep + Phase 6 mechanism ablations +
    T-ABLATION random_split into a unified DataFrame so the paper
    figures show all cells rather than silently dropping the Phase 6
    rows that live in separate artifact directories.
    """
    rows = []
    cells_iter = (
        d for sweep_dir in SWEEP_DIRS
        if sweep_dir.exists()
        for d in sorted(sweep_dir.glob("v7_*"))
    )
    for d in cells_iter:
        sj = d / "summary.json"
        if not sj.exists():
            continue
        meta = parse_cell_name(d.name)
        if meta is None:
            continue
        try:
            s = json.loads(sj.read_text())
        except Exception:
            continue
        pt = s.get("phase_timings_s", {}) or {}
        eg = s.get("energy_measured", {}) or {}
        rows.append({
            **meta,
            "name": d.name,
            "best_val_auc": float(s.get("best_val_auc", float("nan"))),
            "test_auc": float(s.get("test_auc", float("nan"))),
            "TOTAL_s": float(pt.get("TOTAL", float("nan"))),
            "steady_round_s": float(pt.get("7b_steady_round_mean", float("nan"))),
            "energy_total_J": float(eg.get("training_total_mJ", float("nan"))) / 1e3,
            "energy_model_J": float(
                eg.get("training_model_attributable_mJ", float("nan"))
            ) / 1e3,
            "energy_idle_J": float(
                eg.get("training_idle_attributed_mJ", float("nan"))
            ) / 1e3,
        })
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse seeds within (arch, algo, partition_mode, alpha) groups."""
    if df.empty:
        return df
    agg = df.groupby(
        ["arch", "algo", "partition_mode", "alpha"], dropna=False, as_index=False
    ).agg(
        n_seeds=("seed", "nunique"),
        auc_mean=("best_val_auc", "mean"),
        auc_std=("best_val_auc", "std"),
        test_auc_mean=("test_auc", "mean"),
        energy_total_J_mean=("energy_total_J", "mean"),
        energy_total_J_std=("energy_total_J", "std"),
        energy_model_J_mean=("energy_model_J", "mean"),
        energy_model_J_std=("energy_model_J", "std"),
        steady_round_s_mean=("steady_round_s", "mean"),
    )
    # std is NaN for n=1; replace with 0 for cleaner plotting.
    for c in ("auc_std", "energy_total_J_std", "energy_model_J_std"):
        agg[c] = agg[c].fillna(0.0)
    return agg.sort_values(["arch", "algo", "partition_mode", "alpha"]).reset_index(drop=True)


def plot_pareto(agg: pd.DataFrame, out_path: Path) -> None:
    """Scatter: x=model-attributable energy (J), y=mean best_val_auc.
    One marker per (arch, algo, partition) group; error bars = std
    across seeds. Solid markers = full 10 seeds, hollow = partial."""
    fig, ax = plt.subplots(figsize=(10, 6.5))
    if agg.empty:
        ax.text(0.5, 0.5, "no completed cells yet",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=14, color="gray")
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return

    seen_legend: set[tuple[str, str]] = set()
    for _, row in agg.iterrows():
        arch = row["arch"]; algo = row["algo"]
        if arch not in ARCH_COLORS or algo not in ALGO_MARKERS:
            continue
        full = row["n_seeds"] >= 10
        ax.errorbar(
            row["energy_model_J_mean"], row["auc_mean"],
            xerr=row["energy_model_J_std"], yerr=row["auc_std"],
            fmt=ALGO_MARKERS[algo], color=ARCH_COLORS[arch],
            markersize=9 if full else 6,
            markerfacecolor=ARCH_COLORS[arch] if full else "white",
            markeredgewidth=1.4,
            capsize=2.5, elinewidth=0.6, alpha=0.85,
        )
        seen_legend.add((arch, algo))

    # Build a clean two-axis legend: arch (color) + algo (marker)
    arch_handles = [
        plt.Line2D([0], [0], marker="o", color=c, markersize=9,
                   linestyle="", label=ARCH_LABELS[a])
        for a, c in ARCH_COLORS.items()
    ]
    algo_handles = [
        plt.Line2D([0], [0], marker=m, color="black", markersize=9,
                   linestyle="", label=ALGO_LABELS[a],
                   markerfacecolor="white", markeredgewidth=1.2)
        for a, m in ALGO_MARKERS.items()
    ]
    leg1 = ax.legend(handles=arch_handles, loc="upper left",
                     title="Architecture", fontsize=9, framealpha=0.85)
    ax.add_artist(leg1)
    ax.legend(handles=algo_handles, loc="lower right",
              title="Algorithm", fontsize=9, framealpha=0.85)

    ax.set_xlabel("Model-attributable energy per cell  [J]")
    ax.set_ylabel(r"$\overline{\mathrm{best\_val\_auc}}$ (mean across seeds)")
    ax.set_title(
        "Phase 5 — Pareto: AUC vs model-attributable energy "
        f"({len(agg)} groups, {int(agg['n_seeds'].sum())} cells)"
    )
    ax.grid(alpha=0.3)
    ax.text(0.99, 0.02,
            "solid marker = full 10 seeds   hollow = partial",
            transform=ax.transAxes, ha="right", fontsize=8,
            color="gray", style="italic")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_interaction_heatmap(agg: pd.DataFrame, out_path: Path) -> None:
    """3-row heatmap (one per arch). Each cell = mean AUC for the
    (algo, partition) group, annotated with mean and n_seeds."""
    archs = list(ARCH_COLORS.keys())
    algos = list(ALGO_MARKERS.keys())
    parts = PARTITION_ORDER

    fig, axes = plt.subplots(
        len(archs), 1, figsize=(9, 2.4 * len(archs)),
        sharex=True, gridspec_kw={"hspace": 0.4},
    )
    if len(archs) == 1:
        axes = [axes]

    # Compute global vmin/vmax for shared colormap (only over real data)
    valid = agg[agg["auc_mean"].notna()]
    if len(valid):
        vmin = max(0.50, float(valid["auc_mean"].min()) - 0.02)
        vmax = min(1.00, float(valid["auc_mean"].max()) + 0.02)
    else:
        vmin, vmax = 0.50, 1.00

    for ax, arch in zip(axes, archs):
        Z = np.full((len(algos), len(parts)), np.nan)
        N = np.zeros((len(algos), len(parts)), dtype=int)
        for i, algo in enumerate(algos):
            for j, (pmode, alpha) in enumerate(parts):
                m = (
                    (agg["arch"] == arch)
                    & (agg["algo"] == algo)
                    & (agg["partition_mode"] == pmode)
                )
                if pmode == "dirichlet":
                    m = m & (agg["alpha"] == alpha)
                else:
                    m = m & (agg["alpha"].isna())
                row = agg[m]
                if len(row):
                    Z[i, j] = float(row["auc_mean"].iloc[0])
                    N[i, j] = int(row["n_seeds"].iloc[0])

        im = ax.imshow(
            Z, vmin=vmin, vmax=vmax, cmap="RdYlGn",
            aspect="auto", origin="upper",
        )
        ax.set_yticks(range(len(algos)))
        ax.set_yticklabels([ALGO_LABELS[a] for a in algos], fontsize=9)
        ax.set_xticks(range(len(parts)))
        ax.set_xticklabels([PARTITION_LABELS[p] for p in parts], fontsize=9)
        ax.set_title(ARCH_LABELS[arch], fontsize=11,
                     loc="left", color=ARCH_COLORS[arch], fontweight="bold")

        # annotate cells
        for i in range(len(algos)):
            for j in range(len(parts)):
                if np.isnan(Z[i, j]):
                    ax.text(j, i, "—", ha="center", va="center",
                            color="gray", fontsize=9)
                else:
                    txt = f"{Z[i,j]:.3f}\nn={N[i,j]}"
                    color = "white" if (Z[i, j] - vmin) / (vmax - vmin) < 0.4 else "black"
                    ax.text(j, i, txt, ha="center", va="center",
                            color=color, fontsize=8)

    # Single shared colorbar
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("mean best_val_auc", fontsize=9)

    fig.suptitle(
        "Phase 5 — algorithm × partition heatmap, per architecture\n"
        "(does algo ranking flip across architectures?)",
        fontsize=12, y=1.0,
    )
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_algo_ranking(agg: pd.DataFrame, out_path: Path) -> None:
    """Per-arch bar chart: mean TEST AUC averaged over Dirichlet partitions
    only (IID excluded — that's the upper bound, not a stress test). Quick
    read for which algo dominates on which arch.

    Switched from best_val to test AUC per reviewer Minor#2 (P1.4b-GREEN
    2026-05-06): headline figures must use the same metric the body claims
    are evaluated on, to avoid selection-bias appearance."""
    archs = list(ARCH_COLORS.keys())
    algos = list(ALGO_MARKERS.keys())

    fig, axes = plt.subplots(1, len(archs), figsize=(13, 4.5), sharey=True)
    if len(archs) == 1:
        axes = [axes]

    only_dir = agg[agg["partition_mode"] == "dirichlet"]
    for ax, arch in zip(axes, archs):
        sub = only_dir[only_dir["arch"] == arch]
        means = []
        labels = []
        ns = []
        for algo in algos:
            r = sub[sub["algo"] == algo]
            if len(r):
                means.append(float(r["test_auc_mean"].mean()))
                ns.append(int(r["n_seeds"].sum()))
            else:
                means.append(np.nan)
                ns.append(0)
            labels.append(ALGO_LABELS[algo])

        xs = np.arange(len(algos))
        bars = ax.bar(xs, [m if not np.isnan(m) else 0 for m in means],
                      color=ARCH_COLORS[arch], alpha=0.6,
                      edgecolor=ARCH_COLORS[arch], linewidth=1.2)
        for b, m, n in zip(bars, means, ns):
            if np.isnan(m):
                continue
            ax.text(b.get_x() + b.get_width() / 2, m + 0.005,
                    f"{m:.3f}\n(n={n})", ha="center", va="bottom",
                    fontsize=8)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_title(ARCH_LABELS[arch], color=ARCH_COLORS[arch], fontweight="bold")
        ax.set_ylim(0.5, 1.0)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("mean test AUC  (Dirichlet partitions only)")

    fig.suptitle("Phase 5 — algorithm ranking per architecture (Dirichlet stress avg, test AUC)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    df = load_cells()
    if df.empty:
        print(f"no completed cells under {ART}", file=sys.stderr)
        return 1
    agg = aggregate(df)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        plot_pareto(agg, args.out_dir / f"pareto.{ext}")
        plot_interaction_heatmap(agg, args.out_dir / f"interaction_heatmap.{ext}")
        plot_algo_ranking(agg, args.out_dir / f"algo_ranking.{ext}")
    agg.to_csv(args.out_dir / "results_table.csv", index=False)

    print(f"saved 3 figures (png+pdf+svg) + results_table.csv to {args.out_dir}")
    print(f"  total cells loaded:  {len(df)}")
    print(f"  groups (arch×algo×partition): {len(agg)}")
    print(f"  groups with full 10 seeds:    {int((agg['n_seeds']>=10).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
