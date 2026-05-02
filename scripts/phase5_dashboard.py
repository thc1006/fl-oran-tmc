"""Phase 5 v5 live dashboard.

Reads artifacts/v7_stage2_full/v7_*/{summary.json, history.csv} and
emits a 2x2 multi-panel PNG showing:

  Panel A: Cross-cell timing (TOTAL_s vs cell index) — drift detector.
           A flat or noise-bounded line confirms the dynamo cache fix
           held; an upward slope means the bug is back.

  Panel B: Within-cell per-round duration — per-cell line plot of
           history.csv duration_s. Detects within-cell slowdown
           (rounds get slower as a single cell trains).

  Panel C: val_auc trajectory per cell — model convergence visual.

  Panel D: Aggregate progress — stacked bar of cells done by arch +
           text panel with current cell name / round / AUC / ETA.

Usage:
    python scripts/phase5_dashboard.py [--output PATH]

Live refresh:
    watch -n 60 'python scripts/phase5_dashboard.py'

The script is read-only on artifacts/ and safe to run while the
sweep is in progress.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts" / "v7_stage2_full"
DEFAULT_OUT = ROOT / "artifacts" / "phase5_dashboard.png"

ARCH_COLORS = {
    "lstm": "#1f77b4",       # blue
    "mamba": "#ff7f0e",      # orange
    "spiking_expand2": "#d62728",  # red
}
ARCH_TARGET_TOTAL_S = {  # rough per-cell target at 100r (post-fix steady)
    "lstm": 200.0,
    "mamba": 280.0,
    "spiking_expand2": 600.0,
}
# Per-algo color (used in Panel A to differentiate within single-arch phase)
ALGO_COLORS = {
    "fedavg":   "#1f77b4",  # blue
    "fedprox":  "#2ca02c",  # green
    "fedadam":  "#ff7f0e",  # orange
    "scaffold": "#9467bd",  # purple
    "feddyn":   "#d62728",  # red
}
# Per-partition color (used in Panel C to show IID vs Dirichlet stratification)
# IID natural is reference (high AUC); Dirichlet alphas go from blue (mild
# stress) to red (extreme stress) to match the inverted α pattern we found.
PARTITION_COLORS = {
    "iid":         "#2ca02c",  # green = IID baseline (best)
    "5.00":        "#1f77b4",  # blue = mildest Dirichlet (worst we saw)
    "1.00":        "#1f77b4",  # blue = mild Dirichlet
    "0.50":        "#ff7f0e",  # orange = mid
    "0.10":        "#d62728",  # red = severe Dirichlet
    "0.05":        "#8c564b",  # brown = extreme Dirichlet (best Dirichlet)
}
TOTAL_CELLS = 897 + 3  # 3 preserved + 897 to run = 900


def parse_arch_from_name(name: str) -> str:
    """Extract arch token from cell name like v7_lstm_fedavg_iid_n7_s42."""
    parts = name.split("_")
    if "lstm" in parts:
        return "lstm"
    if "mamba" in parts:
        return "mamba"
    if "spiking" in parts:
        return "spiking_expand2"
    return "unknown"


def parse_algo_from_name(name: str) -> str:
    """Extract algo token from cell name."""
    for algo in ("fedavg", "fedprox", "fedadam", "scaffold", "feddyn"):
        if algo in name:
            return algo
    return "unknown"


def parse_partition_from_name(name: str) -> str:
    """Extract partition key — 'iid' or alpha string like '0.05'."""
    if "_iid_" in name:
        return "iid"
    import re as _re
    m = _re.search(r"_a(\d+p\d+)_", name)
    if m:
        return m.group(1).replace("p", ".")
    return "unknown"


def load_cells():
    """Walk artifacts/v7_stage2_full/v7_* and load each cell's summary +
    history. Cells without summary.json are treated as in-flight (only
    history.csv exists). Returns list of dicts sorted by directory mtime
    (= chronological cell completion order)."""
    if not ART.exists():
        return []
    cells = []
    for d in sorted(ART.glob("v7_*"), key=lambda p: p.stat().st_mtime):
        hist_path = d / "history.csv"
        sum_path = d / "summary.json"
        if not hist_path.exists():
            continue
        try:
            history = pd.read_csv(hist_path)
        except Exception:
            continue
        summary = None
        if sum_path.exists():
            try:
                summary = json.loads(sum_path.read_text())
            except Exception:
                summary = None
        cells.append({
            "name": d.name,
            "arch": parse_arch_from_name(d.name),
            "algo": parse_algo_from_name(d.name),
            "partition": parse_partition_from_name(d.name),
            "history": history,
            "summary": summary,
            "mtime": d.stat().st_mtime,
            "in_flight": summary is None,
        })
    return cells


def _aggregate_by_partition(cells):
    """Per-partition round-wise median + 5/95 percentile of val_auc."""
    import collections as _coll
    bucket = _coll.defaultdict(list)
    for c in cells:
        if c["in_flight"]:
            continue
        h = c["history"]
        if "val_auc" not in h.columns:
            continue
        bucket[c["partition"]].append(h[["round", "val_auc"]])
    stats = {}
    for p, dfs in bucket.items():
        merged = pd.concat(dfs)
        agg = merged.groupby("round")["val_auc"].agg(
            median="median",
            p5=lambda x: np.percentile(x, 5),
            p95=lambda x: np.percentile(x, 95),
            n="count",
        ).reset_index()
        stats[p] = agg
    return stats


def _partition_final_auc(cells):
    """For each partition, mean ± std of best_val_auc across completed cells."""
    import collections as _coll
    bucket = _coll.defaultdict(list)
    for c in cells:
        if c["in_flight"] or c["summary"] is None:
            continue
        auc = c["summary"].get("best_val_auc")
        if auc is None:
            continue
        bucket[c["partition"]].append(float(auc))
    return {p: {"mean": float(np.mean(v)),
                "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                "n": len(v)}
            for p, v in bucket.items()}


def _aggregate_round_duration(cells):
    """Round-wise median + 5/95 of duration_s across all completed cells."""
    rows = []
    for c in cells:
        h = c["history"]
        if "duration_s" not in h.columns:
            continue
        rows.append(h[["round", "duration_s"]])
    if not rows:
        return None
    merged = pd.concat(rows)
    return merged.groupby("round")["duration_s"].agg(
        median="median",
        p5=lambda x: np.percentile(x, 5),
        p95=lambda x: np.percentile(x, 95),
    ).reset_index()


def render(cells, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.5),
                             gridspec_kw={"hspace": 0.32, "wspace": 0.22})
    ((axA, axB), (axC, axD)) = axes

    completed = [c for c in cells if not c["in_flight"]]
    in_flight = [c for c in cells if c["in_flight"]]
    archs_seen = {c["arch"] for c in completed}
    color_by_algo = (len(archs_seen) <= 1)

    # ============ Panel A: cross-cell timing with phase boundaries ============
    if completed:
        # Detect algo-phase boundaries (consecutive transitions in cell order)
        boundaries = []  # list of (start_idx, algo)
        prev = None
        for i, c in enumerate(completed):
            if c["algo"] != prev:
                boundaries.append((i, c["algo"]))
                prev = c["algo"]

        # Plot points
        if color_by_algo:
            for algo, color in ALGO_COLORS.items():
                sub = [(i, c["summary"]["phase_timings_s"]["TOTAL"])
                       for i, c in enumerate(completed) if c["algo"] == algo]
                if not sub:
                    continue
                xs, ys = zip(*sub)
                axA.scatter(xs, ys, color=color, s=22, alpha=0.85,
                            edgecolor="white", linewidth=0.4, zorder=3)
        else:
            for arch in ARCH_COLORS:
                sub = [(i, c["summary"]["phase_timings_s"]["TOTAL"])
                       for i, c in enumerate(completed) if c["arch"] == arch]
                if not sub:
                    continue
                xs, ys = zip(*sub)
                axA.scatter(xs, ys, color=ARCH_COLORS[arch], s=22, alpha=0.85,
                            edgecolor="white", linewidth=0.4, zorder=3)

        # Tight Y-axis based on data
        ys_all = [c["summary"]["phase_timings_s"]["TOTAL"] for c in completed]
        y_min = min(ys_all) - 5
        y_max = max(ys_all) + 5
        axA.set_ylim(y_min, y_max)
        axA.set_xlim(-2, len(completed) + 2)

        # Phase boundary lines + labels at top
        for j, (start_idx, algo) in enumerate(boundaries):
            color = ALGO_COLORS.get(algo, "gray") if color_by_algo else \
                    ARCH_COLORS.get(completed[start_idx]["arch"], "gray")
            if j > 0:
                axA.axvline(start_idx - 0.5, color="gray",
                            linestyle="--", lw=0.6, alpha=0.5, zorder=1)
            # Label at top, computed midpoint of phase
            end_idx = boundaries[j+1][0] if j+1 < len(boundaries) else len(completed)
            mid = (start_idx + end_idx) / 2
            mean_total = np.mean([
                completed[k]["summary"]["phase_timings_s"]["TOTAL"]
                for k in range(start_idx, end_idx)
            ])
            axA.text(mid, y_max - (y_max - y_min) * 0.06, algo,
                     ha="center", fontsize=9, color=color, fontweight="bold",
                     bbox=dict(facecolor="white", edgecolor=color,
                               boxstyle="round,pad=0.2", alpha=0.9))
            axA.text(mid, y_min + (y_max - y_min) * 0.04,
                     f"{mean_total:.0f}s",
                     ha="center", fontsize=8, color=color,
                     style="italic", alpha=0.85)

    axA.set_xlabel("cell completion order (chronological)")
    axA.set_ylabel("seconds per cell")
    axA.set_title("A. Cross-cell timing — algorithm phases",
                  fontsize=11, fontweight="bold", loc="left")
    axA.grid(alpha=0.25, axis="y")

    # ============ Panel B: within-cell round timing — median + band ============
    dur_stats = _aggregate_round_duration([c for c in cells if not c["in_flight"]])
    if dur_stats is not None and len(dur_stats):
        axB.fill_between(dur_stats["round"], dur_stats["p5"], dur_stats["p95"],
                         color="#1f77b4", alpha=0.18, label="5–95% band")
        axB.plot(dur_stats["round"], dur_stats["median"],
                 color="#1f77b4", lw=1.8, label="median")
        # Annotate round-1 cold compile spike
        r1 = dur_stats[dur_stats["round"] == 1]
        if len(r1) and len(dur_stats) > 5:
            spike = float(r1["median"].iloc[0])
            steady = float(dur_stats[dur_stats["round"] >= 5]["median"].median())
            axB.annotate(
                f"r1 = {spike:.1f}s\n(cold compile)",
                xy=(1, spike), xytext=(15, spike + 0.15),
                fontsize=8, color="#666",
                arrowprops=dict(arrowstyle="->", color="#666", lw=0.7),
            )
            axB.axhline(steady, color="#999", ls=":", lw=0.6, alpha=0.7)
            axB.text(95, steady, f" steady ≈ {steady:.2f}s",
                     fontsize=8, color="#666", va="bottom", ha="right")

        # In-flight current cell live overlay
        for c in in_flight:
            h = c["history"]
            if "duration_s" in h.columns and len(h) >= 1:
                axB.plot(h["round"], h["duration_s"], color="#d62728", lw=1.2,
                         alpha=0.9, label=f"LIVE r{int(h['round'].iloc[-1])}",
                         zorder=4)

    axB.set_xlabel("round (1..100)")
    axB.set_ylabel("duration per round (s)")
    axB.set_title("B. Within-cell timing — drift check",
                  fontsize=11, fontweight="bold", loc="left")
    axB.grid(alpha=0.25)
    axB.legend(loc="upper right", fontsize=8, framealpha=0.85)

    # ============ Panel C: AUC trajectory — partition stratification ============
    # Median line + 5-95% band per partition; right-edge labels show the
    # paper-meaningful mean ± std AUC (across all seeds) for each partition.
    part_stats = _aggregate_by_partition(cells)
    final_stats = _partition_final_auc(cells)
    legend_order = [
        ("iid",   "natural-BS"),
        ("0.05",  r"$\alpha$=0.05"),
        ("0.10",  r"$\alpha$=0.10"),
        ("0.50",  r"$\alpha$=0.50"),
        ("1.00",  r"$\alpha$=1.00"),
        ("5.00",  r"$\alpha$=5.00"),
    ]
    line_endpoints = []  # (pkey, color, last_r, last_m) for leader lines

    for pkey, _label in legend_order:
        if pkey not in part_stats:
            continue
        df = part_stats[pkey]
        color = PARTITION_COLORS.get(pkey, "gray")
        axC.fill_between(df["round"], df["p5"], df["p95"],
                         color=color, alpha=0.13)
        axC.plot(df["round"], df["median"], color=color, lw=2.0)
        if len(df):
            line_endpoints.append((pkey, color,
                                   float(df["round"].iloc[-1]),
                                   float(df["median"].iloc[-1])))

    # Right-edge labels with mean ± std (sorted by mean desc, deconflict y).
    # x position fixed at 105; labels use leader lines back to the line end.
    label_x = 105
    label_pad = 0.013  # min vertical spacing between labels
    items = []
    for pkey, label in legend_order:
        if pkey not in final_stats:
            continue
        fs = final_stats[pkey]
        endp = next((e for e in line_endpoints if e[0] == pkey), None)
        items.append({
            "pkey": pkey, "label": label,
            "color": PARTITION_COLORS.get(pkey, "gray"),
            "mean": fs["mean"], "std": fs["std"], "n": fs["n"],
            "last_r": endp[2] if endp else 100.0,
            "last_m": endp[3] if endp else fs["mean"],
        })
    # Sort by AUC desc; assign label y using simple descent with min spacing
    items.sort(key=lambda d: -d["mean"])
    placed_y = []
    for d in items:
        y = d["mean"]
        # If too close to a previously placed label, push down
        for py in placed_y:
            if abs(y - py) < label_pad:
                y = py - label_pad
        placed_y.append(y)
        d["label_y"] = y

    for d in items:
        # leader line from trajectory end to label position
        axC.plot([d["last_r"], label_x - 0.5],
                 [d["last_m"], d["label_y"]],
                 color=d["color"], lw=0.6, alpha=0.55, zorder=2)
        # label box with mean ± std and n
        text = (f"{d['label']}  "
                f"{d['mean']:.3f}$\\pm${d['std']:.3f}  (n={d['n']})")
        axC.text(label_x, d["label_y"], text,
                 fontsize=8.5, color=d["color"], fontweight="bold",
                 va="center", ha="left",
                 bbox=dict(facecolor="white", edgecolor=d["color"],
                           boxstyle="round,pad=0.18", alpha=0.92, lw=0.7))

    # Net IID-vs-worst gap as a single right-side summary
    if items and items[0]["pkey"] == "iid" and len(items) >= 2:
        gap = items[0]["mean"] - items[-1]["mean"]
        axC.text(label_x, min(placed_y) - label_pad * 2.5,
                 f"natural-BS gap\n  $\\Delta$={gap:.3f}",
                 fontsize=9, fontweight="bold", va="top", ha="left",
                 color="#222",
                 bbox=dict(facecolor="lemonchiffon", edgecolor="#888",
                           boxstyle="round,pad=0.25", alpha=0.95, lw=0.8))

    axC.set_xlabel("round")
    axC.set_ylabel("val_auc (median across seeds)")
    axC.set_title("C. AUC by partition — IID-vs-Dirichlet stratification (paper main finding)",
                  fontsize=11, fontweight="bold", loc="left")
    if part_stats:
        valid_lo = [v for stats in part_stats.values() for v in stats["p5"].values
                    if not np.isnan(v)]
        valid_hi = [v for stats in part_stats.values() for v in stats["p95"].values
                    if not np.isnan(v)]
        if valid_lo and valid_hi:
            axC.set_ylim(max(0.50, min(valid_lo) - 0.02),
                         min(0.98, max(valid_hi) + 0.02))
    axC.set_xlim(-2, 138)  # extend right for labels
    axC.grid(alpha=0.25)

    # ============ Panel D: progress dashboard — clean hierarchy ============
    axD.axis("off")
    counts = {a: 0 for a in ARCH_COLORS}
    for c in completed:
        if c["arch"] in counts:
            counts[c["arch"]] += 1
    total_done = sum(counts.values()) + (1 if in_flight else 0)
    pct = 100.0 * total_done / TOTAL_CELLS

    # Big headline number (top center)
    axD.text(0.5, 0.92, f"{total_done} / {TOTAL_CELLS}",
             ha="center", va="top", fontsize=30, fontweight="bold",
             color="#222", transform=axD.transAxes)
    axD.text(0.5, 0.78, f"{pct:.1f}% of Phase 5 sweep",
             ha="center", va="top", fontsize=12, color="#555",
             transform=axD.transAxes)

    # Progress bar with arch breakdown (centered)
    bar_left, bar_y, bar_w, bar_h = 0.10, 0.55, 0.80, 0.08
    cum = 0
    for arch, color in ARCH_COLORS.items():
        frac = counts[arch] / TOTAL_CELLS
        if frac > 0:
            axD.add_patch(plt.Rectangle(
                (bar_left + cum, bar_y), frac * bar_w, bar_h,
                color=color, transform=axD.transAxes, ec="white", lw=0.7,
            ))
            cum += frac * bar_w
    axD.add_patch(plt.Rectangle(
        (bar_left + cum, bar_y), bar_w - cum, bar_h,
        color="#e8e8e8", transform=axD.transAxes, ec="white", lw=0.7,
    ))
    # Bar tick labels at quartile marks
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = bar_left + frac * bar_w
        axD.text(x, bar_y - 0.02, f"{int(frac * TOTAL_CELLS)}",
                 ha="center", va="top", fontsize=7, color="#999",
                 transform=axD.transAxes)

    # Arch breakdown legend (under the bar)
    arch_x = bar_left
    for arch, color in ARCH_COLORS.items():
        label_short = {"lstm": "LSTM", "mamba": "Mamba",
                       "spiking_expand2": "Spiking"}[arch]
        axD.add_patch(plt.Rectangle(
            (arch_x, 0.43), 0.020, 0.030,
            color=color, transform=axD.transAxes,
        ))
        axD.text(arch_x + 0.026, 0.445,
                 f"{label_short}: {counts[arch]}",
                 fontsize=9, transform=axD.transAxes, va="center")
        arch_x += 0.27

    # Throughput + ETA (left column)
    if len(completed) >= 2:
        recent = completed[-min(10, len(completed)):]
        span_s = recent[-1]["mtime"] - recent[0]["mtime"]
        span_cells = len(recent) - 1
        if span_s > 0 and span_cells > 0:
            cph = 3600 * span_cells / span_s
            eta_hr = (TOTAL_CELLS - total_done) / cph if cph > 0 else float("inf")
            axD.text(0.10, 0.32, "throughput",
                     fontsize=9, color="#555", transform=axD.transAxes)
            axD.text(0.10, 0.26, f"{cph:.1f} cells/hr",
                     fontsize=14, fontweight="bold", color="#222",
                     transform=axD.transAxes)
            axD.text(0.10, 0.18, "ETA remain",
                     fontsize=9, color="#555", transform=axD.transAxes)
            axD.text(0.10, 0.12, f"{eta_hr:.1f} hr  ({eta_hr/24:.1f} d)",
                     fontsize=14, fontweight="bold", color="#222",
                     transform=axD.transAxes)

    # Live cell info (right column)
    if in_flight:
        cur = in_flight[0]
        h = cur["history"]
        last_round = int(h["round"].iloc[-1]) if len(h) else 0
        last_auc = float(h["val_auc"].iloc[-1]) if "val_auc" in h.columns and len(h) else 0.0
        last_dt = float(h["duration_s"].iloc[-1]) if "duration_s" in h.columns and len(h) else 0.0
        axD.text(0.55, 0.32, "currently running",
                 fontsize=9, color="#555", transform=axD.transAxes)
        # Trim long cell name (drop "v7_" prefix)
        cell_short = cur["name"].replace("v7_", "")
        if len(cell_short) > 36:
            cell_short = cell_short[:33] + "..."
        axD.text(0.55, 0.26, cell_short,
                 fontsize=10, fontweight="bold", color="#222",
                 family="monospace", transform=axD.transAxes)
        axD.text(0.55, 0.18, f"round {last_round}/100",
                 fontsize=9, color="#555", transform=axD.transAxes)
        axD.text(0.55, 0.12,
                 f"AUC={last_auc:.4f}  dt={last_dt:.1f}s",
                 fontsize=11, color="#222",
                 family="monospace", transform=axD.transAxes)
    else:
        axD.text(0.55, 0.22, "between cells…",
                 fontsize=11, color="#999", style="italic",
                 transform=axD.transAxes)

    fig.suptitle(
        f"Phase 5 v5 — fl-oran-tmc · generated {time.strftime('%Y-%m-%d %H:%M:%S')}",
        fontsize=12, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    cells = load_cells()
    if not cells:
        print(f"no cells found under {ART}", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    render(cells, args.output)
    print(f"saved: {args.output}  (n_cells={len(cells)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
