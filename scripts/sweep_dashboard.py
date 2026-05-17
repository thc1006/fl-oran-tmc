#!/usr/bin/env python3
"""Live Path D SAM-family sweep dashboard (paper-decision-focused).

Redesigned 2026-05-17 to answer the question that actually matters during
the Path D execution: **does FedSCAM / FedGMT / FedMoSWA beat the
Phase 5 baseline (FedAvg / FedAdam) at matched (arch, partition, seed)?**

Pairing rule (per paper §2.6 mechanism argument):
- ``fedscam, fedmoswa`` → compared against ``fedavg`` (both are FedAvg-class
  with an added local sharpness / variance-reduction step)
- ``fedgmt`` → compared against ``fedadam`` (FedGMT is adaptive like
  FedAdam)

Layout:
- Panel A (large, top-left): forest plot of Δ AUC with paired-bootstrap
  CI95. y-axis: (arch, algo, partition). Green = CI95 > 0 (Path D wins),
  red = CI95 < 0 (Path D loses), gray = straddles 0 (no signal).
- Panel B (top-right): sweep progress matrix, 3 archs × 3 algos, each
  bucket shows X/60 with a mini bar.
- Panel C (bottom-left): top-5 most-recently-active cells, val_auc curves
  colour-coded by arch.
- Panel D (sidebar bottom-right): hardware status + NaN cell count + ETA.

Plus an HTML table grouped by (arch, algo, partition) with Δ vs baseline.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SSH_OPTS = [
    "-p", "51419", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
    "-i", str(Path.home() / ".ssh/id_ed25519"),
]
SSH_TARGET = "leo07010@203.145.216.194"
MIRROR_DIR = Path("/tmp/v100_sam_mirror")
PHASE5_DIR = Path("artifacts/v7_stage2_full")

# Per-arch colour (used in curves + forest plot row groups).
ARCH_COLOR = {
    "lstm":             "#1f77b4",   # blue
    "mamba":            "#2ca02c",   # green
    "spiking_expand2":  "#9467bd",   # purple
}
ALGO_MARKER = {"fedscam": "o", "fedgmt": "^", "fedmoswa": "s"}
KNOWN_ARCHS = ("lstm", "mamba", "spiking_expand2")
KNOWN_ALGOS = ("fedscam", "fedgmt", "fedmoswa")
PARTITION_ORDER = (
    "iid",
    "dirichlet_a0p05", "dirichlet_a0p10", "dirichlet_a0p50",
    "dirichlet_a1p00", "dirichlet_a5p00",
)
PARTITION_LABEL = {
    "iid": "IID",
    "dirichlet_a0p05": "α=0.05",
    "dirichlet_a0p10": "α=0.10",
    "dirichlet_a0p50": "α=0.50",
    "dirichlet_a1p00": "α=1.00",
    "dirichlet_a5p00": "α=5.00",
}

# Phase 5 baseline pairing rule (paper §2.6 mechanism).
ALGO_TO_BASELINE = {
    "fedscam": "fedavg",
    "fedmoswa": "fedavg",
    "fedgmt": "fedadam",
}

PATH_D_TARGET_CELLS = 540   # 3 archs × 3 algos × 6 partitions × 10 seeds


def _ssh_cmd(cmd: str, default: str = "") -> str:
    try:
        out = subprocess.run(
            ["ssh"] + SSH_OPTS + [SSH_TARGET, cmd],
            capture_output=True, text=True, timeout=15,
        )
        return out.stdout
    except Exception:
        return default


def rsync_v100_mirror() -> None:
    MIRROR_DIR.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["rsync", "-az", "-e", "ssh " + " ".join(SSH_OPTS),
             "--include=v7_*/", "--include=v7_*/summary.json",
             "--include=v7_*/history.csv", "--exclude=*",
             f"{SSH_TARGET}:fl-oran-tmc/artifacts/v7_sam_family/",
             str(MIRROR_DIR) + "/"],
            check=False, timeout=60, capture_output=True,
        )
    except Exception:
        pass


def parse_cell_name(name: str) -> dict:
    """Parse v7_<arch>_<algo>_<partition>_n<N>_s<seed> into fields."""
    m = re.search(r"_s(\d+)$", name)
    seed = int(m.group(1)) if m else None
    m = re.search(r"_n(\d+)_s\d+$", name)
    n_clients = int(m.group(1)) if m else None
    partition = "unknown"
    if "_iid_" in name:
        partition = "iid"
    else:
        m = re.search(r"_dirichlet_a(\d+p\d+)_", name)
        if m:
            partition = f"dirichlet_a{m.group(1)}"
    algo = "unknown"
    for known in KNOWN_ALGOS + ("fedavg", "fedadam"):
        if f"_{known}_" in name:
            algo = known
            break
    arch = "unknown"
    for known in sorted(KNOWN_ARCHS, key=len, reverse=True):
        if name.startswith(f"v7_{known}_"):
            arch = known
            break
    return {"arch": arch, "algo": algo, "partition": partition,
            "seed": seed, "n_clients": n_clients}


def collect_cells(root: Path, label: str) -> list[dict]:
    cells = []
    if not root.exists():
        return cells
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not d.name.startswith("v7_"):
            continue
        meta = parse_cell_name(d.name)
        hist_path = d / "history.csv"
        summary_path = d / "summary.json"
        hist = None
        if hist_path.exists() and hist_path.stat().st_size > 0:
            try:
                hist = pd.read_csv(hist_path)
            except Exception:
                hist = None
        test_auc = None
        status = "in_progress"
        if summary_path.exists() and summary_path.stat().st_size > 0:
            try:
                summary = json.load(open(summary_path))
                test_auc = summary.get("test_auc")
                status = "done" if test_auc is not None else "failed"
            except Exception:
                status = "failed"
        cells.append({
            "name": d.name, "source": label,
            "arch": meta["arch"],
            "algo": meta["algo"], "partition": meta["partition"],
            "seed": meta["seed"], "history": hist,
            "test_auc": test_auc, "status": status,
            "mtime": hist_path.stat().st_mtime if hist_path.exists() else 0,
        })
    return cells


def load_phase5_baselines() -> dict[tuple, float]:
    """Read artifacts/v7_stage2_full/ → dict keyed by (arch, algo, partition, seed) → test_auc.

    Returns the full 360-cell baseline matrix (3 archs × 2 algos × 6 part × 10 seeds).
    """
    out: dict[tuple, float] = {}
    if not PHASE5_DIR.exists():
        return out
    for d in PHASE5_DIR.iterdir():
        if not d.is_dir() or not d.name.startswith("v7_"):
            continue
        meta = parse_cell_name(d.name)
        if meta["algo"] not in ("fedavg", "fedadam"):
            continue
        summary_path = d / "summary.json"
        if not summary_path.exists() or summary_path.stat().st_size == 0:
            continue
        try:
            test_auc = json.load(open(summary_path)).get("test_auc")
        except Exception:
            continue
        if test_auc is None:
            continue
        key = (meta["arch"], meta["algo"], meta["partition"], meta["seed"])
        out[key] = float(test_auc)
    return out


def paired_bootstrap_ci95(deltas: list[float], n_boot: int = 1000,
                          seed: int = 0) -> tuple[float, float, float]:
    """Returns ``(mean, ci_low, ci_high)``. Empty input → all zeros."""
    if not deltas:
        return (0.0, 0.0, 0.0)
    arr = np.array(deltas, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return (float(arr.mean()),
            float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)))


def pair_cells_against_baseline(
    cells: list[dict], baselines: dict,
) -> list[dict]:
    """For each sweep cell, attach baseline AUC + Δ if a match exists.

    Only V100 cells (sample_ratio=1.0, matching Phase 5 baseline conditions)
    are paired. 4060 smoke cells use sample_ratio=0.1 and are NOT comparable
    to V100 baselines — pairing them would silently inject ~−0.2 AUC noise
    into the dashboard's CI95 estimates. Smoke + KL-ablation cells are
    returned with delta=None and excluded from the forest plot.
    """
    paired = []
    for c in cells:
        if c["algo"] not in ALGO_TO_BASELINE or c["test_auc"] is None:
            paired.append({**c, "delta": None, "baseline_algo": None,
                           "baseline_auc": None})
            continue
        # Sample-ratio guard: only V100 (full-data) cells get paired.
        if c.get("source") != "V100":
            paired.append({**c, "delta": None, "baseline_algo": None,
                           "baseline_auc": None})
            continue
        base_algo = ALGO_TO_BASELINE[c["algo"]]
        key = (c["arch"], base_algo, c["partition"], c["seed"])
        b_auc = baselines.get(key)
        if b_auc is None:
            paired.append({**c, "delta": None, "baseline_algo": base_algo,
                           "baseline_auc": None})
        else:
            paired.append({
                **c, "delta": c["test_auc"] - b_auc,
                "baseline_algo": base_algo,
                "baseline_auc": b_auc,
            })
    return paired


def aggregate_by_group(paired: list[dict]) -> list[dict]:
    """Group paired cells by (arch, algo, partition) → mean Δ + CI95 + n.

    Returns a list of group dicts sorted by arch, algo, partition for
    stable plotting. Only groups with ≥1 paired cell are returned.
    Also computes the baseline mean AUC for reference annotation.
    """
    bucket_delta: dict[tuple, list] = {}
    bucket_base: dict[tuple, list] = {}
    bucket_path_d: dict[tuple, list] = {}
    for p in paired:
        if p["delta"] is None:
            continue
        key = (p["arch"], p["algo"], p["partition"])
        bucket_delta.setdefault(key, []).append(p["delta"])
        bucket_base.setdefault(key, []).append(p["baseline_auc"])
        bucket_path_d.setdefault(key, []).append(p["test_auc"])
    groups = []
    for arch in KNOWN_ARCHS:
        for algo in KNOWN_ALGOS:
            for partition in PARTITION_ORDER:
                key = (arch, algo, partition)
                deltas = bucket_delta.get(key, [])
                if not deltas:
                    continue
                mean, lo, hi = paired_bootstrap_ci95(deltas)
                bases = bucket_base.get(key, [])
                path_d_aucs = bucket_path_d.get(key, [])
                groups.append({
                    "arch": arch, "algo": algo, "partition": partition,
                    "n": len(deltas), "mean_delta": mean,
                    "ci_low": lo, "ci_high": hi,
                    "deltas": deltas,
                    "baseline_auc_mean": (
                        float(np.mean(bases)) if bases else None
                    ),
                    "path_d_auc_mean": (
                        float(np.mean(path_d_aucs)) if path_d_aucs else None
                    ),
                })
    return groups


def probe_v100_status() -> dict:
    chains = {}
    for c in range(4):
        log = _ssh_cmd(f"cat ~/fl-oran-tmc/logs/v100_path_d_chain{c}.log 2>/dev/null")
        chains[c] = {
            "done": len(re.findall(r"v7_\w+ done:", log)),
            "fail": len(re.findall(r"NonFiniteLossError|cell.*FAIL", log)),
        }
    gpu_csv = _ssh_cmd(
        "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits"
    ).strip()
    gpus = []
    for line in gpu_csv.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            gpus.append({"idx": int(parts[0]),
                         "mem_mib": int(parts[1]),
                         "util_pct": int(parts[2])})
    master_alive = bool(
        _ssh_cmd("pgrep -f v100_path_d_launcher || "
                 "pgrep -f run_v7_phase_sweep").strip()
    )
    return {"chains": chains, "gpus": gpus, "master_alive": master_alive}


def probe_local_gpu() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        parts = [p.strip() for p in out.split(",")]
        return {"mem_mib": int(parts[0]), "util_pct": int(parts[1])}
    except Exception:
        return {"mem_mib": 0, "util_pct": 0}


def render_png(
    cells: list[dict], groups: list[dict],
    v100: dict, local_gpu: dict, out_path: Path,
) -> None:
    fig = plt.figure(figsize=(20, 13))
    # 3-row layout:
    #   row 0: forest plot (full width, 60% height) — the main paper view
    #   row 1: progress matrix (left 60%) + sidebar (right 40%)
    #   row 2: live val_auc curves (full width)
    gs = fig.add_gridspec(
        3, 5, width_ratios=[1, 1, 1, 1, 1],
        height_ratios=[2.0, 0.9, 0.9],
        hspace=0.45, wspace=0.45,
    )

    # === Panel A: forest plot (full width row 0) ===
    axA = fig.add_subplot(gs[0, :])
    _render_forest(axA, groups)

    # === Panel B: progress matrix (left 3 cols of row 1) ===
    axB = fig.add_subplot(gs[1, 0:3])
    _render_progress_matrix(axB, cells)

    # === Panel C: hardware sidebar (right 2 cols of row 1) ===
    axC = fig.add_subplot(gs[1, 3:])
    _render_sidebar(axC, cells, v100, local_gpu)

    # === Panel D: live curves (full width row 2) ===
    axD = fig.add_subplot(gs[2, :])
    _render_live_curves(axD, cells)

    fig.suptitle(
        f"Path D SAM-family sweep — {datetime.now():%Y-%m-%d %H:%M:%S}",
        fontsize=15, y=0.995, fontweight="bold",
    )
    plt.savefig(out_path, dpi=110, bbox_inches="tight",
                facecolor="white")
    plt.close()


def _render_forest(ax, groups: list[dict]) -> None:
    """Forest plot of mean Δ AUC + CI95 per (arch, algo, partition).

    Layout: rows grouped by (arch, algo) with thin separator lines between
    blocks for easy visual scan. Each row labelled with partition + n +
    baseline_auc (the Phase 5 reference number) + Δ + significance marker.
    """
    if not groups:
        ax.set_title("Δ AUC vs Phase 5 baseline  (no paired V100 cells yet)",
                     fontsize=11)
        ax.text(0.5, 0.5,
                "Path D sweep has not produced any V100 (full-sample) "
                "cells yet.\nForest plot will populate as cells complete.",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="#888")
        ax.axis("off")
        return

    n_win = sum(1 for g in groups if g["ci_low"] > 0)
    n_lose = sum(1 for g in groups if g["ci_high"] < 0)
    n_null = len(groups) - n_win - n_lose
    ax.set_title(
        f"Δ AUC vs Phase 5 baseline  (paired-bootstrap CI95, n_boot=1000) "
        f"  ·   {n_win} win | {n_lose} lose | {n_null} null",
        fontsize=11, fontweight="bold",
    )

    ax.axvline(0.0, color="black", lw=1.0, ls="-", alpha=0.6, zorder=2)

    y_positions = []
    y_labels = []
    prev_block = None
    # Walk groups top-to-bottom in (arch, algo, partition) order
    for i, g in enumerate(groups):
        y = len(groups) - 1 - i
        # Separator line between (arch, algo) blocks
        block = (g["arch"], g["algo"])
        if prev_block is not None and block != prev_block:
            ax.axhline(y + 0.5, color="#dddddd", lw=0.5, zorder=1)
        prev_block = block
        y_positions.append(y)
        # Compact row label: arch · algo · partition · n · baseline ref
        b_auc = g.get("baseline_auc_mean")
        if b_auc is not None:
            ref_str = f"  [base={b_auc:.3f}]"
        else:
            ref_str = ""
        y_labels.append(
            f"{g['arch'][:6]:<6} · {g['algo']:<8} · "
            f"{PARTITION_LABEL[g['partition']]:<8}  n={g['n']}{ref_str}"
        )
        # CI color: green (CI>0), red (CI<0), gray (straddles)
        if g["ci_low"] > 0:
            color = "#1d6b2a"
            band = "#d4f4dd"
        elif g["ci_high"] < 0:
            color = "#8a1f1f"
            band = "#fcdada"
        else:
            color = "#666"
            band = None
        # Background band shading for significant rows
        if band is not None:
            ax.axhspan(y - 0.4, y + 0.4, color=band, alpha=0.5, zorder=1)
        # CI line + point
        ax.plot([g["ci_low"], g["ci_high"]], [y, y], color=color, lw=2.5,
                solid_capstyle="round", alpha=0.95, zorder=3)
        ax.plot([g["mean_delta"]], [y],
                marker=ALGO_MARKER[g["algo"]],
                color=color, markersize=9,
                markeredgecolor="black", markeredgewidth=0.9, zorder=4)
        # Arch hint as colored edge bar on left
        ax.plot([0, 0], [y - 0.4, y + 0.4],
                color=ARCH_COLOR[g["arch"]], lw=5, alpha=0.0)  # invisible

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=8, family="monospace")
    ax.set_xlabel("Δ AUC  (Path D algo − Phase 5 baseline at matched seed)",
                  fontsize=10)
    ax.grid(True, axis="x", alpha=0.25, zorder=0)
    # Symmetric x-axis centered at 0 for fair comparison
    max_abs = max(
        max(abs(g["ci_low"]), abs(g["ci_high"]), abs(g["mean_delta"]))
        for g in groups
    )
    max_abs = max(max_abs * 1.15, 0.01)
    ax.set_xlim(-max_abs, max_abs)
    # Subtle "wins on this side / loses on this side" annotations
    ax.text(0.99, 1.01, "→ Path D wins",
            color="#1d6b2a", fontsize=9, ha="right", va="bottom",
            transform=ax.transAxes, fontweight="bold")
    ax.text(0.01, 1.01, "Path D loses ←",
            color="#8a1f1f", fontsize=9, ha="left", va="bottom",
            transform=ax.transAxes, fontweight="bold")


def _render_progress_matrix(ax, cells: list[dict]) -> None:
    """3 archs × 3 algos progress matrix. Each bucket: done/60 with bar.

    Uses set_xlim(-0.5, 3.0) so the left arch labels have a dedicated
    column outside the data cells; algo headers sit above with extra
    vertical padding to avoid overlap.
    """
    ax.set_title("Path D cell progress  (V100 cells / 60 per bucket)",
                 fontsize=10, fontweight="bold")
    ax.set_xlim(-0.7, 3.0)
    ax.set_ylim(-0.35, 3.5)
    ax.axis("off")

    # Header cells (algo) — pushed up to 3.2 to avoid touching data cells
    for j, algo in enumerate(KNOWN_ALGOS):
        ax.text(j + 0.5, 3.25, algo, fontsize=10, fontweight="bold",
                ha="center", va="center")
    # Side cells (arch) — in their own column at x=-0.35
    for i, arch in enumerate(KNOWN_ARCHS):
        y = 2 - i
        ax.text(-0.35, y + 0.5, arch, fontsize=9, fontweight="bold",
                color=ARCH_COLOR[arch], ha="center", va="center",
                rotation=0)
        for j, algo in enumerate(KNOWN_ALGOS):
            n = sum(1 for c in cells
                    if c["source"] == "V100"
                    and c["arch"] == arch and c["algo"] == algo
                    and c["status"] == "done")
            pct = 100 * n / 60.0
            color = "#2ca02c" if n >= 60 else "#1f77b4" if n > 0 else "#ddd"
            # background rect
            ax.add_patch(plt.Rectangle((j, y), 1, 1, fill=True,
                                        facecolor="#fafafa",
                                        edgecolor="#aaaaaa", lw=0.7))
            # progress bar (bottom strip)
            bar_w = min(n / 60.0, 1.0)
            ax.add_patch(plt.Rectangle((j + 0.05, y + 0.08), 0.9 * bar_w, 0.18,
                                        fill=True, facecolor=color, alpha=0.85,
                                        edgecolor="none"))
            # count text
            ax.text(j + 0.5, y + 0.62, f"{n}/60",
                    fontsize=12, ha="center", va="center", fontweight="bold")
            ax.text(j + 0.5, y + 0.40, f"{pct:.0f}%",
                    fontsize=8, ha="center", va="center", color="#666")

    total_done = sum(1 for c in cells
                     if c["source"] == "V100"
                     and c["status"] == "done"
                     and c["algo"] in KNOWN_ALGOS)
    pct_total = 100 * total_done / PATH_D_TARGET_CELLS
    bar_color = "#2ca02c" if total_done >= PATH_D_TARGET_CELLS else "#1f77b4"
    ax.text(1.15, -0.20, f"Total Path D V100: {total_done} / "
                          f"{PATH_D_TARGET_CELLS}  ({pct_total:.1f}%)",
            fontsize=10, ha="center", fontweight="bold", color=bar_color)


def _render_sidebar(ax, cells: list[dict], v100: dict, local_gpu: dict) -> None:
    """Hardware status + NaN count + ETA."""
    ax.axis("off")
    ax.set_title("status", fontsize=10)
    failed = sum(1 for c in cells if c["status"] == "failed")
    in_prog = sum(1 for c in cells if c["status"] == "in_progress")
    done = sum(1 for c in cells if c["status"] == "done")

    master_marker = "ALIVE" if v100["master_alive"] else "idle"
    master_color = "#2ca02c" if v100["master_alive"] else "#888888"

    lines = [
        ("V100 master:", master_marker, master_color, True),
        ("", "", "black", False),
        ("V100 GPU util:", "", "black", True),
    ]
    for g in v100["gpus"]:
        util_color = "#2ca02c" if g["util_pct"] > 50 else "#888"
        lines.append((f"  gpu{g['idx']}:",
                      f"{g['util_pct']:>3d}%  {g['mem_mib']} MiB",
                      util_color, False))
    lines.extend([
        ("", "", "black", False),
        ("4060 GPU:", f"{local_gpu['util_pct']}% {local_gpu['mem_mib']} MiB",
         "black", True),
        ("", "", "black", False),
        ("Cells:", f"{done} done  {in_prog} live  {failed} fail",
         "#d62728" if failed else "black", True),
        ("", "", "black", False),
        ("refreshed:", datetime.now().strftime("%H:%M:%S"), "#888", False),
    ])
    y = 0.95
    for label, value, color, bold in lines:
        weight = "bold" if bold else "normal"
        ax.text(0.02, y, label, fontsize=8, family="monospace",
                color=color, weight=weight,
                transform=ax.transAxes, va="top")
        ax.text(0.55, y, value, fontsize=8, family="monospace",
                color=color, weight=weight,
                transform=ax.transAxes, va="top")
        y -= 0.065


def _render_live_curves(ax, cells: list[dict]) -> None:
    """Top-5 most-recently-active cells, val_auc curves."""
    ax.set_title("Recent val_auc curves (top 5 most-recently-updated cells)",
                 fontsize=10)
    ax.set_xlabel("round")
    ax.set_ylabel("val_auc")
    ax.grid(True, alpha=0.3)

    # Filter cells with usable history, sort by mtime desc
    live = [c for c in cells if c["history"] is not None
            and "val_auc" in c["history"].columns]
    live.sort(key=lambda c: c["mtime"], reverse=True)
    top = live[:5]
    if not top:
        ax.text(0.5, 0.5, "(no live training history available)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="#888")
        return
    for c in top:
        h = c["history"]
        color = ARCH_COLOR.get(c["arch"], "#888")
        marker = ALGO_MARKER.get(c["algo"], None)
        label = (f"{c['arch'][:4]}·{c['algo'][:4]}·"
                 f"{PARTITION_LABEL.get(c['partition'], c['partition'])}·s{c['seed']}")
        ax.plot(h["round"], h["val_auc"], color=color, lw=1.3, alpha=0.85,
                marker=marker, markersize=4, markevery=max(len(h) // 8, 1),
                label=label)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
    ax.set_ylim(0.5, 1.0)


def _format_group_row(g: dict) -> str:
    """One HTML row for the grouped results table."""
    if g["ci_low"] > 0:
        css = "background:#d4f4dd"
        sig = "★"
    elif g["ci_high"] < 0:
        css = "background:#fcdada"
        sig = "↓"
    else:
        css = ""
        sig = ""
    return (
        f"<tr style='{css}'>"
        f"<td>{g['arch']}</td>"
        f"<td>{g['algo']}</td>"
        f"<td>{PARTITION_LABEL[g['partition']]}</td>"
        f"<td>{g['n']}</td>"
        f"<td>{g['mean_delta']:+.4f}</td>"
        f"<td>[{g['ci_low']:+.4f}, {g['ci_high']:+.4f}]</td>"
        f"<td>{sig}</td>"
        f"</tr>"
    )


def render_html(
    cells: list[dict], groups: list[dict],
    v100: dict, local_gpu: dict, png_name: str,
) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    done = sum(1 for c in cells if c["status"] == "done")
    in_prog = sum(1 for c in cells if c["status"] == "in_progress")
    failed = sum(1 for c in cells if c["status"] == "failed")
    path_d_cells = sum(1 for c in cells
                       if c["algo"] in KNOWN_ALGOS and c["status"] == "done")

    # Per (arch, algo) progress bars
    progress_rows = ""
    for arch in KNOWN_ARCHS:
        for algo in KNOWN_ALGOS:
            n = sum(1 for c in cells if c["arch"] == arch
                    and c["algo"] == algo and c["status"] == "done")
            pct = 100 * n / 60.0
            progress_rows += (
                f"<tr><td>{arch}</td><td>{algo}</td>"
                f"<td>{n}/60</td><td>{pct:.0f}%</td></tr>"
            )

    # Failure list (last 10)
    fails = sorted(
        (c for c in cells if c["status"] == "failed"),
        key=lambda c: c["mtime"], reverse=True,
    )[:10]
    fail_rows = "".join(
        f"<tr><td>{c['name']}</td></tr>" for c in fails
    ) or '<tr><td><i>none</i></td></tr>'

    # Grouped results table sorted to highlight winners first
    sorted_groups = sorted(
        groups, key=lambda g: (-g["mean_delta"], g["arch"], g["algo"])
    )
    grouped_rows = "".join(_format_group_row(g) for g in sorted_groups) \
        or '<tr><td colspan="7"><i>(no Phase-5-paired cells yet)</i></td></tr>'

    return f"""<!doctype html>
<html><head>
<meta http-equiv="refresh" content="30">
<title>Path D dashboard</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 18px; max-width: 1500px; }}
h1 {{ margin-bottom: 4px; }}
h3 {{ margin-top: 1.5em; }}
.ts {{ color: #666; font-size: 0.9em; }}
.kpi {{ display: inline-block; padding: 4px 14px; margin-right: 8px;
       background: #f4f4f4; border-radius: 4px; font-weight: bold; }}
.kpi.win {{ background: #d4f4dd; color: #1d6b2a; }}
.kpi.fail {{ background: #fcdada; color: #8a1f1f; }}
table {{ border-collapse: collapse; margin: 8px 0; }}
th, td {{ border: 1px solid #ccc; padding: 4px 10px; text-align: left;
         font-size: 0.85em; }}
th {{ background: #f4f4f4; }}
img {{ max-width: 100%; border: 1px solid #ddd; }}
.flex {{ display: flex; gap: 24px; flex-wrap: wrap; }}
.legend {{ font-size: 0.85em; color: #555; margin-bottom: 6px; }}
</style>
</head><body>
<h1>Path D SAM-family sweep dashboard</h1>
<p class="ts">{ts} · auto-refresh 30s</p>
<p>
<span class="kpi">Path D: {path_d_cells} / {PATH_D_TARGET_CELLS}</span>
<span class="kpi">{done} done</span>
<span class="kpi">{in_prog} live</span>
<span class="kpi{' fail' if failed else ''}">{failed} fail</span>
</p>

<img src="{png_name}?t={int(time.time())}" alt="dashboard">

<h3>Grouped results (paired vs Phase 5 baseline)</h3>
<p class="legend">Pairing rule: fedscam/fedmoswa → fedavg; fedgmt → fedadam.
   ★ = paired-bootstrap CI95 excludes 0 positively (Path D wins).
   ↓ = CI95 excludes 0 negatively (Path D loses).</p>
<table>
<tr><th>arch</th><th>algo</th><th>partition</th><th>n</th>
    <th>mean Δ AUC</th><th>CI95</th><th>sig</th></tr>
{grouped_rows}
</table>

<div class="flex">
<div>
<h3>Per-(arch, algo) progress</h3>
<table>
<tr><th>arch</th><th>algo</th><th>cells</th><th>%</th></tr>
{progress_rows}
</table>
</div>
<div>
<h3>Recent failures</h3>
<table>
<tr><th>cell name</th></tr>
{fail_rows}
</table>
</div>
</div>

</body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/tmp/sam_dashboard"))
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    png = args.out / "status.png"
    html = args.out / "index.html"

    while True:
        try:
            rsync_v100_mirror()
            v100_status = probe_v100_status()
            local_gpu = probe_local_gpu()
            baselines = load_phase5_baselines()

            cells = []
            cells += [{**c, "source": "V100"} for c in collect_cells(MIRROR_DIR, "V100")]
            cells += [{**c, "source": "4060-off"} for c in collect_cells(
                Path("artifacts/v7_fedgmt_kl_off"), "4060-off")]
            cells += [{**c, "source": "4060-on"} for c in collect_cells(
                Path("artifacts/v7_fedgmt_kl_on"), "4060-on")]
            cells += [{**c, "source": "4060-smoke"} for c in collect_cells(
                Path("artifacts/v7_path_d_smoke"), "4060-smoke")]

            paired = pair_cells_against_baseline(cells, baselines)
            groups = aggregate_by_group(paired)

            render_png(cells, groups, v100_status, local_gpu, png)
            html.write_text(render_html(
                cells, groups, v100_status, local_gpu, "status.png",
            ))

            done = sum(1 for c in cells if c["status"] == "done")
            fail = sum(1 for c in cells if c["status"] == "failed")
            print(f"[{datetime.now():%H:%M:%S}] cells={len(cells)} "
                  f"done={done} fail={fail} baselines={len(baselines)} "
                  f"groups={len(groups)}", flush=True)
        except Exception as e:
            import traceback
            print(f"[{datetime.now():%H:%M:%S}] error: {e}", flush=True)
            traceback.print_exc()
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
