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

# Per-arch baseline wall estimate (s) — used by ETA when no empirical
# Path D cell data is available yet. Values refined 2026-05-17 from
# V100 pilot (4 cells at sample_ratio=1.0, num_rounds=100):
#   - LSTM (existing 60 cells):     phase_timings_s["TOTAL"] ≈ 487-520s
#   - Mamba (pilot cells):          698 / 928s → conservative ~813s
#   - Spiking_expand2 (pilot cells): 842 / 1220s → conservative ~1031s
# As more Path D cells complete on V100 the empirical mean overrides
# this fallback per arch (see compute_eta).
ARCH_WALL_FALLBACK_S = {
    "lstm":             520.0,
    "mamba":            813.0,
    "spiking_expand2":  1031.0,
}
N_PARALLEL_CHAINS = 4
# Cells per chain after shard 1/4 — useful for chain-progress display.
# 540 cells / 4 chains = 135 nominal; --skip-existing-summary trims to
# ~119 new cells per chain (sweep skips 60 LSTM + 4 pilot = 64 cells
# already done). Used purely for "X/Y" progress display.
CELLS_PER_CHAIN_NEW = 119

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
    "iid": "IID 同質",
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
        wall_s = None
        if summary_path.exists() and summary_path.stat().st_size > 0:
            try:
                summary = json.load(open(summary_path))
                test_auc = summary.get("test_auc")
                status = "done" if test_auc is not None else "failed"
                # phase_timings_s["TOTAL"] is the per-cell wall on the
                # training machine; used to refine ETA empirically.
                timings = summary.get("phase_timings_s") or {}
                wall_s = timings.get("TOTAL")
            except Exception:
                status = "failed"
        cells.append({
            "name": d.name, "source": label,
            "arch": meta["arch"],
            "algo": meta["algo"], "partition": meta["partition"],
            "seed": meta["seed"], "history": hist,
            "test_auc": test_auc, "status": status,
            "wall_s": wall_s,
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


def compute_eta(cells: list[dict]) -> dict:
    """Estimate Path D remaining wall + completion time.

    For each arch, compute the empirical mean per-cell wall from completed
    V100 cells. Fall back to ``ARCH_WALL_FALLBACK_S`` when no data is yet
    available. Remaining wall = Σ_arch (remaining_cells_in_arch × wall_arch)
    divided by ``N_PARALLEL_CHAINS`` since chains run in parallel.

    Returns a dict with ``remaining_seconds``, ``eta_datetime``, and
    ``per_arch`` breakdown (each entry: ``{empirical_n, wall_s,
    remaining_cells}``).
    """
    per_arch: dict[str, dict] = {}
    for arch in KNOWN_ARCHS:
        # Empirical mean wall from completed V100 Path D cells.
        walls = [c["wall_s"] for c in cells
                 if c.get("source") == "V100"
                 and c.get("arch") == arch
                 and c.get("algo") in KNOWN_ALGOS
                 and c.get("status") == "done"
                 and isinstance(c.get("wall_s"), (int, float))]
        if walls:
            wall_arch = float(np.mean(walls))
            empirical_n = len(walls)
        else:
            wall_arch = ARCH_WALL_FALLBACK_S[arch]
            empirical_n = 0
        done = sum(1 for c in cells
                   if c.get("source") == "V100"
                   and c.get("arch") == arch
                   and c.get("algo") in KNOWN_ALGOS
                   and c.get("status") == "done")
        target = len(KNOWN_ALGOS) * 60     # 3 algos × 60 cells per (arch, algo)
        remaining_cells = max(target - done, 0)
        per_arch[arch] = {
            "empirical_n": empirical_n,
            "wall_s": wall_arch,
            "remaining_cells": remaining_cells,
            "done": done,
            "target": target,
        }
    # Parallel: divide total remaining by N chains.
    total_remaining_serial = sum(
        a["remaining_cells"] * a["wall_s"] for a in per_arch.values()
    )
    remaining_s = total_remaining_serial / max(N_PARALLEL_CHAINS, 1)
    from datetime import timedelta
    eta_dt = datetime.now() + timedelta(seconds=remaining_s)
    return {
        "remaining_seconds": remaining_s,
        "eta_datetime": eta_dt,
        "per_arch": per_arch,
    }


def _format_eta(eta: dict) -> tuple[str, str]:
    """Returns (short_form, long_form) Traditional Chinese."""
    s = eta["remaining_seconds"]
    if s < 60:
        short = "已完成"
        long_ = "已完成"
    elif s < 3600:
        short = f"剩餘 {s/60:.0f} 分鐘"
        long_ = (
            f"預計剩餘 {s/60:.1f} 分鐘 · "
            f"預計完成：{eta['eta_datetime']:%m-%d %H:%M}"
        )
    else:
        short = f"剩餘 {s/3600:.1f} 小時"
        long_ = (
            f"預計剩餘 {s/3600:.1f} 小時 · "
            f"預計完成：{eta['eta_datetime']:%Y-%m-%d %H:%M}"
        )
    return short, long_


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
    v100: dict, local_gpu: dict, eta: dict, out_path: Path,
) -> None:
    # Configure CJK-capable font for Traditional Chinese rendering.
    # On Ubuntu, the only family matplotlib's PS-name scanner picks up
    # from the multi-CJK Noto .ttc is "Noto Sans CJK JP" — but the same
    # font file contains the TC/SC/HK glyph variants, so this renders
    # Traditional Chinese characters correctly. AR PL UMing CN is the
    # Ubuntu-bundled fallback (uming.ttc). Boxes-instead-of-glyphs would
    # indicate no CJK font installed at all.
    matplotlib.rcParams["font.sans-serif"] = [
        "Noto Sans CJK JP", "AR PL UMing CN", "AR PL UKai CN",
        "DejaVu Sans",
    ]
    # Monospace fallback also needs a CJK font for sidebar value column
    # ("完成 X · 進行 X · 失敗 X"). Noto Sans CJK isn't strictly monospace
    # but renders 全形 (CJK) glyphs at fixed cell width, so columns still
    # align reasonably.
    matplotlib.rcParams["font.monospace"] = [
        "Noto Sans Mono CJK JP", "Noto Sans CJK JP",
        "AR PL UKai CN", "DejaVu Sans Mono",
    ]
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["axes.unicode_minus"] = False

    # Larger figure + constrained_layout so CJK characters never collide.
    # constrained_layout auto-adjusts subplot spacing whenever text would
    # overflow into another axes — strictly better than bbox_inches='tight'
    # for dense dashboards.
    fig = plt.figure(figsize=(22, 16), constrained_layout=True)
    fig.set_constrained_layout_pads(
        w_pad=0.12, h_pad=0.18, hspace=0.10, wspace=0.10,
    )
    # Tall forest panel (50% height), shorter middle row (progress + sidebar),
    # and a bottom row for live curves.
    gs = fig.add_gridspec(
        3, 6, width_ratios=[1, 1, 1, 1, 1, 1],
        height_ratios=[2.2, 1.0, 1.0],
    )
    axA = fig.add_subplot(gs[0, :])
    _render_forest(axA, groups)
    # Progress matrix gets 4 of 6 cols (more horizontal room), sidebar 2 cols.
    axB = fig.add_subplot(gs[1, 0:4])
    _render_progress_matrix(axB, cells, eta)
    axC = fig.add_subplot(gs[1, 4:])
    _render_sidebar(axC, cells, v100, local_gpu, eta)
    axD = fig.add_subplot(gs[2, :])
    _render_live_curves(axD, cells)

    fig.suptitle(
        f"Path D SAM 家族大規模實驗  —  {datetime.now():%Y-%m-%d %H:%M:%S}",
        fontsize=15, fontweight="bold",
    )
    plt.savefig(out_path, dpi=110, facecolor="white")
    plt.close()


def _render_forest(ax, groups: list[dict]) -> None:
    """Δ AUC 森林圖 (與 Phase 5 基線配對)。

    每列依 (架構, 演算法) 分組並以淺色分隔線區隔。標籤包含分割
    模式、樣本數、基線 AUC、平均 Δ AUC 與顯著性標記。
    """
    if not groups:
        ax.set_title(
            "Δ AUC vs Phase 5 基線  (尚無配對的 V100 實驗格)",
            fontsize=11,
        )
        ax.text(0.5, 0.5,
                "Path D 尚未產生 V100 (sample_ratio=1.0) 實驗格。\n"
                "等待 V100 sweep 啟動後，森林圖會自動填入。",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="#888")
        ax.axis("off")
        return

    # n<3 groups have unreliable CIs — count separately + grey-out display
    SMALL_N = 3
    final_groups = [g for g in groups if g["n"] >= SMALL_N]
    prelim_groups = [g for g in groups if g["n"] < SMALL_N]
    n_win = sum(1 for g in final_groups if g["ci_low"] > 0)
    n_lose = sum(1 for g in final_groups if g["ci_high"] < 0)
    n_null = len(final_groups) - n_win - n_lose
    n_prelim = len(prelim_groups)
    title_parts = [
        f"{n_win} 組顯著優於基線",
        f"{n_lose} 組顯著劣於基線",
        f"{n_null} 組無顯著差異",
    ]
    if n_prelim > 0:
        title_parts.append(f"{n_prelim} 組初步 (n<{SMALL_N})")
    ax.set_title(
        "Δ AUC 對 Phase 5 基線之差異  "
        "(配對自助法 95% 信賴區間, 抽樣 1000 次)\n"
        + "   ·   ".join(title_parts),
        fontsize=11, fontweight="bold", pad=14,
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
        # 列標籤：架構 · 演算法 · 切分 · 種子數 · 基線 AUC
        # Display-friendly arch names matched to figure legend.
        arch_disp = {"lstm": "LSTM", "mamba": "Mamba",
                     "spiking_expand2": "Spiking"}.get(g["arch"], g["arch"])
        b_auc = g.get("baseline_auc_mean")
        ref_str = f"  基線 {b_auc:.3f}" if b_auc is not None else ""
        y_labels.append(
            f"{arch_disp:<7} · {g['algo']:<8} · "
            f"{PARTITION_LABEL[g['partition']]:<10}  "
            f"n={g['n']}{ref_str}"
        )
        # CI color: green (CI>0), red (CI<0), gray (straddles).
        # For n < SMALL_N: muted variants since CI95 is unreliable.
        is_prelim = g["n"] < SMALL_N
        if g["ci_low"] > 0 and not is_prelim:
            color = "#1d6b2a"
            band = "#d4f4dd"
        elif g["ci_high"] < 0 and not is_prelim:
            color = "#8a1f1f"
            band = "#fcdada"
        elif is_prelim:
            color = "#bbbbbb"   # grey for low-confidence
            band = None
        else:
            color = "#666"
            band = None
        # Background band shading for significant rows
        if band is not None:
            ax.axhspan(y - 0.4, y + 0.4, color=band, alpha=0.5, zorder=1)
        # CI line + point (lighter linewidth for preliminary)
        lw = 1.2 if is_prelim else 2.5
        alpha = 0.45 if is_prelim else 0.95
        ax.plot([g["ci_low"], g["ci_high"]], [y, y], color=color, lw=lw,
                solid_capstyle="round", alpha=alpha, zorder=3)
        ax.plot([g["mean_delta"]], [y],
                marker=ALGO_MARKER[g["algo"]],
                color=color, markersize=9 if not is_prelim else 6,
                markeredgecolor="black", markeredgewidth=0.9 if not is_prelim else 0.5,
                alpha=alpha, zorder=4)
        # Arch hint as colored edge bar on left
        ax.plot([0, 0], [y - 0.4, y + 0.4],
                color=ARCH_COLOR[g["arch"]], lw=5, alpha=0.0)  # invisible

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=8, family="monospace")
    ax.set_xlabel("Δ AUC  (Path D 演算法 − Phase 5 基線，相同隨機種子)",
                  fontsize=10)
    ax.grid(True, axis="x", alpha=0.25, zorder=0)
    max_abs = max(
        max(abs(g["ci_low"]), abs(g["ci_high"]), abs(g["mean_delta"]))
        for g in groups
    )
    max_abs = max(max_abs * 1.15, 0.01)
    ax.set_xlim(-max_abs, max_abs)
    # Direction-of-effect annotations placed BELOW the x-axis (inside the
    # plotting area), not above the title — avoids overlap with the
    # multiline title and keeps the visual focus on the data area.
    ax.text(0.99, -0.07, "→ Path D 顯著優於基線",
            color="#1d6b2a", fontsize=10, ha="right", va="top",
            transform=ax.transAxes, fontweight="bold")
    ax.text(0.01, -0.07, "Path D 顯著劣於基線 ←",
            color="#8a1f1f", fontsize=10, ha="left", va="top",
            transform=ax.transAxes, fontweight="bold")


def _render_progress_matrix(ax, cells: list[dict], eta: dict) -> None:
    """3 架構 × 3 演算法進度矩陣，每格顯示 done/60 與進度條。

    底部顯示總進度與 ETA，預留充足垂直空間避免重疊。
    """
    ax.set_title("Path D 實驗格完成進度  (每格 60 個 V100 實驗格)",
                 fontsize=11, fontweight="bold", pad=12)
    ax.set_xlim(-1.1, 3.0)
    ax.set_ylim(-0.85, 3.55)
    ax.axis("off")

    # 列標題（演算法）— 在資料格上方 0.35 處避免黏在格子邊緣
    for j, algo in enumerate(KNOWN_ALGOS):
        ax.text(j + 0.5, 3.30, algo, fontsize=11, fontweight="bold",
                ha="center", va="center")
    # Display-friendly arch names — truncate spiking_expand2 to "spiking"
    arch_display = {"lstm": "LSTM", "mamba": "Mamba",
                    "spiking_expand2": "Spiking"}
    # 列標題（架構）— 自己的 column 在 x=-0.55，更寬避免被切掉
    for i, arch in enumerate(KNOWN_ARCHS):
        y = 2 - i
        ax.text(-0.55, y + 0.5, arch_display[arch], fontsize=11,
                fontweight="bold", color=ARCH_COLOR[arch],
                ha="center", va="center", rotation=0)
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
    # Total progress + ETA — stacked with 0.30 vspace to avoid collision.
    ax.text(1.0, -0.30,
            f"Path D V100 總進度: {total_done} / "
            f"{PATH_D_TARGET_CELLS}  ({pct_total:.1f}%)",
            fontsize=11, ha="center", fontweight="bold", color=bar_color)
    _short, eta_long = _format_eta(eta)
    eta_color = "#1f77b4" if eta["remaining_seconds"] > 60 else "#2ca02c"
    ax.text(1.0, -0.65, eta_long, fontsize=10, ha="center", color=eta_color)


def _render_sidebar(
    ax, cells: list[dict], v100: dict, local_gpu: dict, eta: dict,
) -> None:
    """硬體狀態 + ETA + 失敗計數。寬欄位、簡潔排版。"""
    ax.axis("off")
    ax.set_title("即時狀態", fontsize=11, fontweight="bold", pad=10)
    failed = sum(1 for c in cells if c["status"] == "failed")
    in_prog = sum(1 for c in cells if c["status"] == "in_progress")
    done = sum(1 for c in cells if c["status"] == "done")

    master_marker = "執行中" if v100["master_alive"] else "閒置"
    master_color = "#2ca02c" if v100["master_alive"] else "#888888"
    short_eta, _ = _format_eta(eta)

    # 摺疊式排版：每組相關欄位放成單一寬列，減少視覺擁擠。
    lines: list[tuple[str, str, str, str]] = [
        # (label, value, color, weight)
        ("V100 主控", master_marker, master_color, "bold"),
        ("預計剩餘", short_eta, "#1f4b80", "bold"),
        ("預計完成", f"{eta['eta_datetime']:%Y-%m-%d %H:%M}",
         "#1f4b80", "normal"),
        ("__SEP__", "", "", ""),
    ]

    # Per-chain progress (P0 improvement — detect stuck chains)
    # ``v100['chains']`` is populated by probe_v100_status which counts
    # "v7_\w+ done:" matches in each chain log file. The pattern matches
    # the per-cell completion line from run_v7_phase_sweep.
    if v100["chains"]:
        lines.append(("Chain 進度", f"目標 ~{CELLS_PER_CHAIN_NEW}/chain",
                      "black", "bold"))
        chain_done_max = max(
            (st["done"] for st in v100["chains"].values()),
            default=0,
        )
        chain_done_min = min(
            (st["done"] for st in v100["chains"].values()),
            default=0,
        )
        for c_idx, st in sorted(v100["chains"].items()):
            done_n = st["done"]
            fail_n = st["fail"]
            # 落後判定：done 比中位數低 50% 即標紅
            if (chain_done_max - done_n) > 5 and done_n < chain_done_max * 0.5:
                row_color = "#d62728"   # red — stuck/slow
                tag = "⚠"
            elif fail_n > 0:
                row_color = "#cc6600"   # orange — failures
                tag = f"✗{fail_n}"
            else:
                row_color = "#2ca02c" if done_n > 0 else "#888"
                tag = ""
            lines.append((
                f"  Chain{c_idx}",
                f"{done_n:>3d}/{CELLS_PER_CHAIN_NEW} {tag}",
                row_color, "normal",
            ))
        lines.append(("__SEP__", "", "", ""))

    # GPU util 一行一張卡，數字靠右對齊
    if v100["gpus"]:
        lines.append(("V100 GPU", "使用率 / 記憶體", "black", "bold"))
        for g in v100["gpus"]:
            util_color = "#2ca02c" if g["util_pct"] > 50 else "#888"
            lines.append((
                f"  GPU{g['idx']}",
                f"{g['util_pct']:>3d}%   {g['mem_mib']:>5d} MiB",
                util_color, "normal",
            ))
    else:
        lines.append(("V100 GPU", "（無法連線取得資料）", "#888", "normal"))
    lines.extend([
        ("__SEP__", "", "", ""),
        ("4060 GPU",
         f"{local_gpu['util_pct']:>3d}%   {local_gpu['mem_mib']:>5d} MiB",
         "black", "bold"),
        ("__SEP__", "", "", ""),
        ("完成數", f"{done:>3d}",
         "#2ca02c" if done else "black", "bold"),
        ("進行中", f"{in_prog:>3d}",
         "#1f4b80" if in_prog else "black", "bold"),
        ("失敗",   f"{failed:>3d}",
         "#d62728" if failed else "#888", "bold"),
        ("__SEP__", "", "", ""),
        ("更新時間", datetime.now().strftime("%H:%M:%S"),
         "#888", "normal"),
    ])
    y = 0.92
    line_h = 0.060
    for label, value, color, weight in lines:
        if label == "__SEP__":
            # 分隔線 — use Line2D in axes coords (axhline can't take
            # transform=ax.transAxes; it generates its own transform).
            ax.plot([0.05, 0.95], [y - line_h * 0.4, y - line_h * 0.4],
                    color="#dddddd", lw=0.6,
                    transform=ax.transAxes, clip_on=False)
            y -= line_h * 0.7
            continue
        ax.text(0.05, y, label, fontsize=9,
                color=color, weight=weight,
                transform=ax.transAxes, va="top")
        ax.text(0.98, y, value, fontsize=9, family="monospace",
                color=color, weight=weight,
                transform=ax.transAxes, va="top", ha="right")
        y -= line_h


def _render_live_curves(ax, cells: list[dict]) -> None:
    """最近驗證 AUC 訓練曲線（最新更新的前 5 個實驗格）。"""
    ax.set_title("最近驗證 AUC 訓練曲線  (最新更新的前 5 個實驗格)",
                 fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("訓練回合", fontsize=10)
    ax.set_ylabel("驗證 AUC", fontsize=10)
    ax.grid(True, alpha=0.3)

    live = [c for c in cells if c["history"] is not None
            and "val_auc" in c["history"].columns]
    live.sort(key=lambda c: c["mtime"], reverse=True)
    top = live[:5]
    if not top:
        ax.text(0.5, 0.5, "（尚無訓練歷史）",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="#888")
        return
    for c in top:
        h = c["history"]
        color = ARCH_COLOR.get(c["arch"], "#888")
        marker = ALGO_MARKER.get(c["algo"], None)
        # Label uses display-friendly arch name and Chinese partition tag
        arch_short = {"lstm": "LSTM", "mamba": "Mamba",
                      "spiking_expand2": "Spiking"}.get(c["arch"], c["arch"])
        label = (f"{arch_short} · {c['algo']} · "
                 f"{PARTITION_LABEL.get(c['partition'], c['partition'])} · "
                 f"種子 {c['seed']}")
        ax.plot(h["round"], h["val_auc"], color=color, lw=1.5, alpha=0.85,
                marker=marker, markersize=5, markevery=max(len(h) // 8, 1),
                label=label)
    # Legend OUTSIDE the plot area (top-right of axes) so it doesn't
    # overlap with data lines, especially near the rightmost rounds.
    ax.legend(fontsize=9, loc="center left",
              bbox_to_anchor=(1.005, 0.5), framealpha=0.95,
              title="實驗格", title_fontsize=9)
    ax.set_ylim(0.5, 1.0)


def _format_group_row(g: dict) -> str:
    """彙整結果表的單列 HTML。"""
    if g["ci_low"] > 0:
        css = "background:#d4f4dd"
        sig = "★ 勝"
    elif g["ci_high"] < 0:
        css = "background:#fcdada"
        sig = "↓ 負"
    else:
        css = ""
        sig = "—"
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
    v100: dict, local_gpu: dict, eta: dict, png_name: str,
) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    done = sum(1 for c in cells if c["status"] == "done")
    in_prog = sum(1 for c in cells if c["status"] == "in_progress")
    failed = sum(1 for c in cells if c["status"] == "failed")
    path_d_cells = sum(1 for c in cells
                       if c["source"] == "V100"
                       and c["algo"] in KNOWN_ALGOS
                       and c["status"] == "done")

    short_eta, long_eta = _format_eta(eta)

    # ETA 細項：各架構的剩餘實驗格 + 經驗 wall
    eta_arch_rows = ""
    for arch in KNOWN_ARCHS:
        info = eta["per_arch"][arch]
        src_label = (
            f"實測 (n={info['empirical_n']})" if info["empirical_n"] > 0
            else "預估"
        )
        eta_arch_rows += (
            f"<tr><td>{arch}</td>"
            f"<td>{info['done']}/{info['target']}</td>"
            f"<td>{info['remaining_cells']}</td>"
            f"<td>{info['wall_s']/60:.1f} 分</td>"
            f"<td>{src_label}</td></tr>"
        )

    # 各 (架構, 演算法) 進度條
    progress_rows = ""
    for arch in KNOWN_ARCHS:
        for algo in KNOWN_ALGOS:
            n = sum(1 for c in cells if c["source"] == "V100"
                    and c["arch"] == arch
                    and c["algo"] == algo and c["status"] == "done")
            pct = 100 * n / 60.0
            progress_rows += (
                f"<tr><td>{arch}</td><td>{algo}</td>"
                f"<td>{n}/60</td><td>{pct:.0f}%</td></tr>"
            )

    fails = sorted(
        (c for c in cells if c["status"] == "failed"),
        key=lambda c: c["mtime"], reverse=True,
    )[:10]
    fail_rows = "".join(
        f"<tr><td>{c['name']}</td></tr>" for c in fails
    ) or '<tr><td><i>無</i></td></tr>'

    sorted_groups = sorted(
        groups, key=lambda g: (-g["mean_delta"], g["arch"], g["algo"])
    )
    grouped_rows = "".join(_format_group_row(g) for g in sorted_groups) \
        or '<tr><td colspan="7"><i>（尚無與 Phase 5 配對的實驗格）</i></td></tr>'

    return f"""<!doctype html>
<html lang="zh-Hant"><head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="30">
<title>Path D 大規模實驗 — 即時儀表板</title>
<style>
body {{ font-family: -apple-system, "Noto Sans CJK TC", "PingFang TC",
        "Microsoft JhengHei", sans-serif; margin: 18px; max-width: 1500px; }}
h1 {{ margin-bottom: 4px; }}
h3 {{ margin-top: 1.5em; }}
.ts {{ color: #666; font-size: 0.9em; }}
.kpi {{ display: inline-block; padding: 4px 14px; margin-right: 8px;
       background: #f4f4f4; border-radius: 4px; font-weight: bold; }}
.kpi.win {{ background: #d4f4dd; color: #1d6b2a; }}
.kpi.fail {{ background: #fcdada; color: #8a1f1f; }}
.kpi.eta {{ background: #e0eaf5; color: #1f4b80; }}
table {{ border-collapse: collapse; margin: 8px 0; }}
th, td {{ border: 1px solid #ccc; padding: 4px 10px; text-align: left;
         font-size: 0.85em; }}
th {{ background: #f4f4f4; }}
img {{ max-width: 100%; border: 1px solid #ddd; }}
.flex {{ display: flex; gap: 24px; flex-wrap: wrap; }}
.legend {{ font-size: 0.85em; color: #555; margin-bottom: 6px; }}
</style>
</head><body>
<h1>Path D SAM 家族即時儀表板</h1>
<p class="ts">更新時間：{ts} · 自動重新整理 30 秒</p>
<p>
<span class="kpi">Path D 進度：{path_d_cells} / {PATH_D_TARGET_CELLS}</span>
<span class="kpi">完成 {done}</span>
<span class="kpi">進行中 {in_prog}</span>
<span class="kpi{' fail' if failed else ''}">失敗 {failed}</span>
<span class="kpi eta">{long_eta}</span>
</p>

<img src="{png_name}?t={int(time.time())}" alt="dashboard">

<h3>彙整結果（與 Phase 5 基線配對比較）</h3>
<p class="legend">配對規則：fedscam / fedmoswa 對應 fedavg；fedgmt 對應 fedadam。
   ★ 顯著優 = 95% 信賴區間完全為正，Path D 演算法顯著優於基線；
   ↓ 顯著劣 = 95% 信賴區間完全為負，Path D 演算法顯著劣於基線；
   — 無差異 = 信賴區間橫跨 0，差異未達顯著水準。</p>
<table>
<tr><th>架構</th><th>演算法</th><th>資料切分</th><th>種子數</th>
    <th>平均 Δ AUC</th><th>95% 信賴區間</th><th>顯著性</th></tr>
{grouped_rows}
</table>

<div class="flex">
<div>
<h3>各 (架構 × 演算法) 完成進度</h3>
<table>
<tr><th>架構</th><th>演算法</th><th>已完成 / 目標</th><th>百分比</th></tr>
{progress_rows}
</table>
</div>
<div>
<h3>各架構 ETA 細項</h3>
<table>
<tr><th>架構</th><th>已完成</th><th>剩餘格數</th>
    <th>單格耗時</th><th>來源</th></tr>
{eta_arch_rows}
</table>
</div>
<div>
<h3>近期失敗的實驗格</h3>
<table>
<tr><th>實驗格名稱</th></tr>
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
            eta = compute_eta(cells)

            render_png(cells, groups, v100_status, local_gpu, eta, png)
            html.write_text(render_html(
                cells, groups, v100_status, local_gpu, eta, "status.png",
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
