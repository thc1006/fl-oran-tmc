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
- Panel B (top-right): sweep progress matrix, 5 archs × 3 algos, each
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
#   - xLSTM (no V100 pilot yet, 2026-05-18 4060 smoke: steady 0.3s/round
#     after torch.compile warmup, ≈ LSTM tier; use lstm fallback)
#   - Mamba-3 (no V100 pilot yet, 2026-05-18 4060 smoke: steady 0.4s/round,
#     slightly slower than Mamba due to extra λ/θ projections + β term;
#     use 1.05× mamba estimate as a conservative placeholder)
# As more Path D cells complete on V100 the empirical mean overrides
# this fallback per arch (see compute_eta).
ARCH_WALL_FALLBACK_S = {
    "lstm":             520.0,
    "mamba":            813.0,
    "spiking_expand2":  1031.0,
    "xlstm":            520.0,    # placeholder pending V100 pilot (#38)
    "mamba3":           854.0,    # 1.05× mamba, placeholder pending #38
}
N_PARALLEL_CHAINS = 4
# Cells per chain after shard 1/4 — useful for chain-progress display.
# Path D extension: 900-cell spec sharded /4 = 225 nominal/chain;
# --skip-existing-summary trims the 540 done core cells, leaving 90
# new (xLSTM + Mamba-3) cells per chain. Used purely for "X/Y" display.
CELLS_PER_CHAIN_NEW = 90

# Per-arch colour (used in curves + forest plot row groups).
ARCH_COLOR = {
    "lstm":             "#1f77b4",   # blue
    "mamba":            "#2ca02c",   # green
    "spiking_expand2":  "#9467bd",   # purple
    "xlstm":            "#ff7f0e",   # orange (NeurIPS 2024 extension of LSTM)
    "mamba3":           "#d62728",   # red (Mar 2026 extension of Mamba)
}
ALGO_MARKER = {"fedscam": "o", "fedgmt": "^", "fedmoswa": "s"}
KNOWN_ARCHS = ("lstm", "mamba", "spiking_expand2", "xlstm", "mamba3")
KNOWN_ALGOS = ("fedscam", "fedgmt", "fedmoswa")
ARCH_DISPLAY = {
    "lstm": "LSTM", "mamba": "Mamba", "spiking_expand2": "Spiking",
    "xlstm": "xLSTM", "mamba3": "Mamba-3",
}
# Path D core = original 3 archs (Phase 5 reuse); extension = 2 new archs.
CORE_ARCHS = ("lstm", "mamba", "spiking_expand2")
EXT_ARCHS = ("xlstm", "mamba3")
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

PATH_D_TARGET_CELLS = 900   # 5 archs × 3 algos × 6 partitions × 10 seeds


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
    cells: list[dict], groups: list[dict], out_path: Path,
) -> None:
    """繪製聚焦科學結果的 2 區段快照圖。

    重新設計 2026-05-19：移除進度矩陣與硬體側欄（HTML 儀表板已以
    原生 CSS 呈現且更清晰）。PNG 僅保留 HTML 無法取代的兩件事——
    依演算法分面的 Δ AUC 森林圖，與即時驗證 AUC 訓練曲線。
    """
    # CJK-capable font: the multi-CJK Noto .ttc registers as "Noto Sans
    # CJK JP" but the same file carries the TC glyph variants, so
    # Traditional Chinese renders correctly. AR PL UMing CN is the
    # Ubuntu-bundled fallback.
    matplotlib.rcParams["font.sans-serif"] = [
        "Noto Sans CJK JP", "AR PL UMing CN", "AR PL UKai CN",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(16, 10.5), constrained_layout=True)
    fig.set_constrained_layout_pads(
        w_pad=0.10, h_pad=0.16, hspace=0.12, wspace=0.06,
    )
    gs = fig.add_gridspec(2, 3, height_ratios=[2.3, 1.0])

    # 三個演算法分面共用 x 軸範圍，方便直接比較效果量。
    paired = [g for g in groups if g["n"] > 0]
    if paired:
        max_abs = max(
            max(abs(g["ci_low"]), abs(g["ci_high"]), abs(g["mean_delta"]))
            for g in paired
        )
        max_abs = max(max_abs * 1.18, 0.012)
    else:
        max_abs = 0.05

    for j, algo in enumerate(KNOWN_ALGOS):
        ax = fig.add_subplot(gs[0, j])
        _render_forest_panel(ax, groups, algo, max_abs, first=(j == 0))

    ax_curves = fig.add_subplot(gs[1, :])
    _render_live_curves(ax_curves, cells)

    fig.suptitle(
        "Path D · Δ AUC 對 Phase 5 基線之配對比較  與  即時訓練曲線"
        f"      {datetime.now():%Y-%m-%d %H:%M:%S}",
        fontsize=13.5, fontweight="bold",
    )
    plt.savefig(out_path, dpi=120, facecolor="white")
    plt.close()


def _render_forest_panel(
    ax, groups: list[dict], algo: str, max_abs: float, first: bool,
) -> None:
    """單一演算法的 Δ AUC 森林圖分面。

    列依 (架構, 切分) 排序，架構區塊間以淺色分隔線區隔。綠 = 95%
    信賴區間全為正（顯著優於基線），紅 = 全為負，灰 = 橫跨 0。
    三個分面共用 ``max_abs`` x 軸範圍以便直接比較效果量。``first``
    為真時才畫 y 軸標籤（三分面列集相同，只需最左側標一次）。
    """
    arch_idx = {a: i for i, a in enumerate(KNOWN_ARCHS)}
    part_idx = {p: i for i, p in enumerate(PARTITION_ORDER)}
    gp = sorted(
        (g for g in groups if g["algo"] == algo),
        key=lambda g: (arch_idx.get(g["arch"], 9),
                       part_idx.get(g["partition"], 9)),
    )
    SMALL_N = 3
    n_win = sum(1 for g in gp if g["n"] >= SMALL_N and g["ci_low"] > 0)
    n_lose = sum(1 for g in gp if g["n"] >= SMALL_N and g["ci_high"] < 0)
    n_null = len(gp) - n_win - n_lose
    ax.set_title(
        f"{algo}\n{n_win} 優   ·   {n_lose} 劣   ·   {n_null} 無顯著",
        fontsize=11, fontweight="bold", pad=8,
    )
    if not gp:
        ax.text(0.5, 0.5, "尚無配對實驗格", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="#9aa1ab")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    ax.axvline(0.0, color="#9aa1ab", lw=1.0, zorder=2)
    n = len(gp)
    prev_arch = None
    for i, g in enumerate(gp):
        y = n - 1 - i
        if prev_arch is not None and g["arch"] != prev_arch:
            ax.axhline(y + 0.5, color="#e3e6ea", lw=0.9, zorder=1)
        prev_arch = g["arch"]
        prelim = g["n"] < SMALL_N
        if prelim:
            col, band = "#b8bdc4", None
        elif g["ci_low"] > 0:
            col, band = "#16a34a", "#dcfce7"
        elif g["ci_high"] < 0:
            col, band = "#dc2626", "#fee2e2"
        else:
            col, band = "#6b7280", None
        if band is not None:
            ax.axhspan(y - 0.44, y + 0.44, color=band, zorder=1)
        ax.plot([g["ci_low"], g["ci_high"]], [y, y], color=col, lw=2.4,
                solid_capstyle="round", zorder=3)
        ax.plot([g["mean_delta"]], [y], marker="o", color=col,
                markersize=6.5, markeredgecolor="white",
                markeredgewidth=1.0, zorder=4)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(-max_abs, max_abs)
    ax.set_yticks(range(n))
    if first:
        ax.set_yticklabels(
            [f"{ARCH_DISPLAY.get(g['arch'], g['arch'])} · "
             f"{PARTITION_LABEL[g['partition']]}" for g in reversed(gp)],
            fontsize=8.5,
        )
    else:
        ax.set_yticklabels([])
    ax.set_xlabel("Δ AUC  (演算法 − 基線)", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(True, axis="x", alpha=0.22, zorder=0)


def _render_live_curves(ax, cells: list[dict]) -> None:
    """即時驗證 AUC 訓練曲線（最新更新的前 6 個實驗格）。

    每條曲線採用獨立的高對比色，而非依架構著色——延伸 sweep 階段
    最新的 cell 常同屬一個架構，依架構著色會使多條線同色而難以分辨。
    曲線末端標出當前 val_auc 數值。
    """
    ax.set_title("即時訓練曲線  ·  最新更新的 6 個實驗格之驗證 AUC",
                 fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("訓練回合", fontsize=9)
    ax.set_ylabel("驗證 AUC", fontsize=9)
    ax.grid(True, alpha=0.25)

    live = [c for c in cells if c["history"] is not None
            and "val_auc" in c["history"].columns
            and len(c["history"]) > 0]
    live.sort(key=lambda c: c["mtime"], reverse=True)
    top = live[:6]
    if not top:
        ax.text(0.5, 0.5, "（尚無訓練歷史）", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="#9aa1ab")
        ax.set_xticks([])
        ax.set_yticks([])
        return
    line_colors = ["#2563eb", "#dc2626", "#16a34a", "#d97706",
                   "#9333ea", "#0891b2"]
    for k, c in enumerate(top):
        h = c["history"]
        color = line_colors[k % len(line_colors)]
        label = (f"{ARCH_DISPLAY.get(c['arch'], c['arch'])} · {c['algo']} · "
                 f"{PARTITION_LABEL.get(c['partition'], c['partition'])} · "
                 f"種子 {c['seed']}")
        ax.plot(h["round"], h["val_auc"], color=color, lw=1.9, alpha=0.92,
                label=label, zorder=3)
        last_r = h["round"].iloc[-1]
        last_v = h["val_auc"].iloc[-1]
        ax.plot([last_r], [last_v], marker="o", color=color, markersize=5,
                markeredgecolor="white", markeredgewidth=0.8, zorder=4)
        ax.annotate(f"{last_v:.3f}", (last_r, last_v),
                    textcoords="offset points", xytext=(6, 0),
                    fontsize=8, color=color, va="center", fontweight="bold")
    ax.legend(fontsize=8.5, loc="center left", bbox_to_anchor=(1.012, 0.5),
              framealpha=0.96, title="實驗格（依最新更新排序）",
              title_fontsize=8.5)
    ax.set_ylim(0.5, 1.0)
    ax.margins(x=0.03)


DASHBOARD_CSS = """
*{box-sizing:border-box;margin:0;padding:0;}
body{background:#f4f5f7;color:#1b1f24;
  font-family:-apple-system,BlinkMacSystemFont,"Noto Sans CJK TC","PingFang TC","Microsoft JhengHei",sans-serif;
  font-size:14px;line-height:1.55;-webkit-font-smoothing:antialiased;}
main{max-width:1200px;margin:0 auto;padding:24px 22px 72px;}
.hd h1{font-size:20px;font-weight:700;letter-spacing:.01em;}
.hd .sub{color:#6b7280;font-size:12.5px;margin-top:3px;}
section{margin-top:16px;}
.grid{display:grid;gap:16px;}
.card{background:#fff;border:1px solid #e5e8ec;border-radius:14px;
  padding:18px 20px;box-shadow:0 1px 3px rgba(16,24,40,.05);}
.card h2{font-size:13px;font-weight:700;color:#3a414b;margin-bottom:13px;}
.hero{grid-template-columns:1.7fr 1fr;}
.prog-pct{font-size:32px;font-weight:800;line-height:1;}
.prog-pct small{font-size:14px;font-weight:600;color:#6b7280;}
.prog-cap{color:#6b7280;font-size:12.5px;margin:5px 0 13px;}
.track{display:flex;height:24px;background:#e9ecef;border-radius:12px;overflow:hidden;}
.seg{height:100%;}
.seg.core{background:#16a34a;}
.seg.ext{background:repeating-linear-gradient(45deg,#2563eb,#2563eb 9px,#4f86f5 9px,#4f86f5 18px);}
.prog-legend{display:flex;gap:18px;margin-top:11px;font-size:12px;color:#4b5563;}
.dot{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px;vertical-align:middle;}
.kpis{display:grid;grid-template-columns:1fr 1fr;gap:10px;}
.kpi{background:#f8f9fb;border:1px solid #e8eaee;border-radius:10px;padding:10px 12px;}
.kpi .k-lbl{font-size:11px;color:#6b7280;}
.kpi .k-val{font-size:17px;font-weight:700;margin-top:2px;}
.kpi.ok .k-val{color:#16a34a;}
.kpi.bad .k-val{color:#dc2626;}
.kpi.warn .k-val{color:#d97706;}
.kpi.info .k-val{color:#2563eb;}
.matrix{display:grid;grid-template-columns:96px repeat(3,1fr);gap:7px;}
.mx-head{font-weight:700;text-align:center;font-size:12px;color:#3a414b;padding:4px 0;}
.mx-arch{display:flex;align-items:center;font-weight:700;font-size:12.5px;}
.mx-arch.ext{color:#b45309;}
.mx-cell{border-radius:9px;padding:9px 11px;border:1px solid #e5e8ec;background:#fafbfc;}
.mx-cell.done{background:#ecfdf3;border-color:#bbf7d0;}
.mx-cell.run{background:#eff5ff;border-color:#bfd6fb;}
.mx-n{font-size:15px;font-weight:700;}
.mx-n small{font-size:11px;color:#9aa1ab;font-weight:600;}
.mx-bar{display:block;height:6px;background:#e6e8eb;border-radius:3px;margin-top:7px;overflow:hidden;}
.mx-bar i{display:block;height:100%;border-radius:3px;background:#cdd2d8;}
.mx-cell.done .mx-bar i{background:#16a34a;}
.mx-cell.run .mx-bar i{background:#2563eb;}
.cols{grid-template-columns:1fr 1fr;}
table{border-collapse:collapse;width:100%;font-size:12.5px;}
th,td{border-bottom:1px solid #eef0f2;padding:6px 10px;text-align:left;}
th{background:#f8f9fb;font-weight:700;color:#475467;font-size:11.5px;position:sticky;top:0;}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;}
tr.win td{background:#f0fdf4;}
tr.lose td{background:#fef2f2;}
tr.win td:last-child{color:#15803d;font-weight:700;}
tr.lose td:last-child{color:#b91c1c;font-weight:700;}
tr.slow td{color:#dc2626;font-weight:700;}
.scroll{max-height:430px;overflow-y:auto;border:1px solid #e5e8ec;border-radius:10px;}
.note{color:#6b7280;font-size:11.5px;margin-top:9px;}
.snap img{width:100%;border-radius:10px;border:1px solid #e5e8ec;margin-top:2px;}
ul.fails{list-style:none;font-size:12px;}
ul.fails li{padding:4px 2px;border-bottom:1px solid #f0f1f3;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;}
ul.fails li.none{color:#16a34a;font-family:inherit;}
@media(max-width:880px){
  .hero,.cols,.kpis{grid-template-columns:1fr;}
  .matrix{grid-template-columns:74px repeat(3,1fr);}
}
"""


def _format_group_row(g: dict) -> str:
    """彙整結果表的單列 HTML（與 Phase 5 基線之配對比較）。"""
    if g["ci_low"] > 0:
        cls, sig = "win", "★ 顯著優"
    elif g["ci_high"] < 0:
        cls, sig = "lose", "↓ 顯著劣"
    else:
        cls, sig = "null", "— 無差異"
    base = g.get("baseline_auc_mean")
    pathd = g.get("path_d_auc_mean")
    base_s = f"{base:.4f}" if base is not None else "—"
    pathd_s = f"{pathd:.4f}" if pathd is not None else "—"
    return (
        f"<tr class='{cls}'>"
        f"<td>{ARCH_DISPLAY.get(g['arch'], g['arch'])}</td>"
        f"<td>{g['algo']}</td>"
        f"<td>{PARTITION_LABEL[g['partition']]}</td>"
        f"<td class='num'>{g['n']}</td>"
        f"<td class='num'>{base_s}</td>"
        f"<td class='num'>{pathd_s}</td>"
        f"<td class='num'>{g['mean_delta']:+.4f}</td>"
        f"<td class='num'>[{g['ci_low']:+.4f}, {g['ci_high']:+.4f}]</td>"
        f"<td>{sig}</td>"
        f"</tr>"
    )


def render_html(
    cells: list[dict], groups: list[dict],
    v100: dict, local_gpu: dict, eta: dict, png_name: str,
) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    failed = sum(1 for c in cells if c["status"] == "failed")

    def _ndone(archs: tuple) -> int:
        return sum(1 for c in cells
                   if c["source"] == "V100" and c["arch"] in archs
                   and c["algo"] in KNOWN_ALGOS and c["status"] == "done")

    core_done = _ndone(CORE_ARCHS)
    ext_done = _ndone(EXT_ARCHS)
    total_done = core_done + ext_done
    core_w = 100.0 * core_done / PATH_D_TARGET_CELLS
    ext_w = 100.0 * ext_done / PATH_D_TARGET_CELLS
    pct_total = 100.0 * total_done / PATH_D_TARGET_CELLS

    short_eta, long_eta = _format_eta(eta)

    # 即時狀態 KPI
    master_txt = "執行中" if v100["master_alive"] else "閒置"
    master_cls = "ok" if v100["master_alive"] else "warn"
    gpus = v100.get("gpus") or []
    n_gpu_busy = sum(1 for g in gpus if g["util_pct"] > 5)
    gpu_txt = f"{n_gpu_busy} / {len(gpus)} 運轉" if gpus else "無資料"
    gpu_cls = "ok" if (gpus and n_gpu_busy == len(gpus)) else (
        "warn" if gpus else "")
    eta_done = eta["remaining_seconds"] < 60

    # 進度矩陣 5 架構 × 3 演算法
    mx = ["<div></div>"]
    for algo in KNOWN_ALGOS:
        mx.append(f'<div class="mx-head">{algo}</div>')
    for arch in KNOWN_ARCHS:
        ext_cls = " ext" if arch in EXT_ARCHS else ""
        mx.append(f'<div class="mx-arch{ext_cls}">{ARCH_DISPLAY[arch]}</div>')
        for algo in KNOWN_ALGOS:
            n = sum(1 for c in cells if c["source"] == "V100"
                    and c["arch"] == arch and c["algo"] == algo
                    and c["status"] == "done")
            pct = 100.0 * n / 60.0
            state = "done" if n >= 60 else "run" if n > 0 else "idle"
            mx.append(
                f'<div class="mx-cell {state}">'
                f'<span class="mx-n">{n}<small>/60</small></span>'
                f'<span class="mx-bar"><i style="width:{pct:.0f}%"></i></span>'
                f"</div>"
            )
    matrix_html = "".join(mx)

    # 各架構 ETA
    eta_rows = ""
    for arch in KNOWN_ARCHS:
        info = eta["per_arch"][arch]
        src = (f"實測 n={info['empirical_n']}" if info["empirical_n"] > 0
               else "預估")
        eta_rows += (
            f"<tr><td>{ARCH_DISPLAY[arch]}</td>"
            f"<td class='num'>{info['done']}/{info['target']}</td>"
            f"<td class='num'>{info['remaining_cells']}</td>"
            f"<td class='num'>{info['wall_s']/60:.1f} 分</td>"
            f"<td>{src}</td></tr>"
        )

    # V100 4×GPU + chain 健康度（GPU N ↔ chain N 一對一對應）
    gpu_by_idx = {g["idx"]: g for g in (v100.get("gpus") or [])}
    chains = v100["chains"]
    have_v100 = bool(gpu_by_idx) or any(
        st["done"] or st["fail"] for st in chains.values())
    if have_v100:
        done_vals = [st["done"] for st in chains.values()]
        mx_done = max(done_vals) if done_vals else 0
        gpu_rows = ""
        for i in range(4):
            st = chains.get(i, {"done": 0, "fail": 0})
            g = gpu_by_idx.get(i)
            dn, fn = st["done"], st["fail"]
            util = f"{g['util_pct']}%" if g else "—"
            mem = f"{g['mem_mib'] / 1024:.1f} GB" if g else "—"
            # util 是瞬時取樣，聯邦訓練在回合/客戶端切換間會自然落到
            # 0%（CPU-bound 的聚合階段），故不以 util 觸發警示——真正
            # 的卡住由 done 進度落後於同儕來偵測。
            slow = (mx_done - dn) > 5 and dn < mx_done * 0.5
            if fn:
                tag, cls = f"✗ {fn} 失敗", "slow"
            elif slow:
                tag, cls = "進度落後", "slow"
            else:
                tag, cls = "正常", ""
            gpu_rows += (
                f"<tr class='{cls}'><td>GPU {i} · Chain {i}</td>"
                f"<td class='num'>{util}</td><td class='num'>{mem}</td>"
                f"<td class='num'>{dn}/{CELLS_PER_CHAIN_NEW}</td>"
                f"<td>{tag}</td></tr>"
            )
    else:
        gpu_rows = ("<tr><td colspan='5'>（無法連線 V100 取得 "
                    "GPU / chain 狀態）</td></tr>")

    # Δ AUC 配對表
    sorted_groups = sorted(
        groups, key=lambda g: (-g["mean_delta"], g["arch"], g["algo"])
    )
    grouped_rows = "".join(_format_group_row(g) for g in sorted_groups) \
        or ('<tr><td colspan="9"><i>（尚無與 Phase 5 配對的實驗格）'
            '</i></td></tr>')

    # 近期失敗
    fails = sorted((c for c in cells if c["status"] == "failed"),
                   key=lambda c: c["mtime"], reverse=True)[:8]
    fail_html = ("".join(f"<li>{c['name']}</li>" for c in fails)
                 if fails else "<li class='none'>無</li>")

    return f"""<!doctype html>
<html lang="zh-Hant"><head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="30">
<title>Path D 即時儀表板 · 5 架構 SAM 家族</title>
<style>{DASHBOARD_CSS}</style>
</head><body>
<main>
<div class="hd">
  <h1>Path D · SAM 家族 × 5 架構大規模實驗</h1>
  <div class="sub">更新 {ts} · 每 30 秒自動刷新 · 核心 540 + 擴充 360 = 900 實驗格</div>
</div>

<section class="grid hero">
  <div class="card">
    <h2>整體進度</h2>
    <div class="prog-pct">{pct_total:.0f}%<small> · {total_done} / {PATH_D_TARGET_CELLS} 實驗格</small></div>
    <div class="prog-cap">核心 3 架構 {core_done}/540 已完成 · 擴充 2 架構（xLSTM + Mamba-3）{ext_done}/360 進行中</div>
    <div class="track">
      <div class="seg core" style="width:{core_w:.2f}%"></div>
      <div class="seg ext" style="width:{ext_w:.2f}%"></div>
    </div>
    <div class="prog-legend">
      <span><i class="dot" style="background:#16a34a"></i>核心已完成</span>
      <span><i class="dot" style="background:#2563eb"></i>擴充進行中</span>
      <span><i class="dot" style="background:#e9ecef"></i>尚未開始</span>
    </div>
  </div>
  <div class="card">
    <h2>即時狀態</h2>
    <div class="kpis">
      <div class="kpi {'ok' if eta_done else 'info'}"><div class="k-lbl">預計剩餘</div><div class="k-val">{short_eta}</div></div>
      <div class="kpi {master_cls}"><div class="k-lbl">V100 主控</div><div class="k-val">{master_txt}</div></div>
      <div class="kpi {gpu_cls}"><div class="k-lbl">V100 GPU</div><div class="k-val">{gpu_txt}</div></div>
      <div class="kpi {'bad' if failed else 'ok'}"><div class="k-lbl">失敗實驗格</div><div class="k-val">{failed}</div></div>
    </div>
    <div class="note">{long_eta}</div>
  </div>
</section>

<section class="card">
  <h2>各（架構 × 演算法）完成進度 · 每格 60 實驗格</h2>
  <div class="matrix">{matrix_html}</div>
</section>

<section class="grid cols">
  <div class="card">
    <h2>各架構 ETA 細項</h2>
    <table>
      <tr><th>架構</th><th class="num">已完成</th><th class="num">剩餘</th><th class="num">單格耗時</th><th>來源</th></tr>
      {eta_rows}
    </table>
  </div>
  <div class="card">
    <h2>V100 — 4×GPU / Chain 健康度</h2>
    <table>
      <tr><th>GPU · Chain</th><th class="num">使用率</th><th class="num">記憶體</th><th class="num">已完成</th><th>狀態</th></tr>
      {gpu_rows}
    </table>
    <div class="note">GPU N ↔ Chain N 一對一對應;每 chain 約 {CELLS_PER_CHAIN_NEW} 個擴充實驗格(540 核心格已跳過)。V100 單卡 32 GB;使用率為瞬時取樣,回合間落到 0% 屬正常。</div>
  </div>
</section>

<section class="card">
  <h2>Δ AUC · Path D 演算法 vs Phase 5 基線（配對自助法 95% 信賴區間）</h2>
  <div class="scroll">
    <table>
      <tr><th>架構</th><th>演算法</th><th>資料切分</th><th class="num">n</th>
          <th class="num">基線 AUC</th><th class="num">Path D AUC</th>
          <th class="num">平均 Δ AUC</th><th class="num">95% 信賴區間</th><th>顯著性</th></tr>
      {grouped_rows}
    </table>
  </div>
  <div class="note">配對規則：fedscam · fedmoswa → fedavg；fedgmt → fedadam。
    ★ 顯著優 / ↓ 顯著劣 表 95% CI 完全偏離 0；— 無差異 表 CI 橫跨 0。
    擴充架構 xLSTM / Mamba-3 在 Phase 5 無對應基線，完成後僅見於進度矩陣與訓練曲線，不列入此配對表。</div>
</section>

<section class="card snap">
  <h2>完整視覺化快照 · 森林圖 · 進度矩陣 · 即時訓練曲線</h2>
  <img src="{png_name}?t={int(time.time())}" alt="dashboard snapshot">
</section>

<section class="card">
  <h2>近期失敗實驗格</h2>
  <ul class="fails">{fail_html}</ul>
</section>
</main>
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

            render_png(cells, groups, png)
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
