#!/usr/bin/env python3
"""Live SAM-family sweep dashboard (phase5_dashboard.py style).

Polls V100 (rsync history.csv + summary.json to a local mirror) and
4060 locally, renders a 4-panel matplotlib figure:

* Panel A — val_auc training curves per cell, coloured by partition
* Panel B — per-round duration_s drift line (detects late-cell slowdown)
* Panel C — test_auc distribution (strip plot) by (algo, partition) for
            completed cells
* Panel D — progress + GPU util status bars

Pair with ``python -m http.server`` in --out to serve over Tailscale.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


SSH_OPTS = [
    "-p", "51419", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
    "-i", str(Path.home() / ".ssh/id_ed25519"),
]
SSH_TARGET = "leo07010@203.145.216.194"
MIRROR_DIR = Path("/tmp/v100_sam_mirror")

# Partition colour palette — IID green, Dirichlet warm gradient by alpha.
PARTITION_COLORS = {
    "iid":              "#2ca02c",   # green
    "dirichlet_a0p05":  "#67001f",   # dark red (most heterogeneous)
    "dirichlet_a0p10":  "#b2182b",
    "dirichlet_a0p50":  "#d6604d",
    "dirichlet_a1p00":  "#f4a582",
    "dirichlet_a5p00":  "#fddbc7",   # light salmon (most homogeneous)
}

# Algorithm marker for strip plot
ALGO_MARKERS = {"fedscam": "o", "fedgmt": "^"}


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
    """Mirror V100 cell directories into MIRROR_DIR (summary.json + history.csv)."""
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
    # Trailing s<seed>
    m = re.search(r"_s(\d+)$", name)
    seed = int(m.group(1)) if m else None
    # n<N> right before s
    m = re.search(r"_n(\d+)_s\d+$", name)
    n_clients = int(m.group(1)) if m else None
    # Partition tag
    partition = "unknown"
    if "_iid_" in name:
        partition = "iid"
    else:
        m = re.search(r"_dirichlet_a(\d+p\d+)_", name)
        if m:
            partition = f"dirichlet_a{m.group(1)}"
    # Algo
    algo = "unknown"
    for known in ("fedscam", "fedgmt"):
        if f"_{known}_" in name:
            algo = known
            break
    return {"algo": algo, "partition": partition, "seed": seed, "n_clients": n_clients}


def collect_cells(root: Path, label: str) -> list[dict]:
    """Read all v7_*/ dirs under root, return list of cell dicts.

    Each dict: name, algo, partition, seed, source (label), history (DataFrame
    or None), test_auc (float or None), status ('done'|'in_progress').
    """
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
        if summary_path.exists():
            try:
                summary = json.load(open(summary_path))
                test_auc = summary.get("test_auc")
                status = "done"
            except Exception:
                pass
        cells.append({
            "name": d.name, "source": label,
            "algo": meta["algo"], "partition": meta["partition"],
            "seed": meta["seed"], "history": hist,
            "test_auc": test_auc, "status": status,
        })
    return cells


def probe_v100_status() -> dict:
    chains = {}
    for c in range(4):
        log = _ssh_cmd(f"cat ~/fl-oran-tmc/logs/v100_sam_chain{c}.log 2>/dev/null")
        chains[c] = {
            "done": len(re.findall(r"\[gpu\d+\] DONE", log)),
            "fail": len(re.findall(r"\[gpu\d+\] FAIL", log)),
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
    master_alive = bool(_ssh_cmd("pgrep -f v100_sam_family_launcher").strip())
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


def render_png(cells: list[dict], v100: dict, local_gpu: dict,
               out_path: Path) -> None:
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.5, 1.5, 1], hspace=0.32, wspace=0.30)

    # === Panel A: val_auc curves over rounds ===
    axA = fig.add_subplot(gs[0, 0:2])
    axA.set_title("val_auc training curves (live)", fontsize=11)
    axA.set_xlabel("round")
    axA.set_ylabel("val_auc")
    axA.set_xlim(0, 100)
    axA.set_ylim(0.5, 1.0)
    axA.grid(alpha=0.3)
    for cell in cells:
        h = cell["history"]
        if h is None or "val_auc" not in h.columns:
            continue
        color = PARTITION_COLORS.get(cell["partition"], "#888888")
        ls = "-" if cell["algo"] == "fedscam" else "--"
        alpha = 0.85 if cell["status"] == "done" else 0.45
        axA.plot(h["round"], h["val_auc"], color=color, linestyle=ls,
                 lw=1.0, alpha=alpha)
    # Legend
    from matplotlib.lines import Line2D
    partition_handles = [
        Line2D([0], [0], color=c, lw=2, label=p)
        for p, c in PARTITION_COLORS.items()
    ]
    algo_handles = [
        Line2D([0], [0], color="#444", lw=2, ls="-", label="fedscam"),
        Line2D([0], [0], color="#444", lw=2, ls="--", label="fedgmt"),
    ]
    leg1 = axA.legend(handles=partition_handles, loc="lower right",
                      fontsize=7, title="partition", title_fontsize=8,
                      framealpha=0.9, ncol=2)
    axA.add_artist(leg1)
    axA.legend(handles=algo_handles, loc="upper left", fontsize=7,
               title="algo", title_fontsize=8, framealpha=0.9)

    # === Panel B: per-round duration_s drift ===
    axB = fig.add_subplot(gs[0, 2])
    axB.set_title("per-round duration_s (drift)", fontsize=11)
    axB.set_xlabel("round")
    axB.set_ylabel("seconds")
    axB.grid(alpha=0.3)
    for cell in cells:
        h = cell["history"]
        if h is None or "duration_s" not in h.columns:
            continue
        color = PARTITION_COLORS.get(cell["partition"], "#888888")
        axB.plot(h["round"], h["duration_s"], color=color, lw=0.8,
                 alpha=0.5)

    # === Panel C: test_auc distribution by partition ===
    axC = fig.add_subplot(gs[1, 0:2])
    axC.set_title("test_auc by partition × algorithm (completed cells)", fontsize=11)
    axC.set_ylabel("test_auc")
    axC.grid(alpha=0.3, axis="y")
    # Place each (algo, partition) bucket on x axis
    partitions = list(PARTITION_COLORS.keys())
    algos = ["fedscam", "fedgmt"]
    x_labels = []
    x_positions = []
    for p_idx, partition in enumerate(partitions):
        for a_idx, algo in enumerate(algos):
            x = p_idx * 2.5 + a_idx * 1.0
            x_positions.append(x)
            x_labels.append(f"{algo[:3]}\n{partition.replace('dirichlet_', '')}")
            bucket = [c["test_auc"] for c in cells
                      if c["partition"] == partition and c["algo"] == algo
                      and c["test_auc"] is not None]
            if bucket:
                color = PARTITION_COLORS[partition]
                # Jittered scatter
                import numpy as np
                xs = np.random.RandomState(p_idx * 10 + a_idx).normal(x, 0.05, len(bucket))
                marker = ALGO_MARKERS.get(algo, "o")
                axC.scatter(xs, bucket, color=color, marker=marker, s=50,
                            edgecolor="black", linewidth=0.6,
                            alpha=0.85, zorder=3)
                if len(bucket) >= 1:
                    axC.scatter([x], [sum(bucket)/len(bucket)], color="black",
                                marker="_", s=200, lw=2, zorder=4)
    axC.set_xticks(x_positions)
    axC.set_xticklabels(x_labels, fontsize=7, rotation=0)
    # Reference lines from Phase 5 baselines
    axC.axhline(0.9159, color="#666", ls=":", lw=0.6, alpha=0.5)
    axC.text(max(x_positions) + 0.3, 0.9159, "FedAvg IID 0.9159",
             fontsize=6, va="center", color="#666")
    axC.axhline(0.9178, color="#666", ls=":", lw=0.6, alpha=0.5)
    axC.text(max(x_positions) + 0.3, 0.9178, "FedAdam IID 0.9178",
             fontsize=6, va="center", color="#666")
    axC.set_ylim(0.6, 0.95)

    # === Panel D: progress + GPU status (right column bottom) ===
    axD = fig.add_subplot(gs[1, 2])
    axD.axis("off")
    v100_done = sum(c["status"] == "done" for c in cells if c["source"] == "V100")
    off_done = sum(c["status"] == "done" for c in cells
                   if c["source"] == "4060-off")
    on_done = sum(c["status"] == "done" for c in cells
                  if c["source"] == "4060-on")
    master_marker = "ALIVE" if v100["master_alive"] else "DEAD"
    master_color = "#2ca02c" if v100["master_alive"] else "#d62728"
    lines = [
        f"V100 sweep: {v100_done}/60 ({100*v100_done/60:.0f}%)",
        f"  master: {master_marker}",
        "",
        "V100 GPU util:",
    ]
    for g in v100["gpus"]:
        lines.append(f"  gpu{g['idx']}: {g['util_pct']:>3d}%  ({g['mem_mib']} MiB)")
    lines += [
        "",
        f"4060 γ=0: {off_done}/5",
        f"4060 γ=1: {on_done}/5",
        f"  GPU util: {local_gpu['util_pct']}%  ({local_gpu['mem_mib']} MiB)",
        "",
        f"refreshed {datetime.now():%H:%M:%S}",
    ]
    y = 0.95
    for line in lines:
        color = master_color if "master:" in line else "black"
        weight = "bold" if line and line[0] != " " and ":" in line else "normal"
        axD.text(0.02, y, line, fontsize=9, family="monospace",
                 color=color, weight=weight,
                 transform=axD.transAxes, va="top")
        y -= 0.055

    fig.suptitle(
        f"SAM-family live status — {datetime.now():%Y-%m-%d %H:%M:%S}",
        fontsize=13, y=0.995,
    )
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()


def render_html(cells: list[dict], v100: dict, local_gpu: dict,
                png_name: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    v100_done = sum(c["status"] == "done" for c in cells if c["source"] == "V100")
    off_done = sum(c["status"] == "done" for c in cells
                   if c["source"] == "4060-off")
    on_done = sum(c["status"] == "done" for c in cells
                  if c["source"] == "4060-on")
    chain_rows = "".join(
        f"<tr><td>gpu{c}</td><td>{st['done']}/15</td><td>{st['fail']}</td></tr>"
        for c, st in sorted(v100["chains"].items())
    )
    gpu_rows = "".join(
        f"<tr><td>gpu{g['idx']}</td><td>{g['mem_mib']} MiB</td>"
        f"<td>{g['util_pct']}%</td></tr>"
        for g in v100["gpus"]
    )
    done_cells = sorted(
        (c for c in cells if c["status"] == "done"),
        key=lambda c: (c["source"], c["algo"], c["partition"], c["seed"] or 0),
    )
    table_rows = "".join(
        f"<tr><td>{c['source']}</td><td>{c['algo']}</td>"
        f"<td>{c['partition']}</td><td>{c['seed']}</td>"
        f"<td>{c['test_auc']:.4f}</td></tr>"
        for c in done_cells if c["test_auc"] is not None
    ) or '<tr><td colspan="5"><i>(no completed cells with test_auc yet)</i></td></tr>'

    return f"""<!doctype html>
<html><head>
<meta http-equiv="refresh" content="30">
<title>SAM-family sweep dashboard</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 18px; max-width: 1300px; }}
h1 {{ margin-bottom: 4px; }}
.ts {{ color: #666; font-size: 0.9em; }}
table {{ border-collapse: collapse; margin: 8px 0; }}
th, td {{ border: 1px solid #ccc; padding: 3px 8px; text-align: left; font-size: 0.85em; }}
th {{ background: #f4f4f4; }}
img {{ max-width: 100%; border: 1px solid #ddd; }}
.flex {{ display: flex; gap: 24px; flex-wrap: wrap; }}
</style>
</head><body>
<h1>SAM-family sweep dashboard</h1>
<p class="ts">refreshed {ts} · auto-refresh 30s · V100 {v100_done}/60, 4060 γ=0 {off_done}/5, γ=1 {on_done}/5</p>

<img src="{png_name}?t={int(time.time())}" alt="dashboard">

<div class="flex">
<div>
<h3>V100 chain progress</h3>
<table><tr><th>chain</th><th>DONE/15</th><th>FAIL</th></tr>{chain_rows}</table>
</div>
<div>
<h3>V100 GPU util (live)</h3>
<table><tr><th>GPU</th><th>mem</th><th>util</th></tr>{gpu_rows}</table>
</div>
<div>
<h3>4060 GPU util</h3>
<table><tr><th>mem</th><th>util</th></tr><tr>
<td>{local_gpu['mem_mib']} MiB</td><td>{local_gpu['util_pct']}%</td>
</tr></table>
</div>
</div>

<h3>Completed cells (test_auc)</h3>
<table>
<tr><th>source</th><th>algo</th><th>partition</th><th>seed</th><th>test_auc</th></tr>
{table_rows}
</table>
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
            cells = []
            cells += [{**c, "source": "V100"} for c in collect_cells(MIRROR_DIR, "V100")]
            cells += [{**c, "source": "4060-off"} for c in collect_cells(
                Path("artifacts/v7_fedgmt_kl_off"), "4060-off")]
            cells += [{**c, "source": "4060-on"} for c in collect_cells(
                Path("artifacts/v7_fedgmt_kl_on"), "4060-on")]
            render_png(cells, v100_status, local_gpu, png)
            html.write_text(render_html(cells, v100_status, local_gpu, "status.png"))
            done_v100 = sum(c["status"] == "done" for c in cells if c["source"] == "V100")
            done_off = sum(c["status"] == "done" for c in cells if c["source"] == "4060-off")
            done_on = sum(c["status"] == "done" for c in cells if c["source"] == "4060-on")
            print(f"[{datetime.now():%H:%M:%S}] V100 {done_v100}/60, "
                  f"4060 γ=0 {done_off}/5 γ=1 {done_on}/5, "
                  f"cells_with_history={sum(1 for c in cells if c['history'] is not None)}",
                  flush=True)
        except Exception as e:
            print(f"[{datetime.now():%H:%M:%S}] error: {e}", flush=True)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
