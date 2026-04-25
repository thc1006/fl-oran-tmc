"""Aggregate all v5 sweep cell results into paper-grade tables.

Walks ``artifacts/v5_sweep/v5_*/logs/summary.json`` (the standard cell
output dirs; ignores ``_moon_hpo``, ``_seed_checkpoint_logs`` etc.),
groups by (algorithm, alpha), and emits:

- ``artifacts/RESULTS_V5.md`` — Markdown table for the paper
- ``artifacts/v5_sweep/aggregated_table.csv`` — long-form per-cell CSV
- ``artifacts/v5_sweep/pivoted_auc.csv`` — algorithm × alpha pivot of mean ± std AUC

Designed to be safe to re-run any time during a sweep — partial cells
are picked up gracefully.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def _algo_order() -> list[str]:
    return ["fedavg", "fedprox", "fedadam", "scaffold", "feddyn", "moon"]


def collect_cells(sweep_dir: Path) -> list[dict]:
    rows = []
    for cell_dir in sorted(sweep_dir.glob("v5_*")):
        summary = cell_dir / "logs" / "summary.json"
        if not summary.exists():
            continue
        try:
            data = json.loads(summary.read_text())
        except json.JSONDecodeError:
            continue
        cfg = data.get("config", {})
        test = data.get("test", {})
        rows.append({
            "algorithm": cfg.get("algorithm"),
            "alpha": cfg.get("alpha"),
            "seed": cfg.get("seed"),
            "n_clients": cfg.get("n_clients"),
            "num_rounds": cfg.get("num_rounds"),
            "test_auc": test.get("auc"),
            "test_acc": test.get("accuracy"),
            "test_f1": test.get("f1"),
            "best_val_auc": data.get("best_val_auc"),
            "algo_kwargs": json.dumps(cfg.get("algo_kwargs") or {}),
            "name": cfg.get("name"),
        })
    return rows


def _format_mean_std(values: list[float]) -> str:
    if not values:
        return "—"
    if len(values) == 1:
        return f"{values[0]:.4f}"
    m = statistics.mean(values)
    s = statistics.stdev(values)
    return f"{m:.4f} ± {s:.4f}"


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def aggregate(rows: list[dict]) -> tuple[dict, list[float]]:
    """Group by (algo, alpha); return groups + sorted alpha list."""
    groups: dict[tuple[str, float], dict] = defaultdict(
        lambda: {"auc": [], "acc": [], "f1": [], "best_val": [], "seeds": set()}
    )
    alphas: set[float] = set()
    for r in rows:
        if r["test_auc"] is None or r["algorithm"] is None:
            continue
        key = (r["algorithm"], float(r["alpha"]))
        groups[key]["auc"].append(r["test_auc"])
        groups[key]["acc"].append(r["test_acc"])
        groups[key]["f1"].append(r["test_f1"])
        groups[key]["best_val"].append(r["best_val_auc"])
        groups[key]["seeds"].add(r["seed"])
        alphas.add(float(r["alpha"]))
    return groups, sorted(alphas)


def write_pivot_csv(groups: dict, alphas: list[float], path: Path) -> None:
    """algorithm × alpha pivot: each cell = "mean ± std (n)"."""
    rows = []
    for algo in _algo_order():
        row = {"algorithm": algo}
        for a in alphas:
            g = groups.get((algo, a))
            if g and g["auc"]:
                row[f"alpha={a}"] = _format_mean_std(g["auc"])
                row[f"n_seeds_a{a}"] = len(g["seeds"])
            else:
                row[f"alpha={a}"] = "—"
                row[f"n_seeds_a{a}"] = 0
        rows.append(row)
    write_csv(rows, path)


def write_markdown(groups: dict, alphas: list[float], path: Path,
                    cells_total: int) -> None:
    lines = [
        "# V5 Aggregated Results",
        "",
        f"_Auto-generated from {cells_total} sweep cells._",
        "",
        "## Test AUC: algorithm x Dirichlet alpha (mean +/- std across seeds)",
        "",
    ]
    header = "| Algorithm | " + " | ".join(f"alpha={a}" for a in alphas) + " |"
    sep = "|" + "|".join("---" for _ in range(1 + len(alphas))) + "|"
    lines += [header, sep]
    for algo in _algo_order():
        cells = []
        for a in alphas:
            g = groups.get((algo, a))
            if g and g["auc"]:
                txt = _format_mean_std(g["auc"])
                if len(g["auc"]) < 5:
                    txt += f" (n={len(g['auc'])})"
                cells.append(txt)
            else:
                cells.append("—")
        lines.append(f"| {algo} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Test F1 (mean +/- std)",
        "",
        header, sep,
    ]
    for algo in _algo_order():
        cells = []
        for a in alphas:
            g = groups.get((algo, a))
            if g and g["f1"]:
                cells.append(_format_mean_std(g["f1"]))
            else:
                cells.append("—")
        lines.append(f"| {algo} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Test accuracy (mean +/- std)",
        "",
        header, sep,
    ]
    for algo in _algo_order():
        cells = []
        for a in alphas:
            g = groups.get((algo, a))
            if g and g["acc"]:
                cells.append(_format_mean_std(g["acc"]))
            else:
                cells.append("—")
        lines.append(f"| {algo} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Cell counts (seeds completed per (algorithm, alpha))",
        "",
        header, sep,
    ]
    for algo in _algo_order():
        cells = []
        for a in alphas:
            g = groups.get((algo, a))
            cells.append(str(len(g["seeds"]) if g else 0))
        lines.append(f"| {algo} | " + " | ".join(cells) + " |")

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep-dir", default="artifacts/v5_sweep")
    p.add_argument("--output-md", default="artifacts/RESULTS_V5.md")
    args = p.parse_args()

    sweep_dir = Path(args.sweep_dir)
    rows = collect_cells(sweep_dir)
    if not rows:
        print(f"No cells found under {sweep_dir}")
        return

    long_csv = sweep_dir / "aggregated_table.csv"
    write_csv(rows, long_csv)
    print(f"Wrote long-form CSV: {long_csv}  ({len(rows)} cells)")

    groups, alphas = aggregate(rows)
    pivot_csv = sweep_dir / "pivoted_auc.csv"
    write_pivot_csv(groups, alphas, pivot_csv)
    print(f"Wrote pivot CSV: {pivot_csv}")

    md_path = Path(args.output_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(groups, alphas, md_path, cells_total=len(rows))
    print(f"Wrote Markdown report: {md_path}")

    print("\nCells per (algo, alpha):")
    for algo in _algo_order():
        line = f"  {algo:<10}"
        for a in alphas:
            g = groups.get((algo, a))
            line += f"  a={a}: {len(g['seeds']) if g else 0}"
        print(line)


if __name__ == "__main__":
    main()
