#!/usr/bin/env python3
"""Summarize every run's final metrics + render a comparison plot."""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ART = Path("artifacts")
LOGS = ART / "logs"
PLOTS = ART / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)


def main() -> None:
    rows = []
    fig, ax = plt.subplots(figsize=(10, 6))

    for hist_path in sorted(list(LOGS.glob("full_*_history.csv")) + list(LOGS.glob("faithful_*_history.csv"))):
        name = hist_path.stem.replace("_history", "")
        df = pd.read_csv(hist_path)
        # try load config for variant info
        cfg_path = LOGS / f"{name}_config.json"
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        rows.append({
            "run": name,
            "variant": cfg.get("variant", "?"),
            "rounds": len(df),
            "final_train_loss": float(df["train_loss"].iloc[-1]),
            "final_test_loss": float(df["test_loss"].iloc[-1]),
            "best_test_loss": float(df["test_loss"].min()),
            "final_test_metric": float(df["test_metric"].dropna().iloc[-1]) if df["test_metric"].notna().any() else None,
            "final_epsilon": float(df["epsilon"].dropna().iloc[-1]) if df["epsilon"].notna().any() else None,
            "total_time_s": float(df["duration_s"].sum()),
            "dp": cfg.get("dp", {}).get("enabled", False),
        })
        ax.plot(df["round"], df["test_loss"], marker="o", label=name)

    ax.set_xlabel("Round")
    ax.set_ylabel("Test loss (scaled MSE or combined loss)")
    ax.set_yscale("log")
    ax.set_title("Federated Training: test loss per round, all runs")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    plot_path = PLOTS / "all_runs_test_loss.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    summary = pd.DataFrame(rows)
    csv_path = LOGS / "runs_summary.csv"
    summary.to_csv(csv_path, index=False)

    md_path = ART / "RESULTS.md"
    # Hand-roll a markdown table to avoid the tabulate dependency.
    cols = list(summary.columns)
    lines = ["# Federated Training Results", "",
             f"Plot: `{plot_path.relative_to(ART)}`", "",
             "| " + " | ".join(cols) + " |",
             "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in summary.iterrows():
        lines.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r.tolist()) + " |")
    md_path.write_text("\n".join(lines) + "\n")

    print(f"wrote {csv_path}")
    print(f"wrote {plot_path}")
    print(f"wrote {md_path}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
