"""Aggregate Stage 1 v6 sweep cells into paper-grade tables + bootstrap CIs.

Produces:
  docs/RESULTS_V6_STAGE1.md   committed paper-grade markdown
  artifacts/v6_arch_sweep/aggregated.json  raw stats incl. CI bounds

Per ADR-001 D-21:
  C1: paired-bootstrap CI95 upper bound of delta_auc(Spiking, LSTM) >= -0.030
  C2: mean total_energy_ratio(Spiking_vs_LSTM) <= 0.5
  C3: paired-bootstrap CI95 lower bound of delta_auc(Mamba, LSTM) >= -0.030

n_boot = 10_000.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as sps


def load_cells(sweep_dir: Path) -> dict[tuple[str, int], dict]:
    """Read every ``<arch>_s<seed>/summary.json`` + ``energy.json`` under the sweep dir."""
    cells: dict[tuple[str, int], dict] = {}
    for cell_dir in sorted(sweep_dir.glob("*_s*")):
        if not cell_dir.is_dir():
            continue
        summary_path = cell_dir / "summary.json"
        energy_path = cell_dir / "energy.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        if energy_path.exists():
            summary["energy"] = json.loads(energy_path.read_text())
        else:
            summary["energy"] = {}
        cells[(summary["arch"], int(summary["seed"]))] = summary
    return cells


def per_arch_stats(cells: dict[tuple[str, int], dict]) -> dict[str, dict]:
    by_arch: dict[str, list[dict]] = defaultdict(list)
    for (arch, _seed), v in cells.items():
        by_arch[arch].append(v)
    out: dict[str, dict] = {}
    for arch, items in by_arch.items():
        aucs = np.array([float(v["test_auc"]) for v in items])
        f1s = np.array([float(v["test_f1"]) for v in items])
        accs = np.array([float(v["test_accuracy"]) for v in items])
        params = np.array([int(v["params_count"]) for v in items])
        energies = np.array([float(v["energy"].get("total_energy_pJ", 0.0)) for v in items])
        flops = np.array([float(v["energy"].get("flops", 0.0)) for v in items])
        sops = np.array([float(v["energy"].get("sops", 0.0)) for v in items])
        out[arch] = {
            "n": int(len(items)),
            "test_auc_mean": float(aucs.mean()),
            "test_auc_std": float(aucs.std(ddof=1)) if len(aucs) > 1 else 0.0,
            "test_f1_mean": float(f1s.mean()),
            "test_f1_std": float(f1s.std(ddof=1)) if len(f1s) > 1 else 0.0,
            "test_accuracy_mean": float(accs.mean()),
            "params_count_mean": float(params.mean()),
            "energy_pJ_mean": float(energies.mean()),
            "energy_pJ_std": float(energies.std(ddof=1)) if len(energies) > 1 else 0.0,
            "flops_mean": float(flops.mean()),
            "sops_mean": float(sops.mean()),
            "seeds": sorted(int(v["seed"]) for v in items),
        }
    return out


def _paired_aucs(cells: dict[tuple[str, int], dict], arch_a: str, arch_b: str
                 ) -> tuple[np.ndarray, np.ndarray, list[int]]:
    common = sorted(
        set(s for (a, s) in cells if a == arch_a)
        & set(s for (a, s) in cells if a == arch_b)
    )
    auc_a = np.array([cells[(arch_a, s)]["test_auc"] for s in common])
    auc_b = np.array([cells[(arch_b, s)]["test_auc"] for s in common])
    return auc_a, auc_b, common


def paired_bootstrap_delta_ci(cells: dict[tuple[str, int], dict],
                              arch_a: str, arch_b: str,
                              n_boot: int = 10_000,
                              ci_level: float = 0.95,
                              seed: int = 2026) -> dict:
    """delta_auc(arch_a, arch_b) via paired bootstrap CI on per-seed deltas."""
    auc_a, auc_b, seeds = _paired_aucs(cells, arch_a, arch_b)
    delta = auc_a - auc_b
    if len(delta) < 2:
        return {
            "n_paired_seeds": int(len(delta)),
            "delta_mean": float(delta.mean()) if len(delta) else 0.0,
            "ci_lo": None,
            "ci_hi": None,
            "wilcoxon_p": None,
            "seeds": seeds,
        }
    rng = np.random.default_rng(seed)
    n = len(delta)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(delta, size=n, replace=True)
        boot_means[i] = sample.mean()
    alpha = 1.0 - ci_level
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    # Wilcoxon as secondary diagnostic.
    try:
        wilcoxon = float(sps.wilcoxon(delta, alternative="two-sided",
                                       zero_method="wilcox").pvalue)
    except (ValueError, ZeroDivisionError):
        wilcoxon = None
    return {
        "n_paired_seeds": int(n),
        "delta_mean": float(delta.mean()),
        "delta_std": float(delta.std(ddof=1)) if n > 1 else 0.0,
        "ci_lo": lo,
        "ci_hi": hi,
        "wilcoxon_p": wilcoxon,
        "seeds": seeds,
    }


def evaluate_d21_criteria(stats: dict, deltas: dict) -> dict:
    """Apply ADR-001 D-21 GO/NO-GO criteria; return result dict + flags."""
    spiking = stats.get("spiking", {})
    lstm = stats.get("lstm", {})
    mamba_lstm = deltas.get(("mamba", "lstm"), {})
    spiking_lstm = deltas.get(("spiking", "lstm"), {})

    # C1: spiking vs lstm CI95 upper bound >= -0.030
    c1_hi = spiking_lstm.get("ci_hi")
    c1_pass = c1_hi is not None and c1_hi >= -0.030

    # C2: total_energy_ratio(spiking, lstm) <= 0.5
    if spiking.get("energy_pJ_mean", 0) > 0 and lstm.get("energy_pJ_mean", 0) > 0:
        ratio = spiking["energy_pJ_mean"] / lstm["energy_pJ_mean"]
    else:
        ratio = None
    c2_pass = ratio is not None and ratio <= 0.5

    # C3: mamba vs lstm CI95 lower bound >= -0.030 + sanity (positive_rate > 0 etc.)
    c3_lo = mamba_lstm.get("ci_lo")
    c3_pass = c3_lo is not None and c3_lo >= -0.030

    decision = "NO-GO Stage 2"
    if c1_pass and c2_pass and c3_pass:
        decision = "GO Stage 2: Spiking-led"
    elif (not c1_pass) and c3_pass and mamba_lstm.get("ci_lo", -1) > 0.005:
        decision = "GO Stage 2: Mamba-led fallback"

    return {
        "C1_accuracy_gap_spiking_vs_lstm": {
            "ci95": [spiking_lstm.get("ci_lo"), c1_hi],
            "threshold": -0.030,
            "pass": c1_pass,
        },
        "C2_energy_ratio_spiking_vs_lstm": {
            "ratio": ratio,
            "threshold": 0.5,
            "pass": c2_pass,
        },
        "C3_mamba_arm_healthy": {
            "ci95": [c3_lo, mamba_lstm.get("ci_hi")],
            "threshold": -0.030,
            "pass": c3_pass,
        },
        "decision": decision,
    }


def render_results_md(stats: dict, deltas: dict, criteria: dict) -> str:
    lines: list[str] = []
    lines.append("# Stage 1 (v6) Results — 3-Architecture Centralized Benchmark on ColO-RAN\n")
    lines.append("Generated by `scripts/aggregate_v6_results.py`. See ADR-001 D-19/D-20/D-21.\n")

    # Per-arch table
    lines.append("## Per-architecture metrics (n_seeds aggregated)\n")
    lines.append("| Arch | n | params | test AUC (mean ± std) | test F1 (mean ± std) | test acc | flops/inf | sops/inf | energy_pJ/inf |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for arch in ("lstm", "mamba", "spiking"):
        s = stats.get(arch)
        if s is None:
            continue
        lines.append(
            f"| {arch} | {s['n']} | {int(s['params_count_mean'])} | "
            f"{s['test_auc_mean']:.4f} ± {s['test_auc_std']:.4f} | "
            f"{s['test_f1_mean']:.4f} ± {s['test_f1_std']:.4f} | "
            f"{s['test_accuracy_mean']:.4f} | "
            f"{s['flops_mean']:.0f} | {s['sops_mean']:.0f} | "
            f"{s['energy_pJ_mean']:.2e} |"
        )
    lines.append("")

    # Paired deltas
    lines.append("## Pairwise delta_auc with 95% paired-bootstrap CI (n_boot=10000)\n")
    lines.append("| Comparison | n_seeds | delta mean | CI95 [lo, hi] | Wilcoxon p |")
    lines.append("|---|---|---|---|---|")
    for (a, b), d in deltas.items():
        if d.get("ci_lo") is None:
            continue
        wilcoxon = "n/a" if d.get("wilcoxon_p") is None else f"{d['wilcoxon_p']:.4f}"
        lines.append(
            f"| {a} − {b} | {d['n_paired_seeds']} | "
            f"{d['delta_mean']:+.4f} | [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}] | "
            f"{wilcoxon} |"
        )
    lines.append("")

    # Criteria
    lines.append("## ADR-001 D-21 GO/NO-GO criteria\n")
    c1 = criteria["C1_accuracy_gap_spiking_vs_lstm"]
    c2 = criteria["C2_energy_ratio_spiking_vs_lstm"]
    c3 = criteria["C3_mamba_arm_healthy"]
    def _fmt(value):
        return "n/a" if value is None else f"{value:.4f}"
    lines.append(
        f"- **C1 Spiking accuracy vs LSTM** (CI95 upper bound ≥ -0.030): "
        f"hi = {_fmt(c1['ci95'][1])} → **{'PASS' if c1['pass'] else 'FAIL'}**"
    )
    lines.append(
        f"- **C2 Spiking energy ratio ≤ 0.5**: ratio = {_fmt(c2['ratio'])} "
        f"→ **{'PASS' if c2['pass'] else 'FAIL'}**"
    )
    lines.append(
        f"- **C3 Mamba arm healthy** (CI95 lower bound ≥ -0.030): "
        f"lo = {_fmt(c3['ci95'][0])} → **{'PASS' if c3['pass'] else 'FAIL'}**"
    )
    lines.append("")
    lines.append(f"### Decision: **{criteria['decision']}**\n")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", type=str, default="artifacts/v6_arch_sweep")
    parser.add_argument("--out-md", type=str, default="docs/RESULTS_V6_STAGE1.md")
    parser.add_argument("--out-json", type=str, default="artifacts/v6_arch_sweep/aggregated.json")
    parser.add_argument("--n-boot", type=int, default=10_000)
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    cells = load_cells(sweep_dir)
    if not cells:
        raise RuntimeError(f"No v6 cells found under {sweep_dir}")

    stats = per_arch_stats(cells)
    deltas: dict[tuple[str, str], dict] = {}
    for a, b in [("mamba", "lstm"), ("spiking", "lstm"), ("spiking", "mamba")]:
        if a in stats and b in stats:
            deltas[(a, b)] = paired_bootstrap_delta_ci(cells, a, b, n_boot=args.n_boot)

    criteria = evaluate_d21_criteria(stats, deltas)

    out_md = Path(args.out_md)
    out_json = Path(args.out_json)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_results_md(stats, deltas, criteria))
    out_json.write_text(json.dumps({
        "stats": stats,
        "deltas": {f"{a}_vs_{b}": d for (a, b), d in deltas.items()},
        "criteria": criteria,
    }, indent=2))
    print(f"wrote {out_md} and {out_json}")
    print(f"decision: {criteria['decision']}")


if __name__ == "__main__":
    main()
