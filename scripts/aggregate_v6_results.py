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


def _parse_cell_name(name: str) -> tuple[str, int]:
    """Parse a cell directory name into ``(arch_label, seed)``.

    Directory name shape is ``<arch>_s<seed>[<suffix>]`` where ``suffix``
    is e.g. ``_t5`` from a recovery sweep. The suffix becomes part of
    ``arch_label`` so recovery cells aggregate as a distinct architecture.
    """
    arch, _, rest = name.partition("_s")
    if not rest:
        raise ValueError(f"unexpected cell dir name: {name!r}")
    seed_part, _, suffix = rest.partition("_")
    seed = int(seed_part)
    if suffix:
        arch_label = f"{arch}_{suffix}"
    else:
        arch_label = arch
    return arch_label, seed


def load_cells(sweep_dir: Path) -> dict[tuple[str, int], dict]:
    """Read every ``<arch>_s<seed>[_<suffix>]/summary.json`` + ``energy.json``."""
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
        # Re-derive (arch_label, seed) from the directory name so that
        # recovery sweeps with the same arch + seed but a non-empty
        # output_suffix do not collide with main-sweep cells.
        arch_label, seed = _parse_cell_name(cell_dir.name)
        # Override the arch field to match the directory-derived label
        # (so per_arch_stats and pairwise deltas key on it).
        summary["arch"] = arch_label
        summary["seed"] = seed
        cells[(arch_label, seed)] = summary
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
        # Legacy single-accounting energy (== sparsity_aware on new cells).
        energies = np.array([float(v["energy"].get("total_energy_pJ", 0.0)) for v in items])
        flops = np.array([float(v["energy"].get("flops", 0.0)) for v in items])
        sops = np.array([float(v["energy"].get("sops", 0.0)) for v in items])
        # Three-accounting fields (fall back to total_energy_pJ if a
        # cell predates the three-accounting energy_metrics fix).
        energies_gpu = np.array([
            float(v["energy"].get("total_energy_pJ_gpu_dense",
                                   v["energy"].get("total_energy_pJ", 0.0)))
            for v in items
        ])
        energies_sparse = np.array([
            float(v["energy"].get("total_energy_pJ_sparsity_aware",
                                   v["energy"].get("total_energy_pJ", 0.0)))
            for v in items
        ])
        energies_neuro = np.array([
            float(v["energy"].get("total_energy_pJ_neuromorphic",
                                   v["energy"].get("total_energy_pJ", 0.0)))
            for v in items
        ])
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
            "energy_pJ_gpu_mean": float(energies_gpu.mean()),
            "energy_pJ_sparsity_mean": float(energies_sparse.mean()),
            "energy_pJ_neuromorphic_mean": float(energies_neuro.mean()),
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
                              n_comparisons_for_bonferroni: int = 1,
                              seed: int = 2026) -> dict:
    """delta_auc(arch_a, arch_b) via paired bootstrap CI on per-seed deltas.

    ``n_comparisons_for_bonferroni`` controls a Bonferroni-corrected CI.
    With N comparisons, the per-test alpha is ci_level / N (e.g. for
    family-wise 0.95 across 8 tests, per-test alpha is 0.05/8 = 0.00625
    → CI99.375). Both the uncorrected CI and the Bonferroni-corrected CI
    are reported in the output.
    """
    auc_a, auc_b, seeds = _paired_aucs(cells, arch_a, arch_b)
    delta = auc_a - auc_b
    if len(delta) < 2:
        return {
            "n_paired_seeds": int(len(delta)),
            "delta_mean": float(delta.mean()) if len(delta) else 0.0,
            "ci_lo": None,
            "ci_hi": None,
            "ci_lo_bonferroni": None,
            "ci_hi_bonferroni": None,
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
    # Bonferroni-corrected CI: alpha_per_test = alpha / N_comparisons.
    alpha_bonf = alpha / max(n_comparisons_for_bonferroni, 1)
    lo_bonf = float(np.percentile(boot_means, 100 * alpha_bonf / 2))
    hi_bonf = float(np.percentile(boot_means, 100 * (1 - alpha_bonf / 2)))
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
        "ci_lo_bonferroni": lo_bonf,
        "ci_hi_bonferroni": hi_bonf,
        "n_comparisons_for_bonferroni": int(n_comparisons_for_bonferroni),
        "wilcoxon_p": wilcoxon,
        "seeds": seeds,
    }


def evaluate_d21_criteria(stats: dict, deltas: dict,
                           spiking_key: str = "spiking",
                           lstm_key: str = "lstm",
                           mamba_key: str = "mamba",
                           energy_field: str = "energy_pJ_mean") -> dict:
    """Apply ADR-001 D-21 GO/NO-GO criteria; return result dict + flags.

    ``spiking_key`` / ``lstm_key`` / ``mamba_key`` select which variant of
    each architecture to compare. The preregistered evaluation uses the
    original 5k cells (``"spiking"``, ``"lstm"``, ``"mamba"``). The
    matched-25k evaluation uses ``"spiking_lr5e4_25k"``, ``"lstm_25k"``,
    ``"mamba_25k"`` so all three archs are at the same step budget.
    """
    spiking = stats.get(spiking_key, {})
    lstm = stats.get(lstm_key, {})
    mamba_lstm = deltas.get((mamba_key, lstm_key), {})
    spiking_lstm = deltas.get((spiking_key, lstm_key), {})

    # C1: spiking vs lstm CI95 upper bound >= -0.030
    c1_hi = spiking_lstm.get("ci_hi")
    c1_pass = c1_hi is not None and c1_hi >= -0.030

    # C2: total_energy_ratio(spiking, lstm) <= 0.5
    sp_e = spiking.get(energy_field, 0)
    ls_e = lstm.get(energy_field, 0)
    if sp_e > 0 and ls_e > 0:
        ratio = sp_e / ls_e
    else:
        ratio = None
    c2_pass = ratio is not None and ratio <= 0.5

    # C3: mamba vs lstm CI95 lower bound >= -0.030 + sanity (positive_rate > 0 etc.)
    c3_lo = mamba_lstm.get("ci_lo")
    c3_pass = c3_lo is not None and c3_lo >= -0.030

    decision = "NO-GO Stage 2"
    if c1_pass and c2_pass and c3_pass:
        decision = "GO Stage 2: Spiking-led"
    elif c1_pass and (not c2_pass) and c3_pass:
        decision = "GO Stage 2: Trade-off study (C1 met, C2 fail)"
    elif (not c1_pass) and c3_pass and mamba_lstm.get("ci_lo", -1) > 0.005:
        decision = "GO Stage 2: Mamba-led fallback"

    return {
        "spiking_variant_evaluated": spiking_key,
        "lstm_variant_evaluated": lstm_key,
        "energy_field": energy_field,
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


def render_results_md(stats: dict, deltas: dict, criteria: dict,
                       audit_criteria: dict | None = None) -> str:
    lines: list[str] = []
    lines.append("# Stage 1 (v6) Results — 3-Architecture Centralized Benchmark on ColO-RAN\n")
    lines.append("Generated by `scripts/aggregate_v6_results.py`. See ADR-001 D-19/D-20/D-21.\n")

    # Per-arch table — main archs first, then any recovery variants.
    main_archs = [a for a in ("lstm", "mamba", "spiking") if a in stats]
    extra_archs = sorted(a for a in stats if a not in main_archs)
    arch_order = main_archs + extra_archs
    lines.append("## Per-architecture metrics (n_seeds aggregated)\n")
    lines.append("| Arch | n | params | test AUC (mean ± std) | test F1 (mean ± std) | test acc | flops/inf | sops/inf | energy_pJ/inf |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for arch in arch_order:
        s = stats[arch]
        lines.append(
            f"| {arch} | {s['n']} | {int(s['params_count_mean'])} | "
            f"{s['test_auc_mean']:.4f} ± {s['test_auc_std']:.4f} | "
            f"{s['test_f1_mean']:.4f} ± {s['test_f1_std']:.4f} | "
            f"{s['test_accuracy_mean']:.4f} | "
            f"{s['flops_mean']:.0f} | {s['sops_mean']:.0f} | "
            f"{s['energy_pJ_mean']:.2e} |"
        )
    lines.append("")

    # Paired deltas — uncorrected CI95 + Bonferroni-corrected.
    n_compare_in_table = next(
        (d.get("n_comparisons_for_bonferroni", 1) for d in deltas.values() if d), 1
    )
    lines.append(
        f"## Pairwise delta_auc with 95% paired-bootstrap CI (n_boot=10000), "
        f"and Bonferroni-corrected CI ({n_compare_in_table} comparisons)\n"
    )
    lines.append("| Comparison | n_seeds | delta mean | CI95 [lo, hi] | Bonferroni CI [lo, hi] | Wilcoxon p |")
    lines.append("|---|---|---|---|---|---|")
    for (a, b), d in deltas.items():
        if d.get("ci_lo") is None:
            continue
        wilcoxon = "n/a" if d.get("wilcoxon_p") is None else f"{d['wilcoxon_p']:.4f}"
        lo_b = d.get("ci_lo_bonferroni")
        hi_b = d.get("ci_hi_bonferroni")
        bonf = "n/a" if lo_b is None else f"[{lo_b:+.4f}, {hi_b:+.4f}]"
        lines.append(
            f"| {a} − {b} | {d['n_paired_seeds']} | "
            f"{d['delta_mean']:+.4f} | [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}] | "
            f"{bonf} | {wilcoxon} |"
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
    lines.append(f"### Decision (preregistered Spiking): **{criteria['decision']}**\n")

    if audit_criteria:
        lines.append("## ADR-001 D-21 GO/NO-GO criteria — post-hoc audit variants\n")
        for variant, c in audit_criteria.items():
            ac1 = c["C1_accuracy_gap_spiking_vs_lstm"]
            ac2 = c["C2_energy_ratio_spiking_vs_lstm"]
            ac3 = c["C3_mamba_arm_healthy"]
            lines.append(f"### Variant: `{variant}`")
            lines.append(
                f"- **C1**: hi = {_fmt(ac1['ci95'][1])} → "
                f"**{'PASS' if ac1['pass'] else 'FAIL'}**"
            )
            lines.append(
                f"- **C2**: ratio = {_fmt(ac2['ratio'])} → "
                f"**{'PASS' if ac2['pass'] else 'FAIL'}**"
            )
            lines.append(
                f"- **C3**: lo = {_fmt(ac3['ci95'][0])} → "
                f"**{'PASS' if ac3['pass'] else 'FAIL'}**"
            )
            lines.append(f"- **Decision**: **{c['decision']}**\n")
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
    # Two-pass: first count how many delta comparisons we will compute, then
    # use that as N for Bonferroni on each.
    pending_pairs: list[tuple[str, str]] = []
    for a, b in [("mamba", "lstm"), ("spiking", "lstm"), ("spiking", "mamba")]:
        if a in stats and b in stats:
            pending_pairs.append((a, b))
    for a in sorted(stats):
        if a in ("lstm", "mamba", "spiking"):
            continue
        for b in ("lstm", "mamba", "spiking"):
            if b in stats and (a, b) not in pending_pairs:
                pending_pairs.append((a, b))
    for a in sorted(stats):
        if a in ("lstm", "mamba", "spiking") or a.startswith(("lstm_", "mamba_")):
            continue
        for b in sorted(stats):
            if a == b:
                continue
            if (a, b) in pending_pairs:
                continue
            if b.startswith(("lstm_", "mamba_")):
                pending_pairs.append((a, b))
    for a in sorted(stats):
        if not a.startswith("mamba_"):
            continue
        for b in sorted(stats):
            if not b.startswith("lstm_") or a == b:
                continue
            if (a, b) not in pending_pairs:
                pending_pairs.append((a, b))
    n_compare = max(len(pending_pairs), 1)
    deltas: dict[tuple[str, str], dict] = {}
    for a, b in pending_pairs:
        deltas[(a, b)] = paired_bootstrap_delta_ci(
            cells, a, b, n_boot=args.n_boot,
            n_comparisons_for_bonferroni=n_compare,
        )

    # Evaluate D-21 against the preregistered "spiking" arch first.
    criteria_pre = evaluate_d21_criteria(stats, deltas,
                                          spiking_key="spiking",
                                          lstm_key="lstm",
                                          mamba_key="mamba")
    # If a post-hoc audit variant is present, evaluate it as well.
    # For each audit-spiking variant, also evaluate against the matched-budget
    # LSTM/Mamba baseline if available (lstm_25k, mamba_25k).
    audit_keys = sorted(k for k in stats if k.startswith("spiking_") and k != "spiking_t5")
    audit_criteria: dict[str, dict] = {}
    for k in audit_keys:
        # Decide which LSTM/Mamba baseline to match against based on the
        # variant's training-budget suffix. **Note**: every Tier-A/B
        # audit variant is trained at the 25k audit regime unless its
        # name explicitly carries a budget suffix (`_50k`, etc.). The
        # generic `_expand2` ablation falls into the 25k bucket.
        is_50k = "_50k" in k
        is_25k = (not is_50k) and (
            "_25k" in k
            or k.endswith("_lr5e4_25k")
            or "_t5sum" in k
            or "_lif_" in k
            or "_expand2" in k
        )
        if is_25k and "lstm_25k" in stats:
            audit_criteria[f"{k}_vs_5k_baselines"] = evaluate_d21_criteria(
                stats, deltas, spiking_key=k, lstm_key="lstm", mamba_key="mamba",
            )
            audit_criteria[f"{k}_vs_25k_baselines"] = evaluate_d21_criteria(
                stats, deltas, spiking_key=k, lstm_key="lstm_25k", mamba_key="mamba_25k",
            )
            for hw_label, hw_field in (
                ("gpu_dense", "energy_pJ_gpu_mean"),
                ("sparsity_aware", "energy_pJ_sparsity_mean"),
                ("neuromorphic", "energy_pJ_neuromorphic_mean"),
            ):
                audit_criteria[f"{k}_vs_25k_baselines_{hw_label}"] = evaluate_d21_criteria(
                    stats, deltas, spiking_key=k,
                    lstm_key="lstm_25k", mamba_key="mamba_25k",
                    energy_field=hw_field,
                )
            if "lstm_50k" in stats:
                audit_criteria[f"{k}_vs_50k_baselines"] = evaluate_d21_criteria(
                    stats, deltas, spiking_key=k, lstm_key="lstm_50k",
                    mamba_key=("mamba_50k" if "mamba_50k" in stats else "mamba_25k"),
                )
            if "lstm_100k" in stats:
                audit_criteria[f"{k}_vs_100k_baselines"] = evaluate_d21_criteria(
                    stats, deltas, spiking_key=k, lstm_key="lstm_100k",
                    mamba_key=("mamba_100k" if "mamba_100k" in stats else
                                ("mamba_50k" if "mamba_50k" in stats else "mamba_25k")),
                )
        elif is_50k and "lstm_50k" in stats:
            # Variant trained at 50k matched budget — compare against lstm_50k
            # / mamba_50k natively under all three hardware accountings.
            mamba_target = "mamba_50k" if "mamba_50k" in stats else "mamba_25k"
            audit_criteria[f"{k}_vs_50k_baselines"] = evaluate_d21_criteria(
                stats, deltas, spiking_key=k, lstm_key="lstm_50k", mamba_key=mamba_target,
            )
            for hw_label, hw_field in (
                ("gpu_dense", "energy_pJ_gpu_mean"),
                ("sparsity_aware", "energy_pJ_sparsity_mean"),
                ("neuromorphic", "energy_pJ_neuromorphic_mean"),
            ):
                audit_criteria[f"{k}_vs_50k_baselines_{hw_label}"] = evaluate_d21_criteria(
                    stats, deltas, spiking_key=k,
                    lstm_key="lstm_50k", mamba_key=mamba_target,
                    energy_field=hw_field,
                )
            # Cross-budget reference: same Spiking variant against the 25k
            # baseline (helpful to see if extending budget moves the gap).
            if "lstm_25k" in stats:
                audit_criteria[f"{k}_vs_25k_baselines"] = evaluate_d21_criteria(
                    stats, deltas, spiking_key=k, lstm_key="lstm_25k", mamba_key="mamba_25k",
                )
            # Tier A.1 (100k matched): if 100k baselines exist, also report
            # the cross-budget comparison. Same Spiking variant trained at
            # 50k vs LSTM/Mamba trained at 100k is unfair to LSTM/Mamba but
            # useful as a sensitivity diagnostic.
            if "lstm_100k" in stats:
                mamba_100k = "mamba_100k" if "mamba_100k" in stats else mamba_target
                audit_criteria[f"{k}_vs_100k_baselines"] = evaluate_d21_criteria(
                    stats, deltas, spiking_key=k, lstm_key="lstm_100k", mamba_key=mamba_100k,
                )
        else:
            audit_criteria[k] = evaluate_d21_criteria(stats, deltas, spiking_key=k)

    out_md = Path(args.out_md)
    out_json = Path(args.out_json)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_results_md(stats, deltas, criteria_pre, audit_criteria))
    out_json.write_text(json.dumps({
        "stats": stats,
        "deltas": {f"{a}_vs_{b}": d for (a, b), d in deltas.items()},
        "criteria_preregistered": criteria_pre,
        "criteria_audit": audit_criteria,
    }, indent=2))
    print(f"wrote {out_md} and {out_json}")
    print(f"decision (preregistered): {criteria_pre['decision']}")
    for k, c in audit_criteria.items():
        print(f"decision ({k}): {c['decision']}")


if __name__ == "__main__":
    main()
