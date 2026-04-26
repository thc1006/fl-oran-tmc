"""Aggregate Phase 2 v7 sweep cells into Stage 2 paper Table 4 + bootstrap CIs.

Reads each completed cell directory under ``--sweep-dir`` (default
``artifacts/v7_fl_arch_sweep``); writes:

* ``--out-md``    Stage 2 paper Markdown (default ``docs/RESULTS_V7_PHASE2.md``)
* ``--out-json``  machine-readable aggregated stats + paired-bootstrap deltas

Per ADR-001 D-22 + S2-W1..3, Phase 2 sweep cells vary 5 dimensions::

    arch ∈ {lstm, mamba, spiking_expand2}        (3 values, +ablations)
    algorithm ∈ {fedavg, fedprox, ...}           (6 in registry; MOON gated
                                                   to LSTM-only per D-22)
    partition_mode ∈ {iid, dirichlet}            (2 values)
    alpha ∈ {0.05, 0.1, 0.5, 1.0, 10.0}          (Dirichlet only)
    seed ∈ {42, 0, 1, 2, 3, 7, 11, 13, 17, 23}   (10 standardised seeds)

Aggregation contract:

* Group key = ``(arch, algorithm, partition_mode, alpha)``. Seeds collapse
  via mean / std AUC + n-seeds count. ``alpha`` is ``None`` for IID and
  preserved as such (not coerced to 0.0 / "n/a") so JSON consumers can
  distinguish unconditionally.
* Pairwise deltas are paired-bootstrap CI95 on the per-seed delta,
  mirroring ``aggregate_v6_results.paired_bootstrap_delta_ci`` so the
  statistics are directly comparable across phases.
* Stage 2 paper §5 narrative compares (a) FL algorithms within an
  architecture and (b) architectures within an FL algorithm. The
  aggregator emits both axis cuts in the JSON so downstream tooling can
  produce either Table 4 view without re-aggregating.

Defensive contract (lessons from the v6 round-4 audit):

* Single corrupt ``summary.json`` MUST NOT crash the whole pipeline:
  log + skip the cell.
* Cells without ``summary.json`` (e.g. partial run aborted mid-write)
  are silently dropped.
* Empty sweep directory raises ``RuntimeError`` rather than silently
  emitting an empty Table 4 — a zero-cell "successful" report is the
  worst kind of false positive.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as sps


# ---------------------------------------------------------------------------
# Re-use the canonical atomic-write helper via _v7_cell_metadata
# ---------------------------------------------------------------------------

_V7_HELPER_PATH = Path(__file__).resolve().parent / "_v7_cell_metadata.py"


def _load_v7_helper():
    spec = importlib.util.spec_from_file_location(
        "_v7_cell_metadata_for_aggregator", _V7_HELPER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load v7 helper from {_V7_HELPER_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Cell discovery + load
# ---------------------------------------------------------------------------

# Required summary.json fields. A cell missing any of these is malformed
# and gets skipped with a warning.
_REQUIRED_SUMMARY_FIELDS = (
    "arch", "algorithm", "partition_mode", "seed", "test_auc",
)


def _warn(msg: str) -> None:
    """Single point of truth for warnings so tests can capture them."""
    print(f"warning: {msg}")


def load_cells(sweep_dir: Path) -> dict:
    """Discover every cell directory under ``sweep_dir``; load summaries.

    Returns a dict keyed by
    ``(arch, algorithm, partition_mode, alpha, seed)`` whose values are
    the parsed ``summary.json`` contents (with ``arch`` / ``algorithm`` /
    etc. preserved as written).

    Discovery: any subdirectory containing ``summary.json``. The cell
    name is irrelevant — we trust the JSON, not the path.

    Defensive:
      * malformed JSON → log and skip
      * missing required field → log and skip
      * non-numeric ``test_auc`` → log and skip
      * dir without ``summary.json`` → silently skip (probably a
        partial run aborted before write)
    """
    sweep_dir = Path(sweep_dir)
    cells: dict = {}
    if not sweep_dir.is_dir():
        return cells
    for cell_dir in sorted(p for p in sweep_dir.iterdir() if p.is_dir()):
        summary_path = cell_dir / "summary.json"
        if not summary_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            _warn(f"{cell_dir.name} — summary.json unreadable ({exc}); skipping")
            continue
        if not isinstance(summary, dict):
            _warn(f"{cell_dir.name} — summary.json is not a JSON object; skipping")
            continue
        missing = [f for f in _REQUIRED_SUMMARY_FIELDS if f not in summary]
        if missing:
            _warn(
                f"{cell_dir.name} — summary.json missing fields {missing}; skipping"
            )
            continue
        try:
            arch = str(summary["arch"])
            algorithm = str(summary["algorithm"])
            partition_mode = str(summary["partition_mode"])
            seed = int(summary["seed"])
            # alpha is optional and may be None for IID. Cast to float
            # only if present and non-None.
            alpha_raw = summary.get("alpha")
            alpha = None if alpha_raw is None else float(alpha_raw)
            test_auc = float(summary["test_auc"])
        except (TypeError, ValueError) as exc:
            _warn(
                f"{cell_dir.name} — summary.json has invalid field types ({exc}); "
                "skipping"
            )
            continue
        # Normalise back into the dict so per_group_stats can rely on
        # field types being correct.
        summary["test_auc"] = test_auc
        summary["seed"] = seed
        summary["alpha"] = alpha
        key = (arch, algorithm, partition_mode, alpha, seed)
        if key in cells:
            # Two cell dirs with identical metadata (re-run with a
            # different output dir, or a renamed copy) — last-write-wins
            # would silently drop the previous AUC. Warn so the user
            # decides whether to clean up or keep both.
            _warn(
                f"{cell_dir.name} — duplicate metadata key {key}; "
                "previous cell's AUC will be overwritten"
            )
        cells[key] = summary
    return cells


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def per_group_stats(cells: dict) -> dict:
    """Aggregate seeds within each ``(arch, algo, partition, alpha)``.

    Output keyed by the 4-tuple group key; values include n, mean / std
    AUC, mean / std F1 / accuracy (if present in summary), mean
    params_count, and the sorted list of seeds for traceability.
    """
    by_group: dict = defaultdict(list)
    for (arch, algo, pmode, alpha, _seed), summary in cells.items():
        by_group[(arch, algo, pmode, alpha)].append(summary)

    out: dict = {}
    for key, items in by_group.items():
        aucs = np.array([float(v["test_auc"]) for v in items])
        f1s = np.array(
            [float(v.get("test_f1", float("nan"))) for v in items], dtype=float,
        )
        accs = np.array(
            [float(v.get("test_accuracy", float("nan"))) for v in items],
            dtype=float,
        )
        params = np.array(
            [int(v.get("params_count", 0)) for v in items], dtype=int,
        )
        seeds = sorted(int(v["seed"]) for v in items)
        out[key] = {
            "arch": key[0],
            "algorithm": key[1],
            "partition_mode": key[2],
            "alpha": key[3],
            "n": int(len(items)),
            "test_auc_mean": float(aucs.mean()),
            "test_auc_std": float(aucs.std(ddof=1)) if len(aucs) > 1 else None,
            "test_f1_mean": _nanmean(f1s),
            "test_f1_std": _nanstd(f1s),
            "test_accuracy_mean": _nanmean(accs),
            "params_count_mean": float(params.mean()) if len(params) else 0.0,
            "seeds": seeds,
        }
    return out


def _nanmean(arr: np.ndarray) -> float | None:
    """Mean ignoring NaN; returns None when all-NaN so that downstream
    can distinguish "no data" from "value happens to be 0.0" (which the
    earlier 0.0-fallback would silently conflate). JSON serialises None
    as ``null`` — still serialisable."""
    if arr.size == 0 or np.all(np.isnan(arr)):
        return None
    return float(np.nanmean(arr))


def _nanstd(arr: np.ndarray) -> float | None:
    """Sample std ignoring NaN; ``None`` when fewer than 2 valid entries
    (mirrors :func:`_nanmean` so the JSON ``null`` distinguishes "n<2"
    from a genuine std of 0)."""
    valid = arr[~np.isnan(arr)]
    if valid.size < 2:
        return None
    return float(valid.std(ddof=1))


# ---------------------------------------------------------------------------
# Paired-bootstrap pairwise delta
# ---------------------------------------------------------------------------

def _select_seed_aucs(cells: dict, axes: dict) -> dict[int, float]:
    """Return ``{seed: test_auc}`` for cells matching every (k, v) in
    ``axes`` (strict equality on each field). Used to build paired AUC
    arrays for bootstrap deltas."""
    out: dict[int, float] = {}
    for (arch, algo, pmode, alpha, seed), summary in cells.items():
        if axes.get("arch", arch) != arch:
            continue
        if axes.get("algorithm", algo) != algo:
            continue
        if axes.get("partition_mode", pmode) != pmode:
            continue
        # alpha may be None (IID); equality on None is well-defined.
        if "alpha" in axes and axes["alpha"] != alpha:
            continue
        out[seed] = float(summary["test_auc"])
    return out


def paired_bootstrap_delta(cells: dict, *, a: dict, b: dict,
                           n_boot: int = 10_000, ci_level: float = 0.95,
                           seed: int = 2026) -> dict:
    """delta_auc(a, b) via paired bootstrap on per-seed AUC pairs.

    ``a`` and ``b`` are dicts of axis filters (e.g.
    ``{"arch": "mamba", "algorithm": "fedavg", "partition_mode":
    "dirichlet", "alpha": 0.5}``). Only seeds present under BOTH
    filter combinations contribute. CI fields are ``None`` if fewer than
    2 paired seeds — bootstrap on n=1 is meaningless and would emit
    misleading point estimates.
    """
    aucs_a = _select_seed_aucs(cells, a)
    aucs_b = _select_seed_aucs(cells, b)
    common = sorted(set(aucs_a) & set(aucs_b))
    deltas = np.array([aucs_a[s] - aucs_b[s] for s in common], dtype=float)
    n = len(deltas)
    if n < 2:
        return {
            "n_paired_seeds": int(n),
            "delta_mean": float(deltas.mean()) if n else 0.0,
            "ci_lo": None,
            "ci_hi": None,
            "wilcoxon_p": None,
            "seeds": common,
        }
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boot_means[i] = rng.choice(deltas, size=n, replace=True).mean()
    alpha = 1.0 - ci_level
    ci_lo = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    try:
        wilcoxon_p = float(
            sps.wilcoxon(deltas, alternative="two-sided",
                         zero_method="wilcox").pvalue
        )
    except (ValueError, ZeroDivisionError):
        wilcoxon_p = None
    return {
        "n_paired_seeds": int(n),
        "delta_mean": float(deltas.mean()),
        "delta_std": float(deltas.std(ddof=1)),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "wilcoxon_p": wilcoxon_p,
        "seeds": common,
    }


def all_pairwise_algo_deltas(cells: dict, *, n_boot: int = 10_000) -> dict:
    """For every ``(arch, partition_mode, alpha)`` cell, compute the
    pairwise delta between every algorithm pair. Used for Stage 2 §5
    "FL algorithm comparison within architecture" subtable.

    Returns dict keyed by ``(arch, pmode, alpha, algo_a, algo_b)``.
    """
    out: dict = {}
    by_group: dict = defaultdict(set)
    for (arch, algo, pmode, alpha, _seed) in cells:
        by_group[(arch, pmode, alpha)].add(algo)
    for (arch, pmode, alpha), algos in by_group.items():
        algos_sorted = sorted(algos)
        for i, algo_a in enumerate(algos_sorted):
            for algo_b in algos_sorted[i + 1:]:
                out[(arch, pmode, alpha, algo_a, algo_b)] = paired_bootstrap_delta(
                    cells,
                    a={"arch": arch, "algorithm": algo_a,
                       "partition_mode": pmode, "alpha": alpha},
                    b={"arch": arch, "algorithm": algo_b,
                       "partition_mode": pmode, "alpha": alpha},
                    n_boot=n_boot,
                )
    return out


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _alpha_str(alpha) -> str:
    return "n/a (IID)" if alpha is None else f"{alpha:.2f}"


def render_results_md(stats: dict, deltas: dict) -> str:
    """Render the Stage 2 §5 paper-grade Markdown summary.

    Layout:
      1. Table 4 (per-(algo, arch, partition, alpha) mean ± std AUC)
      2. Pairwise FL-algorithm deltas within each arch + partition cell
    """
    lines: list[str] = []
    lines.append("# Stage 2 Phase 2 Results — FL × Architecture Sweep on ColO-RAN\n")
    lines.append(
        "Generated by `scripts/aggregate_v7_results.py`. See ADR-001 D-22 + "
        "Phase 2 minimum scope.\n"
    )

    if not stats:
        lines.append("> _No cells aggregated yet — run the Phase 2 sweep first._\n")
        return "\n".join(lines)

    lines.append("## Table 4: per-cell aggregated statistics\n")
    lines.append(
        "| arch | algorithm | partition | alpha | n_seeds | test AUC (mean ± std) "
        "| test F1 | params |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    # Stable sort: arch, then algorithm, then partition, then alpha
    # (None first so IID rows come before Dirichlet rows for the same arch+algo).
    def _sort_key(item):
        key, _v = item
        arch, algo, pmode, alpha = key
        return (arch, algo, pmode, -1.0 if alpha is None else alpha)
    for key, s in sorted(stats.items(), key=_sort_key):
        arch, algo, pmode, alpha = key
        f1_mean = s.get("test_f1_mean")
        f1_cell = "n/a" if f1_mean is None else f"{f1_mean:.4f}"
        std_val = s["test_auc_std"]
        std_str = "n/a" if std_val is None else f"{std_val:.4f}"
        lines.append(
            f"| {arch} | {algo} | {pmode} | {_alpha_str(alpha)} | {s['n']} | "
            f"{s['test_auc_mean']:.4f} ± {std_str} | "
            f"{f1_cell} | {int(s['params_count_mean'])} |"
        )
    lines.append("")

    if deltas:
        lines.append(
            "## Pairwise FL-algorithm deltas (within arch + partition cell)\n"
        )
        lines.append(
            "Paired-bootstrap CI95 on per-seed AUC delta. "
            "n_boot reported per cell.\n"
        )
        lines.append(
            "| arch | partition | alpha | comparison | n | delta mean | "
            "CI95 [lo, hi] | Wilcoxon p |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for key, d in sorted(deltas.items()):
            arch, pmode, alpha, algo_a, algo_b = key
            if d.get("ci_lo") is None:
                ci = "n/a (n<2)"
            else:
                ci = f"[{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]"
            wp = "n/a" if d.get("wilcoxon_p") is None else f"{d['wilcoxon_p']:.4f}"
            lines.append(
                f"| {arch} | {pmode} | {_alpha_str(alpha)} | "
                f"{algo_a} − {algo_b} | {d['n_paired_seeds']} | "
                f"{d['delta_mean']:+.4f} | {ci} | {wp} |"
            )
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON serialisation helpers (4-tuple keys → "::"-joined strings)
# ---------------------------------------------------------------------------

def _alpha_jsonkey(alpha) -> str:
    """Encode alpha as a JSON-key fragment using the canonical 2-decimal
    'p'-separator convention (mirrors ``_v7_cell_metadata.cell_name``).
    Keeps JSON keys lexically alignable with on-disk cell directory names."""
    return "iid" if alpha is None else "a" + f"{alpha:.2f}".replace(".", "p")


def _stats_to_jsonable(stats: dict) -> dict:
    """Group keys are tuples — JSON requires str. Join with '::' separator."""
    return {
        "::".join([arch, algo, pmode, _alpha_jsonkey(alpha)]): v
        for (arch, algo, pmode, alpha), v in stats.items()
    }


def _deltas_to_jsonable(deltas: dict) -> dict:
    return {
        "::".join([arch, pmode, _alpha_jsonkey(alpha), algo_a, "vs", algo_b]): v
        for (arch, pmode, alpha, algo_a, algo_b), v in deltas.items()
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-dir", type=str, default="artifacts/v7_fl_arch_sweep",
        help="Directory containing per-cell subdirectories.",
    )
    parser.add_argument(
        "--out-md", type=str, default="docs/RESULTS_V7_PHASE2.md",
        help="Where to write the paper-grade Markdown.",
    )
    parser.add_argument(
        "--out-json", type=str,
        default="artifacts/v7_fl_arch_sweep/aggregated.json",
        help="Where to write machine-readable aggregated stats.",
    )
    parser.add_argument(
        "--n-boot", type=int, default=10_000,
        help="Bootstrap resample count for paired-delta CI.",
    )
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    cells = load_cells(sweep_dir)
    if not cells:
        raise RuntimeError(
            f"No v7 cells found under {sweep_dir}. Refusing to emit "
            "a zero-cell Table 4 — verify the sweep ran and produced "
            "summary.json files."
        )

    stats = per_group_stats(cells)
    deltas = all_pairwise_algo_deltas(cells, n_boot=args.n_boot)

    helper = _load_v7_helper()
    out_md = Path(args.out_md)
    out_json = Path(args.out_json)
    helper.atomic_write_text(out_md, render_results_md(stats, deltas))
    helper.atomic_write_text(out_json, json.dumps({
        "stats": _stats_to_jsonable(stats),
        "deltas": _deltas_to_jsonable(deltas),
        "n_cells": len(cells),
    }, indent=2))
    print(f"wrote {out_md} and {out_json}")
    print(f"aggregated {len(cells)} cells into {len(stats)} groups, "
          f"{len(deltas)} pairwise deltas")


if __name__ == "__main__":
    main()
