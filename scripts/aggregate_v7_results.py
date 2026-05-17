"""Aggregate Phase 2 v7 sweep cells into Stage 2 paper Table 4 + bootstrap CIs.

Reads each completed cell directory under ``--sweep-dir`` (default
``artifacts/v7_fl_arch_sweep``); writes:

* ``--out-md``    Stage 2 paper Markdown (default ``docs/RESULTS_V7_PHASE2.md``)
* ``--out-json``  machine-readable aggregated stats + paired-bootstrap deltas

Sweep dimensions actually emitted by fl_v7 (Phase 5 + Phase 6 ablations)::

    arch ∈ {lstm, mamba, spiking_expand2,         (5 FL-phase archs; Phase 5
            xlstm, mamba3}                          + Phase 6 used the first 3,
                                                   Path D extended sweep
                                                   (2026-05-18) added xlstm and
                                                   mamba3 — see
                                                   ``docs/PAPER_NOTES_XLSTM.md``
                                                   and ``docs/PAPER_NOTES_MAMBA3.md``
                                                   for the per-arch design
                                                   rationale. ARCH_REGISTRY also
                                                   exposes ``mamba_expand2`` and
                                                   ``spiking`` for Stage 1
                                                   centralised ablations only,
                                                   which no FL sweep uses.)
    algorithm ∈ {fedavg, fedprox, fedadam,        (5 values; MOON deferred
                 scaffold, feddyn}                  per ADR-001 D-22)
    partition_mode ∈ {iid, dirichlet,             (4 values: Phase 5 used
                      random_split,                 iid + dirichlet; Phase 6
                      per_bs_dirichlet}             added random_split (T-ABLATION)
                                                   and per_bs_dirichlet
                                                   (Rank 3 mechanism
                                                   disambiguation, §7.1.1).
                                                   IID uses natural-by-BS
                                                   over 7 ColO-RAN gNBs.
    alpha ∈ {0.05, 0.10, 0.50, 1.00, 5.00, 10.00} (Dirichlet & per_bs_dirichlet
                                                   only; alpha is FORCED to
                                                   None for IID & random_split
                                                   regardless of the V7Config
                                                   default 0.5)
    seed ∈ {42, 0, 1, 2, 3, ...}                 (5–10 seeds per cell
                                                   depending on phase scope)

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
import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as sps


def _derive_pair_seed(base_seed: int, pair_id: str) -> int:
    """Hash a pair identifier into an unsigned 32-bit seed offset added
    to ``base_seed``. Uses BLAKE2b for stability across Python versions
    (unlike ``hash()`` which respects PYTHONHASHSEED). Result fits in
    numpy's accepted seed range and is deterministic for a given
    (base_seed, pair_id)."""
    digest = hashlib.blake2b(pair_id.encode("utf-8"), digest_size=4).digest()
    offset = int.from_bytes(digest, byteorder="big", signed=False)
    return (base_seed + offset) % (2 ** 32)


# ---------------------------------------------------------------------------
# Re-use the canonical atomic-write helper via _v7_cell_metadata
# ---------------------------------------------------------------------------

_V7_HELPER_PATH = Path(__file__).resolve().parent / "_v7_cell_metadata.py"
_V7_HELPER_MODULE_KEY = "_v7_cell_metadata"


def _load_v7_helper():
    """Load _v7_cell_metadata once per process.

    Uses the canonical module key + ``sys.modules.setdefault`` so that
    other tools loading the same helper (e.g. ``_v7_spec_loader``) share
    one instance — otherwise each loader would re-import v6's runner
    module (transitively pulling in torch) and the per-module
    ``_RUNNER_CACHE`` would be wasted.
    """
    cached = sys.modules.get(_V7_HELPER_MODULE_KEY)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(
        _V7_HELPER_MODULE_KEY, _V7_HELPER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load v7 helper from {_V7_HELPER_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_V7_HELPER_MODULE_KEY] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Cell discovery + load
# ---------------------------------------------------------------------------

# Required summary fields, by location inside the cell summary.
#
# fl_v7 writes config metadata (arch / algorithm / partition_mode / seed /
# alpha) under ``summary["config"]`` and per-task metrics (auc / accuracy /
# f1) under ``summary["test"]``. Earlier drafts of this aggregator looked
# for these at the top level and skipped every cell — a 900-cell sweep was
# silently dropped to zero. The schema below mirrors what
# ``src/fl_oran/training/fl_v7.py`` actually emits.
_REQUIRED_CONFIG_FIELDS = ("arch", "algorithm", "partition_mode", "seed")
_REQUIRED_TEST_FIELDS = ("auc",)


# Hard-coded per-arch parameter counts. Source: docs/RESULTS_V6_STAGE1_ANALYSIS.md
# (Stage 1 audit) cross-checked against the v7 build_model pin tests in
# ``tests/test_v7_fl_arch_agnostic.py`` (one per arch with name
# ``test_v7_build_model_*_pins_params_*``). All 5 values below are
# pin-tested against the V3 schema (V3_CATEGORICAL = [bs_id, slice_id,
# sched, tr] with sizes {8, 4, 4, 29}; V3_CONTINUOUS has 17 features).
# Path D extension (2026-05-18) added xLSTM and Mamba-3 with their own
# pin tests:
#   xlstm:  43241 params (xLSTMForecaster, hidden_size=48, n_layers=2)
#   mamba3: 40635 params (Mamba3Forecaster, d_model=64, n_blocks=2,
#                         d_state=16 → 8 complex pairs)
# Schema drift in V3_CAT_SIZES will fail ALL 5 pin tests loudly. The
# other two entries in ARCH_REGISTRY — ``mamba_expand2`` and
# ``spiking`` — are Stage 1 centralised ablation archs that no FL sweep
# cell uses, so they're intentionally absent here. ``.get(arch)`` returns
# ``None`` for any missing arch, which renders as "n/a" in the Markdown
# table rather than raising — keeps the aggregator forward-compatible.
_ARCH_PARAMS_COUNT: dict[str, int] = {
    "lstm": 44553,
    "mamba": 40489,
    "spiking_expand2": 43593,
    "xlstm": 43241,
    "mamba3": 40635,
}


def _warn(msg: str) -> None:
    """Single point of truth for warnings so tests can capture them.

    Written to stderr so CI/cron logs cleanly separate warnings from the
    final ``wrote ...`` success line on stdout.
    """
    print(f"warning: {msg}", file=sys.stderr)


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
        cfg = summary.get("config")
        test_block = summary.get("test")
        if not isinstance(cfg, dict):
            _warn(f"{cell_dir.name} — summary.config missing or not a dict; skipping")
            continue
        if not isinstance(test_block, dict):
            _warn(f"{cell_dir.name} — summary.test missing or not a dict; skipping")
            continue
        missing_cfg = [f for f in _REQUIRED_CONFIG_FIELDS if f not in cfg]
        missing_test = [f for f in _REQUIRED_TEST_FIELDS if f not in test_block]
        if missing_cfg or missing_test:
            _warn(
                f"{cell_dir.name} — missing required fields "
                f"config={missing_cfg} test={missing_test}; skipping"
            )
            continue
        try:
            arch = str(cfg["arch"])
            algorithm = str(cfg["algorithm"])
            partition_mode = str(cfg["partition_mode"])
            seed = int(cfg["seed"])
            # IID and random_split ignore alpha by definition; fl_v7's
            # V7Config leaves the default 0.5 in place for those cells,
            # so we MUST coerce alpha → None here (overriding whatever
            # the config emitted). Otherwise alpha-free cells would
            # aggregate into a spurious (arch, algo, "iid", 0.5) group
            # instead of (arch, algo, "iid", None). dirichlet and
            # per_bs_dirichlet keep the configured alpha.
            if partition_mode in {"iid", "random_split"}:
                alpha = None
            else:
                alpha_raw = cfg.get("alpha")
                alpha = None if alpha_raw is None else float(alpha_raw)
            test_auc = float(test_block["auc"])
        except (TypeError, ValueError) as exc:
            _warn(
                f"{cell_dir.name} — invalid field types ({exc}); skipping"
            )
            continue
        # AUC must be finite (mathematically AUC ∈ [0, 1]; NaN/Inf
        # indicates a single-class test split or a broken metric and
        # would silently propagate as 'NaN' in the output JSON).
        if not np.isfinite(test_auc):
            _warn(
                f"{cell_dir.name} — test.auc={test_auc} is not finite; skipping"
            )
            continue
        # Promote nested fields to a flat normalised view so per_group_stats
        # can read them without re-walking the schema. Mutates the loaded
        # summary in place; that's fine because it's a freshly-parsed copy.
        summary["test_auc"] = test_auc
        summary["test_f1"] = float(test_block["f1"]) if "f1" in test_block else None
        summary["test_accuracy"] = (
            float(test_block["accuracy"]) if "accuracy" in test_block else None
        )
        summary["seed"] = seed
        summary["alpha"] = alpha
        summary["arch"] = arch
        summary["algorithm"] = algorithm
        summary["partition_mode"] = partition_mode
        # Energy: training_model_attributable_mJ is the paper §6 number
        # (NVML idle-baseline subtracted). Other keys are kept on the
        # cell summary; we only surface the headline figure here.
        em = summary.get("energy_measured")
        if isinstance(em, dict):
            mj = em.get("training_model_attributable_mJ")
            summary["energy_model_mJ"] = float(mj) if mj is not None else None
        else:
            summary["energy_model_mJ"] = None
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

    def _to_float_or_nan(x) -> float:
        return float(x) if x is not None else float("nan")

    out: dict = {}
    for key, items in by_group.items():
        aucs = np.array([float(v["test_auc"]) for v in items])
        f1s = np.array([_to_float_or_nan(v.get("test_f1")) for v in items], dtype=float)
        accs = np.array(
            [_to_float_or_nan(v.get("test_accuracy")) for v in items], dtype=float,
        )
        energies = np.array(
            [_to_float_or_nan(v.get("energy_model_mJ")) for v in items], dtype=float,
        )
        seeds = sorted(int(v["seed"]) for v in items)
        params_count = _ARCH_PARAMS_COUNT.get(key[0])
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
            "energy_model_mJ_mean": _nanmean(energies),
            "energy_model_mJ_std": _nanstd(energies),
            "params_count": params_count,
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
                           seed: int = 2026,
                           bonferroni_n: int | None = None) -> dict:
    """delta_auc(a, b) via paired bootstrap on per-seed AUC pairs.

    ``a`` and ``b`` are dicts of axis filters (e.g.
    ``{"arch": "mamba", "algorithm": "fedavg", "partition_mode":
    "dirichlet", "alpha": 0.5}``). Only seeds present under BOTH
    filter combinations contribute. CI fields are ``None`` if fewer than
    2 paired seeds — bootstrap on n=1 is meaningless and would emit
    misleading point estimates.

    When ``bonferroni_n`` is set (typically the family size — total
    number of pairwise comparisons in the same sweep), the dict also
    includes ``ci_lo_bonferroni`` / ``ci_hi_bonferroni`` computed at
    the family-corrected level ``1 - (1 - ci_level) / bonferroni_n``.
    Adjusted keys are always present (``None`` when not requested or
    when n<2 paired seeds) so downstream JSON consumers do not KeyError.
    """
    aucs_a = _select_seed_aucs(cells, a)
    aucs_b = _select_seed_aucs(cells, b)
    common = sorted(set(aucs_a) & set(aucs_b))
    deltas = np.array([aucs_a[s] - aucs_b[s] for s in common], dtype=float)
    n = len(deltas)
    if n < 3:
        # GH#3: n=1 gives zero-width CI (every bootstrap sample is the
        # same observation); n=2 gives a 3-point CI bracket. Both are
        # degenerate and would mislead readers into interpreting them
        # as tight estimates. Refuse and emit an explicit warning.
        return {
            "n_paired_seeds": int(n),
            "delta_mean": float(deltas.mean()) if n else 0.0,
            "delta_std": None,
            "ci_lo": None,
            "ci_hi": None,
            "ci_lo_bonferroni": None,
            "ci_hi_bonferroni": None,
            "wilcoxon_p": None,
            "warning": f"bootstrap CI requires n>=3 paired seeds, got n={n}",
            "seeds": common,
        }
    rng = np.random.default_rng(seed)
    # Vectorised resample: draw an (n_boot, n) index matrix in one call,
    # gather, then mean along the inner axis. ~10× faster than the
    # per-bootstrap rng.choice loop on n_boot=10_000 + 100+ pairs.
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = deltas[idx].mean(axis=1)
    alpha = 1.0 - ci_level
    ci_lo = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    if bonferroni_n is not None and bonferroni_n > 0:
        alpha_adj = alpha / bonferroni_n
        ci_lo_b = float(np.percentile(boot_means, 100 * alpha_adj / 2))
        ci_hi_b = float(np.percentile(boot_means, 100 * (1 - alpha_adj / 2)))
    else:
        ci_lo_b = None
        ci_hi_b = None
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
        "ci_lo_bonferroni": ci_lo_b,
        "ci_hi_bonferroni": ci_hi_b,
        "wilcoxon_p": wilcoxon_p,
        "warning": None,
        "seeds": common,
    }


def all_pairwise_algo_deltas(cells: dict, *, n_boot: int = 10_000,
                             base_seed: int = 2026) -> dict:
    """For every ``(arch, partition_mode, alpha)`` cell, compute the
    pairwise delta between every algorithm pair. Used for Stage 2 §5
    "FL algorithm comparison within architecture" subtable.

    Each pair gets its own RNG seed derived from a stable hash of the
    pair identifier (BLAKE2b). Without this, every pair would resample
    the bootstrap distribution at exactly the same indices, producing
    correlated CIs — fine for "report each CI independently" but bad
    for any joint coverage claim.

    Returns dict keyed by ``(arch, pmode, alpha, algo_a, algo_b)``.
    """
    out: dict = {}
    by_group: dict = defaultdict(set)
    for (arch, algo, pmode, alpha, _seed) in cells:
        by_group[(arch, pmode, alpha)].add(algo)
    total_pairs = sum(len(algos) * (len(algos) - 1) // 2
                      for algos in by_group.values())
    for (arch, pmode, alpha), algos in by_group.items():
        algos_sorted = sorted(algos)
        for i, algo_a in enumerate(algos_sorted):
            for algo_b in algos_sorted[i + 1:]:
                pair_id = f"{arch}::{pmode}::{alpha}::{algo_a}::vs::{algo_b}"
                pair_seed = _derive_pair_seed(base_seed, pair_id)
                out[(arch, pmode, alpha, algo_a, algo_b)] = paired_bootstrap_delta(
                    cells,
                    a={"arch": arch, "algorithm": algo_a,
                       "partition_mode": pmode, "alpha": alpha},
                    b={"arch": arch, "algorithm": algo_b,
                       "partition_mode": pmode, "alpha": alpha},
                    n_boot=n_boot, seed=pair_seed,
                    bonferroni_n=total_pairs,
                )
    return out


def all_pairwise_arch_deltas(cells: dict, *, n_boot: int = 10_000,
                             base_seed: int = 2027) -> dict:
    """Mirror of :func:`all_pairwise_algo_deltas` but holding ``algorithm``
    and ``partition_mode`` + ``alpha`` fixed and pairing across
    architectures. This answers the §1 contribution-4 question
    ("architecture leverage dominates algorithm leverage") with a
    proper paired-bootstrap CI95 instead of a group-mean comparison
    (which would conflate seed noise into the architecture spread).

    Uses ``base_seed=2027`` (different from the algo-pair sweep's 2026)
    so the two pair sweeps' bootstrap streams are independent — joint
    coverage claims across the two pair sets remain valid.

    Returns dict keyed by ``(algorithm, pmode, alpha, arch_a, arch_b)``.
    """
    out: dict = {}
    by_group: dict = defaultdict(set)
    for (arch, algo, pmode, alpha, _seed) in cells:
        by_group[(algo, pmode, alpha)].add(arch)
    total_pairs = sum(len(archs) * (len(archs) - 1) // 2
                      for archs in by_group.values())
    for (algo, pmode, alpha), archs in by_group.items():
        archs_sorted = sorted(archs)
        for i, arch_a in enumerate(archs_sorted):
            for arch_b in archs_sorted[i + 1:]:
                pair_id = f"{algo}::{pmode}::{alpha}::{arch_a}::vs::{arch_b}"
                pair_seed = _derive_pair_seed(base_seed, pair_id)
                out[(algo, pmode, alpha, arch_a, arch_b)] = paired_bootstrap_delta(
                    cells,
                    a={"arch": arch_a, "algorithm": algo,
                       "partition_mode": pmode, "alpha": alpha},
                    b={"arch": arch_b, "algorithm": algo,
                       "partition_mode": pmode, "alpha": alpha},
                    n_boot=n_boot, seed=pair_seed,
                    bonferroni_n=total_pairs,
                )
    return out


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _alpha_str(alpha) -> str:
    return "n/a (IID)" if alpha is None else f"{alpha:.2f}"


def render_results_md(stats: dict, deltas: dict,
                      arch_deltas: dict | None = None) -> str:
    """Render the Stage 2 §5 paper-grade Markdown summary.

    Layout:
      1. Table 4 (per-(algo, arch, partition, alpha) mean ± std AUC)
      2. Pairwise FL-algorithm deltas within each arch + partition cell
      3. Pairwise architecture deltas within each algo + partition cell
         (only emitted if ``arch_deltas`` is supplied — kept optional so
         legacy callers that only pass two args still work).
    """
    lines: list[str] = []
    lines.append("# Stage 2 Results — FL × Architecture Sweep on ColO-RAN\n")
    lines.append(
        "Generated by `scripts/aggregate_v7_results.py`. The sweep scope "
        "(Phase 2 minimum / Phase 5 full / ablation) is determined by the "
        "``--sweep-dir`` argument; this report aggregates whatever cells "
        "are present in that directory.\n"
    )

    if not stats:
        lines.append("> _No cells aggregated yet — run the Phase 2 sweep first._\n")
        return "\n".join(lines)

    lines.append("## Table 4: per-cell aggregated statistics\n")
    lines.append(
        "| arch | algorithm | partition | alpha | n_seeds | test AUC (mean ± std) "
        "| test F1 | model energy (J, mean ± std) | params |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
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
        e_mean = s.get("energy_model_mJ_mean")
        e_std = s.get("energy_model_mJ_std")
        if e_mean is None:
            energy_cell = "n/a"
        else:
            e_std_str = "n/a" if e_std is None else f"{e_std / 1000.0:.2f}"
            energy_cell = f"{e_mean / 1000.0:.2f} ± {e_std_str}"
        params = s.get("params_count")
        params_cell = "n/a" if params is None else f"{int(params)}"
        lines.append(
            f"| {arch} | {algo} | {pmode} | {_alpha_str(alpha)} | {s['n']} | "
            f"{s['test_auc_mean']:.4f} ± {std_str} | "
            f"{f1_cell} | {energy_cell} | {params_cell} |"
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

    if arch_deltas:
        lines.append(
            "## Pairwise architecture deltas (within algo + partition cell)\n"
        )
        lines.append(
            "Paired-bootstrap CI95 on per-seed AUC delta, holding "
            "(algorithm, partition, alpha) fixed. Substantiates §1 "
            "contribution 4 (architecture leverage).\n"
        )
        lines.append(
            "| algorithm | partition | alpha | comparison | n | delta mean | "
            "CI95 [lo, hi] | Wilcoxon p |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for key, d in sorted(arch_deltas.items()):
            algo, pmode, alpha, arch_a, arch_b = key
            if d.get("ci_lo") is None:
                ci = "n/a (n<2)"
            else:
                ci = f"[{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]"
            wp = "n/a" if d.get("wilcoxon_p") is None else f"{d['wilcoxon_p']:.4f}"
            lines.append(
                f"| {algo} | {pmode} | {_alpha_str(alpha)} | "
                f"{arch_a} − {arch_b} | {d['n_paired_seeds']} | "
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


def _arch_deltas_to_jsonable(arch_deltas: dict) -> dict:
    """Mirror of :func:`_deltas_to_jsonable` for arch-pair entries.
    Key order is ``algorithm::partition::alpha::arch_a::vs::arch_b`` —
    distinguishable from algo-pair keys by the leading token (algo names
    do not collide with arch names in our registries)."""
    return {
        "::".join([algo, pmode, _alpha_jsonkey(alpha), arch_a, "vs", arch_b]): v
        for (algo, pmode, alpha, arch_a, arch_b), v in arch_deltas.items()
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
        "--out-json", type=str, default=None,
        help="Where to write machine-readable aggregated stats. "
             "Defaults to ``<sweep_dir>/aggregated.json`` so a non-default "
             "--sweep-dir does not silently land its JSON in the legacy "
             "v7_fl_arch_sweep/ directory.",
    )
    parser.add_argument(
        "--n-boot", type=int, default=10_000,
        help="Bootstrap resample count for paired-delta CI.",
    )
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    out_json_path = (
        Path(args.out_json) if args.out_json is not None
        else sweep_dir / "aggregated.json"
    )
    cells = load_cells(sweep_dir)
    if not cells:
        raise RuntimeError(
            f"No v7 cells found under {sweep_dir}. Refusing to emit "
            "a zero-cell Table 4 — verify the sweep ran and produced "
            "summary.json files."
        )

    stats = per_group_stats(cells)
    deltas = all_pairwise_algo_deltas(cells, n_boot=args.n_boot)
    arch_deltas = all_pairwise_arch_deltas(cells, n_boot=args.n_boot)

    helper = _load_v7_helper()
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    helper.atomic_write_text(
        out_md, render_results_md(stats, deltas, arch_deltas=arch_deltas),
    )
    helper.atomic_write_text(out_json_path, json.dumps({
        "stats": _stats_to_jsonable(stats),
        "deltas": _deltas_to_jsonable(deltas),
        "arch_deltas": _arch_deltas_to_jsonable(arch_deltas),
        "n_cells": len(cells),
    }, indent=2))
    print(f"wrote {out_md} and {out_json_path}")
    print(f"aggregated {len(cells)} cells into {len(stats)} groups, "
          f"{len(deltas)} algo-pair deltas, {len(arch_deltas)} arch-pair deltas")


if __name__ == "__main__":
    main()
