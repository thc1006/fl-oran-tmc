"""Tests for scripts/aggregate_v7_results.py — Stage 2 paper Table 4 emitter.

The v7 aggregator reads completed Phase 2 sweep cells (one directory per
``(arch, algorithm, partition_mode, alpha, seed)`` combination) and
produces both the paper-grade Markdown table for Stage 2 §5 and a
machine-readable JSON for downstream tooling.

Design contract (driven by ADR-001 D-22):

* The aggregator does NOT depend on parsing cell directory names.
  fl_v7's ``summary.json`` carries explicit ``arch``, ``algorithm``,
  ``partition_mode``, ``alpha``, ``seed`` fields. Parsing names would
  be redundant with the JSON and brittle if a non-canonical name is
  used by a one-off cell. We open the JSON and trust its contents.
* Per-(arch, algorithm, partition_mode, alpha) cells aggregate via mean
  ± std AUC across seeds, plus an explicit n-seeds count.
* Pairwise FL-algorithm deltas (e.g. FedProx vs FedAvg holding arch +
  partition + alpha fixed) computed via paired-bootstrap CI95 on the
  per-seed delta, mirroring the v6 aggregator's stats stack.
* Cells with corrupt or missing ``summary.json`` are skipped with a
  warning, never crashing the whole run.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "aggregate_v7_results.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("aggregate_v7_results", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["aggregate_v7_results"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def agg():
    return _load_module()


# ---------------------------------------------------------------------------
# Synthetic cell-tree fixture
# ---------------------------------------------------------------------------

def _write_cell(sweep_dir: Path, *, arch: str, algorithm: str,
                partition_mode: str, alpha, seed: int,
                test_auc: float, test_f1: float = 0.5,
                test_accuracy: float = 0.6,
                params_count: int = 44553,
                cell_name: str | None = None,
                history: list[dict] | None = None,
                extra: dict | None = None) -> Path:
    """Materialise a single fl_v7 cell directory with summary + history."""
    if cell_name is None:
        if partition_mode == "iid":
            cell_name = f"v7_{arch}_{algorithm}_iid_s{seed}"
        else:
            tag = f"{alpha:.2f}".replace(".", "p")
            cell_name = f"v7_{arch}_{algorithm}_dir_a{tag}_s{seed}"
    cell = sweep_dir / cell_name
    cell.mkdir(parents=True, exist_ok=True)
    summary = {
        "arch": arch,
        "algorithm": algorithm,
        "partition_mode": partition_mode,
        "alpha": alpha,
        "seed": seed,
        "test_auc": float(test_auc),
        "test_f1": float(test_f1),
        "test_accuracy": float(test_accuracy),
        "params_count": int(params_count),
    }
    if extra:
        summary.update(extra)
    (cell / "summary.json").write_text(json.dumps(summary))
    if history is None:
        history = [{"round": r, "train_loss": 0.5 - 0.05 * r,
                    "val_auc": 0.7 + 0.01 * r} for r in range(3)]
    # Minimal history.csv (column "round,train_loss,val_auc")
    lines = ["round,train_loss,val_auc"]
    for h in history:
        lines.append(f"{h['round']},{h['train_loss']},{h['val_auc']}")
    (cell / "history.csv").write_text("\n".join(lines) + "\n")
    return cell


@pytest.fixture
def synthetic_sweep(tmp_path):
    """Build a small synthetic sweep matrix:
    2 archs × 2 algos × 1 partition × 1 alpha × 3 seeds = 12 cells.
    AUCs designed so (arch=mamba) > (arch=lstm) and (algo=fedprox) >
    (algo=fedavg) by enough margin that bootstrap CIs do not bracket 0.
    """
    sweep = tmp_path / "v7_sweep"
    base_aucs = {
        ("lstm", "fedavg"):  [0.80, 0.81, 0.79],
        ("lstm", "fedprox"): [0.82, 0.83, 0.81],
        ("mamba", "fedavg"): [0.84, 0.85, 0.83],
        ("mamba", "fedprox"): [0.86, 0.87, 0.85],
    }
    for (arch, algo), aucs in base_aucs.items():
        for seed, auc in zip([42, 0, 1], aucs):
            _write_cell(sweep, arch=arch, algorithm=algo,
                        partition_mode="dirichlet", alpha=0.5, seed=seed,
                        test_auc=auc)
    return sweep


# ---------------------------------------------------------------------------
# 1. Cell discovery + load
# ---------------------------------------------------------------------------

def test_load_cells_returns_one_entry_per_cell_dir(agg, synthetic_sweep):
    """``load_cells`` returns a dict keyed by
    ``(arch, algorithm, partition_mode, alpha, seed)``."""
    cells = agg.load_cells(synthetic_sweep)
    assert len(cells) == 12
    # Spot-check: a known key must be present and carry the AUC we wrote.
    key = ("lstm", "fedavg", "dirichlet", 0.5, 42)
    assert key in cells
    assert cells[key]["test_auc"] == pytest.approx(0.80)


def test_load_cells_skips_corrupt_summary_with_warning(agg, synthetic_sweep, capsys):
    """A single half-written summary.json must NOT crash the load. A
    warning goes to stdout / stderr; the cell is silently dropped."""
    bad = synthetic_sweep / "v7_lstm_fedavg_dir_a0p50_s99"
    bad.mkdir()
    (bad / "summary.json").write_text("{not valid json")
    cells = agg.load_cells(synthetic_sweep)
    assert len(cells) == 12  # the corrupt cell is dropped
    out = capsys.readouterr()
    assert "warning" in (out.out + out.err).lower()


def test_load_cells_skips_dir_without_summary(agg, synthetic_sweep):
    """Stray subdirectories (e.g. partial run aborted before summary
    write) must be ignored, not error."""
    (synthetic_sweep / "v7_lstm_fedavg_dir_a0p50_s100").mkdir()
    cells = agg.load_cells(synthetic_sweep)
    assert len(cells) == 12


def test_load_cells_returns_empty_when_dir_empty(agg, tmp_path):
    """An empty sweep directory yields an empty dict — caller decides
    whether to error."""
    sweep = tmp_path / "empty_sweep"
    sweep.mkdir()
    assert agg.load_cells(sweep) == {}


def test_warn_writes_to_stderr_not_stdout(agg, tmp_path, capsys):
    """Review I2: warnings should go to stderr so production stdout
    stays clean (only the final 'wrote ...' line)."""
    sweep = tmp_path / "stderr_check"
    sweep.mkdir()
    bad_cell = sweep / "v7_lstm_fedavg_iid_s99"
    bad_cell.mkdir()
    (bad_cell / "summary.json").write_text("{not valid json")
    agg.load_cells(sweep)
    captured = capsys.readouterr()
    assert "warning" in captured.err.lower()
    assert "warning" not in captured.out.lower()


def test_per_group_stats_handles_missing_params_count(agg, tmp_path, capsys):
    """Review I1: a cell missing ``params_count`` must not silently
    contribute 0 to the mean (which would drag the table value down).
    Mean is computed over present cells only; a warning surfaces the
    skipped cell count."""
    sweep = tmp_path / "missing_params"
    # 3 cells, 1 of which lacks params_count.
    _write_cell(sweep, arch="lstm", algorithm="fedavg", partition_mode="iid",
                alpha=None, seed=42, test_auc=0.8, params_count=44553)
    _write_cell(sweep, arch="lstm", algorithm="fedavg", partition_mode="iid",
                alpha=None, seed=0, test_auc=0.8, params_count=44553)
    # Third cell with explicit params_count=None to trigger the guard.
    cell = sweep / "v7_lstm_fedavg_iid_s1"
    cell.mkdir()
    (cell / "summary.json").write_text(json.dumps({
        "arch": "lstm", "algorithm": "fedavg", "partition_mode": "iid",
        "alpha": None, "seed": 1, "test_auc": 0.8, "params_count": None,
    }))
    (cell / "history.csv").write_text("round\n0\n")
    cells = agg.load_cells(sweep)
    stats = agg.per_group_stats(cells)
    s = stats[("lstm", "fedavg", "iid", None)]
    # Mean over the 2 present params; the missing one is excluded
    # rather than counted as 0.
    assert s["params_count_mean"] == pytest.approx(44553.0)
    captured = capsys.readouterr()
    assert "params_count" in (captured.out + captured.err)


def test_load_cells_warns_on_duplicate_metadata_key(agg, tmp_path, capsys):
    """Two cell dirs with identical (arch, algo, partition, alpha, seed)
    are a data-integrity bug (typically a rename/copy mistake). Last-
    write-wins would silently drop one AUC; we must warn so the user
    decides whether to clean up."""
    sweep = tmp_path / "dup_sweep"
    _write_cell(sweep, arch="lstm", algorithm="fedavg",
                partition_mode="iid", alpha=None, seed=42, test_auc=0.7,
                cell_name="v7_lstm_fedavg_iid_s42")
    _write_cell(sweep, arch="lstm", algorithm="fedavg",
                partition_mode="iid", alpha=None, seed=42, test_auc=0.8,
                cell_name="v7_lstm_fedavg_iid_s42_DUPLICATE")
    cells = agg.load_cells(sweep)
    assert len(cells) == 1
    out = capsys.readouterr()
    assert "duplicate" in (out.out + out.err).lower()


# ---------------------------------------------------------------------------
# 2. Per-(arch, algo) aggregation
# ---------------------------------------------------------------------------

def test_per_group_stats_aggregates_over_seeds(agg, synthetic_sweep):
    """``per_group_stats`` returns one entry per
    ``(arch, algorithm, partition_mode, alpha)`` group, with mean / std
    AUC computed across the 3 seeds."""
    cells = agg.load_cells(synthetic_sweep)
    stats = agg.per_group_stats(cells)
    key = ("lstm", "fedavg", "dirichlet", 0.5)
    assert key in stats
    s = stats[key]
    assert s["n"] == 3
    assert s["test_auc_mean"] == pytest.approx(0.80, abs=1e-9)
    # Standard deviation of [0.79, 0.80, 0.81] with ddof=1 = 0.01.
    assert s["test_auc_std"] == pytest.approx(0.01, abs=1e-9)
    assert sorted(s["seeds"]) == [0, 1, 42]


def test_per_group_stats_iid_alpha_is_none(agg, tmp_path):
    """IID cells carry ``alpha=None``; the group key must reflect that
    (not coerced to 0.0 or string)."""
    sweep = tmp_path / "iid_sweep"
    for seed in (42, 0, 1):
        _write_cell(sweep, arch="lstm", algorithm="fedavg",
                    partition_mode="iid", alpha=None, seed=seed,
                    test_auc=0.7 + 0.01 * seed)
    stats = agg.per_group_stats(agg.load_cells(sweep))
    assert ("lstm", "fedavg", "iid", None) in stats


# ---------------------------------------------------------------------------
# 3. Paired-bootstrap pairwise deltas
# ---------------------------------------------------------------------------

def test_paired_bootstrap_delta_holds_axes_fixed(agg, synthetic_sweep):
    """``paired_bootstrap_delta`` for (mamba vs lstm) under (fedavg, dir,
    0.5) takes per-seed paired AUCs and returns CI95 on the mean delta.
    Mamba > LSTM by ~0.04 in our fixture; CI must be strictly positive."""
    cells = agg.load_cells(synthetic_sweep)
    d = agg.paired_bootstrap_delta(
        cells,
        a={"arch": "mamba", "algorithm": "fedavg",
           "partition_mode": "dirichlet", "alpha": 0.5},
        b={"arch": "lstm", "algorithm": "fedavg",
           "partition_mode": "dirichlet", "alpha": 0.5},
        n_boot=2000, seed=2026,
    )
    assert d["n_paired_seeds"] == 3
    assert d["delta_mean"] == pytest.approx(0.04, abs=1e-9)
    assert d["ci_lo"] is not None and d["ci_hi"] is not None
    assert d["ci_lo"] > 0  # Mamba arm's lower CI bound is above 0


def test_paired_bootstrap_delta_dict_shape_consistent_across_branches(agg, tmp_path):
    """Review B1: the n<2 early return must include the same keys as the
    n>=2 branch (specifically ``delta_std``) so JSON consumers /
    downstream code do not KeyError on single-seed cells."""
    sweep = tmp_path / "shape"
    _write_cell(sweep, arch="lstm", algorithm="fedavg",
                partition_mode="iid", alpha=None, seed=42, test_auc=0.7)
    _write_cell(sweep, arch="mamba", algorithm="fedavg",
                partition_mode="iid", alpha=None, seed=42, test_auc=0.8)
    cells = agg.load_cells(sweep)
    d_n1 = agg.paired_bootstrap_delta(
        cells,
        a={"arch": "mamba", "algorithm": "fedavg",
           "partition_mode": "iid", "alpha": None},
        b={"arch": "lstm", "algorithm": "fedavg",
           "partition_mode": "iid", "alpha": None},
        n_boot=200,
    )
    assert "delta_std" in d_n1, f"n<2 dict missing delta_std: {sorted(d_n1)}"
    assert d_n1["delta_std"] is None


def test_paired_bootstrap_delta_returns_none_ci_when_under_two_seeds(agg, tmp_path):
    """Cannot bootstrap meaningfully with n=1; CI fields must be None
    rather than NaN (so the JSON / Markdown stays serialisable)."""
    sweep = tmp_path / "thin"
    _write_cell(sweep, arch="lstm", algorithm="fedavg",
                partition_mode="iid", alpha=None, seed=42, test_auc=0.7)
    _write_cell(sweep, arch="mamba", algorithm="fedavg",
                partition_mode="iid", alpha=None, seed=42, test_auc=0.8)
    cells = agg.load_cells(sweep)
    d = agg.paired_bootstrap_delta(
        cells,
        a={"arch": "mamba", "algorithm": "fedavg",
           "partition_mode": "iid", "alpha": None},
        b={"arch": "lstm", "algorithm": "fedavg",
           "partition_mode": "iid", "alpha": None},
        n_boot=500, seed=2026,
    )
    assert d["n_paired_seeds"] == 1
    assert d["ci_lo"] is None
    assert d["ci_hi"] is None


# ---------------------------------------------------------------------------
# 4. Markdown rendering
# ---------------------------------------------------------------------------

def test_render_results_md_contains_table_4_header_and_all_groups(agg, synthetic_sweep):
    """Rendered Markdown must contain the Stage 2 Table 4 header + an
    entry for each (algo, arch) group present in the data."""
    cells = agg.load_cells(synthetic_sweep)
    stats = agg.per_group_stats(cells)
    md = agg.render_results_md(stats, deltas={})
    assert "Table 4" in md or "FL × architecture" in md
    for arch in ("lstm", "mamba"):
        assert arch in md
    for algo in ("fedavg", "fedprox"):
        assert algo in md


def test_render_results_md_handles_empty_stats(agg):
    """No cells → header-only output, no crash."""
    md = agg.render_results_md(stats={}, deltas={})
    assert isinstance(md, str)
    assert len(md) > 0


# ---------------------------------------------------------------------------
# 5. End-to-end main()
# ---------------------------------------------------------------------------

def test_main_writes_md_and_json_atomically(agg, synthetic_sweep, tmp_path,
                                             monkeypatch):
    """Invoke main() with overridden args. Output md + json must exist
    and json must round-trip via json.loads."""
    out_md = tmp_path / "RESULTS_V7_PHASE2.md"
    out_json = tmp_path / "aggregated_v7.json"
    monkeypatch.setattr(sys, "argv", [
        "aggregate_v7_results.py",
        "--sweep-dir", str(synthetic_sweep),
        "--out-md", str(out_md),
        "--out-json", str(out_json),
        "--n-boot", "500",
    ])
    agg.main()
    assert out_md.exists() and out_md.stat().st_size > 0
    assert out_json.exists() and out_json.stat().st_size > 0
    payload = json.loads(out_json.read_text())
    assert "stats" in payload
    # 4 groups: 2 archs × 2 algos × 1 partition × 1 alpha
    assert len(payload["stats"]) == 4


def test_main_errors_loudly_on_empty_sweep(agg, tmp_path, monkeypatch):
    """Running against an empty directory must raise — silently writing
    an empty Table 4 would mislead a reader into thinking the sweep
    succeeded with zero cells."""
    sweep = tmp_path / "empty"
    sweep.mkdir()
    monkeypatch.setattr(sys, "argv", [
        "aggregate_v7_results.py",
        "--sweep-dir", str(sweep),
        "--out-md", str(tmp_path / "out.md"),
        "--out-json", str(tmp_path / "out.json"),
    ])
    with pytest.raises(RuntimeError, match=r"[Nn]o.*cells"):
        agg.main()
