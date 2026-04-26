"""TDD tests for Phase 0 latency + EDP aggregation.

Pins the expected behavior of ``aggregate_v6_results.py`` after the
Phase 0 extension. The aggregator must, in addition to the existing
theoretical-energy fields, surface for each arch:

* ``measured_pJ_mean`` / ``measured_pJ_std`` — NVML-measured energy
  per inference, aggregated across the cells of that arch (n=10 for
  multi-seed cells).
* ``latency_ms_mean`` / ``latency_ms_std`` — per-inference wallclock
  latency derived from ``wallclock_sec / n_inferences_measured × 1000``.
* ``edp_pJ_s_mean`` / ``edp_pJ_s_std`` — Energy-Delay Product per
  inference (= ``measured_pJ × latency_ms / 1000``), the canonical
  edge-AI joint metric that penalises both energy and latency.

These three groups of fields are *derived from* per-cell
``energy_measured.json`` files; cells lacking that file are skipped
in the per-arch aggregation (n drops, but the arch still appears with
its other stats unchanged).

The hand-calc anchor below uses the LSTM s=42 cell measurement
documented in ``RESULTS_V6_STAGE1_ANALYSIS.md`` §3.2 to verify the
formula: ``wallclock_sec / 128000 × 1000``.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "aggregate_v6_results.py"


def _load_agg():
    spec = importlib.util.spec_from_file_location("aggregate_v6_results", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["aggregate_v6_results"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def agg():
    return _load_agg()


def _make_cell(tmp_path: Path, name: str, *, auc=0.92, energy_pJ=1e6,
               measured_pJ=None, wallclock_sec=None, n_inferences=128000):
    """Helper: drop a minimal valid cell directory under ``tmp_path``."""
    cell_dir = tmp_path / name
    cell_dir.mkdir()
    (cell_dir / "summary.json").write_text(json.dumps({
        "test_auc": auc, "test_f1": 0.78, "test_accuracy": 0.85,
        "params_count": 44553, "arch": name.split("_s")[0], "seed": 0,
    }))
    (cell_dir / "energy.json").write_text(json.dumps({
        "flops": 210112.0, "sops": 0.0, "total_energy_pJ": energy_pJ,
        "total_energy_pJ_gpu_dense": energy_pJ,
        "total_energy_pJ_sparsity_aware": energy_pJ,
        "total_energy_pJ_neuromorphic": energy_pJ,
    }))
    if measured_pJ is not None and wallclock_sec is not None:
        (cell_dir / "energy_measured.json").write_text(json.dumps({
            "energy_pJ_per_inference_total": measured_pJ,
            "wallclock_sec": wallclock_sec,
            "n_inferences_measured": n_inferences,
        }))
    return cell_dir


def test_load_cells_loads_energy_measured(agg, tmp_path):
    """``load_cells`` must surface energy_measured.json content under
    ``summary['energy_measured']`` so per_arch_stats can use it.
    Cells without energy_measured.json get an empty dict (graceful)."""
    _make_cell(tmp_path, "lstm_s42",
               measured_pJ=1.92e8, wallclock_sec=320.0, n_inferences=128000)
    _make_cell(tmp_path, "lstm_s0")  # no energy_measured.json
    cells = agg.load_cells(tmp_path)
    assert ("lstm", 42) in cells
    assert ("lstm", 0) in cells
    em42 = cells[("lstm", 42)].get("energy_measured", {})
    assert em42.get("energy_pJ_per_inference_total") == 1.92e8
    assert em42.get("wallclock_sec") == 320.0
    em0 = cells[("lstm", 0)].get("energy_measured", {})
    assert em0 == {}


def test_per_arch_stats_computes_latency_ms_hand_calc(agg, tmp_path):
    """Per-cell latency = wallclock_sec / n_inferences_measured × 1000.

    Anchor: 320s / 128000 × 1000 = 2.5 ms per inference.
    With two cells at 320s and 256s and matching n=128000:
      latencies = [2.5, 2.0]; mean = 2.25; std (ddof=1) = 0.353.
    """
    _make_cell(tmp_path, "lstm_s42",
               measured_pJ=1.92e8, wallclock_sec=320.0, n_inferences=128000)
    _make_cell(tmp_path, "lstm_s0",
               measured_pJ=1.80e8, wallclock_sec=256.0, n_inferences=128000)
    cells = agg.load_cells(tmp_path)
    stats = agg.per_arch_stats(cells)
    s = stats["lstm"]
    assert s["latency_ms_mean"] == pytest.approx(2.25, rel=1e-9)
    # stdev with ddof=1 of [2.5, 2.0] = sqrt(((2.5-2.25)^2 + (2.0-2.25)^2)/(2-1)) = 0.3536
    assert s["latency_ms_std"] == pytest.approx(0.35355, rel=1e-3)


def test_per_arch_stats_computes_measured_pJ_aggregates(agg, tmp_path):
    """measured_pJ_mean is the mean of ``energy_pJ_per_inference_total``
    across cells of the same arch."""
    _make_cell(tmp_path, "lstm_s42",
               measured_pJ=1.92e8, wallclock_sec=320.0, n_inferences=128000)
    _make_cell(tmp_path, "lstm_s0",
               measured_pJ=1.80e8, wallclock_sec=256.0, n_inferences=128000)
    cells = agg.load_cells(tmp_path)
    stats = agg.per_arch_stats(cells)
    s = stats["lstm"]
    assert s["measured_pJ_mean"] == pytest.approx(1.86e8, rel=1e-9)
    assert s["measured_pJ_std"] == pytest.approx(8.485e6, rel=1e-3)


def test_per_arch_stats_computes_edp(agg, tmp_path):
    """EDP = measured_pJ × latency_ms / 1000  (units: pJ·s).

    For a cell with measured_pJ=1.92e8, latency_ms=2.5:
      edp = 1.92e8 × 2.5 / 1000 = 4.8e5 pJ·s.
    Aggregated mean across two cells (1.92e8 × 2.5 and 1.80e8 × 2.0):
      [4.8e5, 3.6e5] -> mean 4.2e5.
    """
    _make_cell(tmp_path, "lstm_s42",
               measured_pJ=1.92e8, wallclock_sec=320.0, n_inferences=128000)
    _make_cell(tmp_path, "lstm_s0",
               measured_pJ=1.80e8, wallclock_sec=256.0, n_inferences=128000)
    cells = agg.load_cells(tmp_path)
    stats = agg.per_arch_stats(cells)
    s = stats["lstm"]
    assert s["edp_pJ_s_mean"] == pytest.approx(4.2e5, rel=1e-3)


def test_arch_with_no_energy_measured_gets_nan_metrics(agg, tmp_path):
    """If no cell of an arch has energy_measured.json, the latency /
    measured / EDP fields must be NaN (not crash, not 0 — silently
    reporting 0 would be a false claim).

    NaN convention: stored as None in JSON-friendly form."""
    _make_cell(tmp_path, "lstm_s42")
    cells = agg.load_cells(tmp_path)
    stats = agg.per_arch_stats(cells)
    s = stats["lstm"]
    assert s.get("measured_pJ_mean") is None
    assert s.get("latency_ms_mean") is None
    assert s.get("edp_pJ_s_mean") is None


def test_partial_coverage_uses_only_measured_cells(agg, tmp_path):
    """When 1/2 cells of an arch have energy_measured.json, the
    aggregation uses n=1 (the measured one) and reports std=0 (single
    sample), while AUC stats remain n=2."""
    _make_cell(tmp_path, "lstm_s42",
               measured_pJ=1.92e8, wallclock_sec=320.0, n_inferences=128000)
    _make_cell(tmp_path, "lstm_s0")  # no energy_measured.json
    cells = agg.load_cells(tmp_path)
    stats = agg.per_arch_stats(cells)
    s = stats["lstm"]
    assert s["n"] == 2  # auc stats unchanged
    # measured fields should reflect only the 1 cell that had data
    assert s["measured_pJ_mean"] == pytest.approx(1.92e8)
    assert s["latency_ms_mean"] == pytest.approx(2.5)
    # std with n=1 is undefined; we report 0 for consistency with auc_std
    assert s["measured_pJ_std"] == pytest.approx(0.0)
    assert s["latency_ms_std"] == pytest.approx(0.0)


def test_zero_n_inferences_is_skipped(agg, tmp_path):
    """A malformed energy_measured.json with n_inferences_measured=0
    must not produce a divide-by-zero. The cell is skipped from
    latency/EDP aggregation."""
    _make_cell(tmp_path, "lstm_s42",
               measured_pJ=1.92e8, wallclock_sec=320.0, n_inferences=0)
    cells = agg.load_cells(tmp_path)
    stats = agg.per_arch_stats(cells)
    s = stats["lstm"]
    # The malformed cell is the only one; no measured aggregation possible.
    assert s.get("latency_ms_mean") is None
