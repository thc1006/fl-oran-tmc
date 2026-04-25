"""Routing tests for scripts/aggregate_v6_results.py.

Covers two areas the post-Option-B code review flagged:

A. ``_parse_cell_name`` correctly derives ``(arch_label, seed)`` from
   directory names produced by every output_suffix in active use.
B. The matched-budget D-21 routing for spiking variants picks the
   correct lstm/mamba baseline (5k vs 25k vs 50k) based on the
   variant's name.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# The aggregator script is at scripts/aggregate_v6_results.py — load it
# as a module without going through a package import.
SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "aggregate_v6_results.py"


def _load_aggregator():
    spec = importlib.util.spec_from_file_location("aggregate_v6_results", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["aggregate_v6_results"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def agg():
    return _load_aggregator()


@pytest.mark.parametrize(
    "cell_name, expected_arch_label, expected_seed",
    [
        # Plain main-sweep variants.
        ("lstm_s42", "lstm", 42),
        ("mamba_s0", "mamba", 0),
        ("spiking_s7", "spiking", 7),
        # Suffix-bearing audit variants.
        ("lstm_s42_25k", "lstm_25k", 42),
        ("lstm_s0_50k", "lstm_50k", 0),
        ("mamba_s11_25k", "mamba_25k", 11),
        ("mamba_s23_50k", "mamba_50k", 23),
        ("spiking_s17_lr5e4_25k", "spiking_lr5e4_25k", 17),
        ("spiking_s2_t5", "spiking_t5", 2),
        ("spiking_s13_t5sum", "spiking_t5sum", 13),
        ("spiking_s7_t5sum_50k", "spiking_t5sum_50k", 7),
        ("spiking_s42_lif_t05_b09", "spiking_lif_t05_b09", 42),
        # Architectural-ablation variants.
        ("mamba_expand2_s1", "mamba_expand2", 1),
        ("spiking_expand2_s3", "spiking_expand2", 3),
    ],
)
def test_parse_cell_name(agg, cell_name, expected_arch_label, expected_seed):
    arch_label, seed = agg._parse_cell_name(cell_name)
    assert arch_label == expected_arch_label
    assert seed == expected_seed


def test_evaluate_d21_routing_picks_25k_baselines_for_lr5e4_25k(agg):
    """A variant labeled with the ``_lr5e4_25k`` budget must match against
    the lstm_25k / mamba_25k baselines, not the 5k baselines."""
    stats = {
        "lstm": {"test_auc_mean": 0.91, "energy_pJ_mean": 1e6, "n": 10},
        "lstm_25k": {"test_auc_mean": 0.92, "energy_pJ_mean": 1e6, "n": 10},
        "mamba": {"test_auc_mean": 0.91, "n": 10},
        "mamba_25k": {"test_auc_mean": 0.92, "n": 10},
        "spiking_lr5e4_25k": {"test_auc_mean": 0.89, "energy_pJ_mean": 5e5, "n": 10},
    }
    deltas = {
        ("spiking_lr5e4_25k", "lstm_25k"): {
            "ci_lo": -0.032, "ci_hi": -0.028, "delta_mean": -0.030,
            "n_paired_seeds": 10,
        },
        ("mamba_25k", "lstm_25k"): {
            "ci_lo": -0.001, "ci_hi": +0.001, "delta_mean": 0.0,
            "n_paired_seeds": 10,
        },
    }
    out = agg.evaluate_d21_criteria(
        stats, deltas, spiking_key="spiking_lr5e4_25k",
        lstm_key="lstm_25k", mamba_key="mamba_25k",
    )
    assert out["lstm_variant_evaluated"] == "lstm_25k"
    # Sanity: with hi=-0.028, C1 PASSES (>= -0.030 threshold).
    assert out["C1_accuracy_gap_spiking_vs_lstm"]["pass"] is True


def test_evaluate_d21_routing_50k_uses_50k_baselines(agg):
    stats = {
        "lstm_50k": {"test_auc_mean": 0.927, "energy_pJ_mean": 1e6, "n": 10},
        "mamba_50k": {"test_auc_mean": 0.927, "n": 10},
        "spiking_t5sum_50k": {"test_auc_mean": 0.90, "energy_pJ_mean": 5e5, "n": 10},
    }
    deltas = {
        ("spiking_t5sum_50k", "lstm_50k"): {
            "ci_lo": -0.030, "ci_hi": -0.024, "delta_mean": -0.027,
            "n_paired_seeds": 10,
        },
        ("mamba_50k", "lstm_50k"): {
            "ci_lo": -0.001, "ci_hi": +0.001, "delta_mean": 0.0,
            "n_paired_seeds": 10,
        },
    }
    out = agg.evaluate_d21_criteria(
        stats, deltas, spiking_key="spiking_t5sum_50k",
        lstm_key="lstm_50k", mamba_key="mamba_50k",
    )
    assert out["lstm_variant_evaluated"] == "lstm_50k"
    assert out["C1_accuracy_gap_spiking_vs_lstm"]["pass"] is True
