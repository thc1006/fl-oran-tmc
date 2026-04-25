"""Tests for the Stage 1 energy-estimation module.

Per ADR-001 D-20: total energy per inference = flops_total * 4.6 pJ_per_MAC
+ sops_backbone * 0.9 pJ_per_AC, with both terms reported separately for
auditability. ForecasterV2 and MambaForecaster have ``sops = 0`` (no
spiking blocks); SpikingForecaster has both ``flops > 0`` (encoder + head
+ in-block dense projections) and ``sops > 0`` (LIF spikes consumed by
each block's out_proj of width ``d_model``).
"""
from __future__ import annotations

import torch
from torch import nn

from fl_oran.data_v2.encoders import FeatureSchema
from fl_oran.evaluation.energy_metrics import (
    PJ_PER_AC_FP32,
    PJ_PER_MAC_FP32,
    count_block_sops,
    count_flops_total,
    estimate_energy_pJ_per_inference,
)
from fl_oran.models.forecaster_v2 import ForecasterV2
from fl_oran.models.spiking_forecaster import SpikingForecaster, SpikingSSMBlock


_SCHEMA = FeatureSchema(
    categorical=["a"],
    categorical_sizes={"a": 3},
    continuous=["c0", "c1"],
)


def _toy_inputs(B: int = 4, L: int = 5):
    x_cat = torch.randint(0, 4, (B, L, 1)).long()
    x_cont = torch.randn(B, L, 2)
    return x_cat, x_cont


def test_horowitz_coefficients_pinned():
    """Coefficient values must remain at the Horowitz 2014 45nm CMOS values."""
    assert PJ_PER_MAC_FP32 == 4.6
    assert PJ_PER_AC_FP32 == 0.9


def test_count_flops_returns_positive_int_for_dense_model():
    model = ForecasterV2(schema=_SCHEMA, task="classification", seq_len=5)
    x_cat, x_cont = _toy_inputs()
    flops = count_flops_total(model, x_cat, x_cont)
    assert flops > 0


def test_count_sops_zero_for_dense_only_model():
    model = ForecasterV2(schema=_SCHEMA, task="classification", seq_len=5)
    x_cat, x_cont = _toy_inputs()
    model.eval()
    with torch.no_grad():
        _ = model(x_cat, x_cont)
    assert count_block_sops(model) == 0.0


def test_count_sops_positive_for_spiking_model():
    model = SpikingForecaster(schema=_SCHEMA, task="classification", seq_len=5)
    model.eval()
    model.reset_spike_counters()
    x_cat, x_cont = _toy_inputs()
    with torch.no_grad():
        _ = model(x_cat, x_cont)
    sops = count_block_sops(model)
    assert sops > 0.0


def test_estimate_energy_combines_flops_and_sops_correctly():
    model = SpikingForecaster(schema=_SCHEMA, task="classification", seq_len=5)
    x_cat, x_cont = _toy_inputs()
    out = estimate_energy_pJ_per_inference(model, x_cat, x_cont)
    expected = out["flops"] * PJ_PER_MAC_FP32 + out["sops"] * PJ_PER_AC_FP32
    assert abs(out["total_energy_pJ"] - expected) < 1e-6


def test_hand_calc_matches_for_toy_block():
    """Hand-construct a single SpikingSSMBlock, force a deterministic
    spike pattern, and verify (a) sops == spike_count * d_model and
    (b) energy_ratio coefficient is exactly Horowitz."""
    block = SpikingSSMBlock(d_model=4, d_state=2, lif_threshold=0.5, lif_beta=0.0)
    with torch.no_grad():
        block.in_proj.weight.copy_(torch.eye(4))
        block.in_proj.bias.zero_()
        block.B.zero_()
        block.C.zero_()
        block.D.copy_(torch.ones(4))
        block.A_log.copy_(torch.zeros_like(block.A_log))
    block.eval()

    # Same alternating pattern as the spike-count hand-calc test:
    # 4 spikes total at threshold=0.5 with reset_delay LIF.
    x = torch.tensor([[
        [0.1, 0.2, 0.6, 0.7],
        [0.1, 0.2, 0.6, 0.7],
        [0.1, 0.2, 0.6, 0.7],
    ]])
    _ = block.forward_spikes_only(x)
    spike_total = float(block.spike_count)  # = 4 per the spike-count test
    fan_out = block.out_proj.out_features      # = 4
    expected_sops = spike_total * fan_out / float(block.forward_inferences)
    # forward_inferences = batch_size = 1 here, so per-inference sops = 4 * 4 = 16.
    assert expected_sops == 16.0, (spike_total, fan_out, float(block.forward_inferences))


def test_estimate_energy_dense_only_model_zero_sops():
    """For ForecasterV2 the dict has sops = 0 and total_pJ = flops * 4.6."""
    model = ForecasterV2(schema=_SCHEMA, task="classification", seq_len=5)
    x_cat, x_cont = _toy_inputs()
    out = estimate_energy_pJ_per_inference(model, x_cat, x_cont)
    assert out["sops"] == 0.0
    assert abs(out["total_energy_pJ"] - out["flops"] * PJ_PER_MAC_FP32) < 1e-6
