"""Tests for the per-block spike-count buffer used by Stage 1 energy metrics.

Per ADR-001 D-20: ``SpikingSSMBlock`` accumulates a ``spike_count`` buffer
across forward passes (eval mode only). ``reset_spike_counters`` zeros it.
The aggregated count drives the ``sops`` term in the energy formula
``total_energy = (flops_encoder_head + flops_backbone_dense) * 4.6 + sops * 0.9``.
"""
from __future__ import annotations

import torch

from fl_oran.models.spiking_forecaster import SpikingForecaster, SpikingSSMBlock


def test_spike_count_zero_on_init():
    block = SpikingSSMBlock(d_model=8, d_state=4)
    assert float(block.spike_count) == 0.0
    assert float(block.forward_inferences) == 0.0


def test_spike_count_accumulates_in_eval_mode():
    block = SpikingSSMBlock(d_model=8, d_state=4)
    block.eval()
    x = torch.randn(2, 3, 8)
    _ = block(x)
    assert float(block.spike_count) >= 0.0
    # forward_inferences = batch_size per inference (averaged across L steps).
    # For one (B=2, L=3) call, total = 2 * (3 * 1/3) = 2.
    assert abs(float(block.forward_inferences) - 2.0) < 1e-5


def test_spike_count_does_not_update_in_train_mode():
    block = SpikingSSMBlock(d_model=8, d_state=4)
    block.train()
    x = torch.randn(2, 3, 8)
    _ = block(x)
    assert float(block.spike_count) == 0.0
    assert float(block.forward_inferences) == 0.0


def test_spike_count_monotonic_across_calls():
    block = SpikingSSMBlock(d_model=8, d_state=4)
    block.eval()
    x = torch.randn(2, 5, 8)
    _ = block(x)
    first = float(block.spike_count)
    _ = block(x)
    second = float(block.spike_count)
    assert second >= first


def test_reset_spike_counters_zeros_buffers():
    block = SpikingSSMBlock(d_model=8, d_state=4)
    block.eval()
    x = torch.randn(2, 5, 8)
    _ = block(x)
    block.reset_spike_counters()
    assert float(block.spike_count) == 0.0
    assert float(block.forward_inferences) == 0.0


def test_spike_count_is_integer_valued():
    """Spikes are 0 or 1 each, so cumulative spike_count must be integer."""
    block = SpikingSSMBlock(d_model=8, d_state=4)
    block.eval()
    x = torch.randn(2, 5, 8)
    _ = block(x)
    val = float(block.spike_count)
    assert abs(val - round(val)) < 1e-5, f"non-integer spike count: {val}"


def test_spike_count_matches_manual_hand_count_with_forced_threshold_crossing():
    """Drive a 1-block, 4-channel net with deterministic input and verify
    that the cumulative spike count equals the manually-computed
    threshold-crossings tally on the LIF outputs."""
    torch.manual_seed(0)
    block = SpikingSSMBlock(d_model=4, d_state=2, lif_threshold=0.5, lif_beta=0.0)
    # Force the in_proj to the identity-like map and zero the SSM contribution
    # by setting B=0, C=0, D=1, A_log → log(1)=0; with these, y_t == u[:, t, :].
    # That makes spikes purely a function of the input passing through in_proj.
    with torch.no_grad():
        block.in_proj.weight.copy_(torch.eye(4))
        block.in_proj.bias.zero_()
        block.B.zero_()
        block.C.zero_()
        block.D.copy_(torch.ones(4))
        block.A_log.copy_(torch.zeros_like(block.A_log))
    block.eval()

    # Input: a sequence where some channels exceed threshold and some don't.
    # Channels 0,1 stay below 0.5; channels 2,3 cross.
    x = torch.tensor([
        [[0.1, 0.2, 0.6, 0.7],   # t=0
         [0.1, 0.2, 0.6, 0.7],   # t=1
         [0.1, 0.2, 0.6, 0.7]],  # t=2
    ])
    spikes = block.forward_spikes_only(x)
    # snntorch.Leaky uses ``reset_delay=True`` semantics (the post-spike
    # subtraction is applied at the NEXT timestep, so spikes alternate
    # between 1 and 0 even when the input keeps exceeding threshold).
    # For input [0.6, 0.7] held over 3 timesteps with beta=0, threshold=0.5:
    #   t=0: mem=0.7, spike=1, mem stays 0.7 (delayed reset)
    #   t=1: mem=0+0.7-0.5=0.2, spike=0 (below threshold after reset)
    #   t=2: mem=0+0.7-0=0.7, spike=1
    # → 2 spikes per crossing channel × 2 channels = 4. Channels 0,1
    # never cross because their inputs (0.1, 0.2) are below threshold.
    expected = 4.0
    assert spikes.sum().item() == expected, spikes
    assert float(block.spike_count) == expected


def test_full_spiking_forecaster_reset_propagates_to_all_blocks():
    """SpikingForecaster.reset_spike_counters() should zero every block."""
    from fl_oran.data_v2.encoders import FeatureSchema

    schema = FeatureSchema(
        categorical=["a"],
        categorical_sizes={"a": 3},
        continuous=["c0", "c1"],
    )
    model = SpikingForecaster(schema=schema, task="classification", seq_len=5)
    model.eval()
    x_cat = torch.randint(0, 4, (2, 5, 1)).long()
    x_cont = torch.randn(2, 5, 2)
    _ = model(x_cat, x_cont)
    # At least one block should have observed spikes.
    assert any(float(b.spike_count) > 0 for b in model.blocks)
    model.reset_spike_counters()
    for b in model.blocks:
        assert float(b.spike_count) == 0.0
        assert float(b.forward_inferences) == 0.0
