"""Shape, spike-binarisation, and surrogate-gradient tests for SpikingForecaster.

Per ADR-001 D-20: SpikingForecaster wraps in-tree ``SpikingSSMBlock`` (which
internally uses ``snntorch.Leaky`` with atan surrogate, alpha=2) and shares
the same encoder + classifier head as ForecasterV2.
"""
from __future__ import annotations

import pytest
import torch

from fl_oran.data_v2.encoders import FeatureSchema


def _make_schema() -> FeatureSchema:
    return FeatureSchema(
        categorical=["bs_id", "slice_id"],
        categorical_sizes={"bs_id": 7, "slice_id": 3},
        continuous=["dl_throughput_mbps", "ul_throughput_mbps", "prb_util"],
    )


def _make_inputs(B: int = 4, L: int = 5):
    schema = _make_schema()
    x_cat = torch.stack(
        [torch.randint(0, sz + 1, (B, L)) for sz in schema.categorical_sizes.values()],
        dim=-1,
    ).long()
    x_cont = torch.randn(B, L, schema.n_continuous)
    return x_cat, x_cont, schema


def test_spiking_forecaster_classification_output_shape():
    from fl_oran.models.spiking_forecaster import SpikingForecaster

    x_cat, x_cont, schema = _make_inputs()
    model = SpikingForecaster(schema=schema, task="classification", seq_len=5)
    out = model(x_cat, x_cont)
    assert out.shape == (4, 1), out.shape


def test_spiking_block_emits_binary_spikes_only():
    """The intermediate spike tensor inside SpikingSSMBlock must be ∈ {0, 1}."""
    from fl_oran.models.spiking_forecaster import SpikingSSMBlock

    block = SpikingSSMBlock(d_model=16, d_state=8, lif_threshold=1.0, lif_beta=0.9)
    x = torch.randn(3, 5, 16)
    # Capture the raw spikes (pre-out_proj) via the helper.
    spikes = block.forward_spikes_only(x)
    assert spikes.shape == (3, 5, 16), spikes.shape
    unique = torch.unique(spikes).tolist()
    assert set(unique).issubset({0.0, 1.0}), f"non-binary spikes: {unique}"


def test_spiking_forecaster_gradient_flows_through_surrogate():
    from fl_oran.models.spiking_forecaster import SpikingForecaster

    torch.manual_seed(11)
    x_cat, x_cont, schema = _make_inputs()
    model = SpikingForecaster(schema=schema, task="classification", seq_len=5)
    out = model(x_cat, x_cont)
    loss = out.sum()
    loss.backward()

    # Embeddings should have gradient.
    for col, emb in model.embeddings.items():
        assert emb.weight.grad is not None and torch.isfinite(emb.weight.grad).all()
        assert emb.weight.grad.norm().item() > 0, f"no grad on embedding {col}"

    # The recurrent A_log parameter inside at least one spiking block should have grad.
    grads_total = 0.0
    for block in model.blocks:
        assert block.A_log.grad is not None and torch.isfinite(block.A_log.grad).all()
        grads_total += block.A_log.grad.norm().item()
    assert grads_total > 0, "no gradient flowed back through the spiking blocks"


def test_spiking_forecaster_deterministic_across_two_runs_same_seed():
    from fl_oran.models.spiking_forecaster import SpikingForecaster

    schema = _make_schema()
    torch.manual_seed(123)
    x_cat = torch.stack(
        [torch.randint(0, sz + 1, (2, 5)) for sz in schema.categorical_sizes.values()],
        dim=-1,
    ).long()
    x_cont = torch.randn(2, 5, schema.n_continuous)

    torch.manual_seed(7)
    m1 = SpikingForecaster(schema=schema, task="classification", seq_len=5)
    out1 = m1(x_cat, x_cont)

    torch.manual_seed(7)
    m2 = SpikingForecaster(schema=schema, task="classification", seq_len=5)
    out2 = m2(x_cat, x_cont)

    assert torch.allclose(out1, out2, atol=0.0), (out1 - out2).abs().max().item()
