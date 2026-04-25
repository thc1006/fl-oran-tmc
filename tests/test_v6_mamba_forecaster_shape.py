"""Shape, gradient, and determinism tests for MambaForecaster.

Per ADR-001 D-20: MambaForecaster wraps a pure-PyTorch MambaS6Block backbone
inside the same encoder + classifier head as ForecasterV2. The interface
must be drop-in compatible with ForecasterV2 — same forward signature
`forward(x_cat, x_cont) -> (B, 1)`, same FeatureSchema, same dtype contracts.
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


def test_mamba_forecaster_classification_output_shape():
    from fl_oran.models.mamba_forecaster import MambaForecaster

    x_cat, x_cont, schema = _make_inputs()
    model = MambaForecaster(schema=schema, task="classification", seq_len=5)
    out = model(x_cat, x_cont)
    assert out.shape == (4, 1), out.shape


def test_mamba_forecaster_gradient_flows_to_embeddings():
    from fl_oran.models.mamba_forecaster import MambaForecaster

    torch.manual_seed(42)
    x_cat, x_cont, schema = _make_inputs()
    model = MambaForecaster(schema=schema, task="classification", seq_len=5)
    out = model(x_cat, x_cont)
    out.sum().backward()
    for col, emb in model.embeddings.items():
        assert emb.weight.grad is not None, f"no grad on {col}"
        assert torch.isfinite(emb.weight.grad).all()
        # At least one row should have non-zero gradient (the row of an actual sampled id).
        assert emb.weight.grad.norm().item() > 0


def test_mamba_forecaster_deterministic_across_two_runs_same_seed():
    from fl_oran.models.mamba_forecaster import MambaForecaster

    schema = _make_schema()
    torch.manual_seed(123)
    x_cat = torch.stack(
        [torch.randint(0, sz + 1, (2, 5)) for sz in schema.categorical_sizes.values()],
        dim=-1,
    ).long()
    x_cont = torch.randn(2, 5, schema.n_continuous)

    torch.manual_seed(7)
    m1 = MambaForecaster(schema=schema, task="classification", seq_len=5)
    out1 = m1(x_cat, x_cont)

    torch.manual_seed(7)
    m2 = MambaForecaster(schema=schema, task="classification", seq_len=5)
    out2 = m2(x_cat, x_cont)

    assert torch.allclose(out1, out2, atol=0.0), (out1 - out2).abs().max().item()


def test_mamba_forecaster_handles_no_persistence_for_regression():
    """Regression task without persistence_feature must still produce (B, 1) outputs."""
    from fl_oran.models.mamba_forecaster import MambaForecaster

    x_cat, x_cont, schema = _make_inputs()
    model = MambaForecaster(schema=schema, task="regression", seq_len=5)
    out = model(x_cat, x_cont)
    assert out.shape == (4, 1)
