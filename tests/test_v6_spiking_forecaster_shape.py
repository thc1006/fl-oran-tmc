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


def test_spiking_expand2_param_count_and_shape():
    """SpikingForecaster with backbone_expand=2 must have d_inner = 2 × d_model
    inside every block, the output shape is unchanged at (B, 1), and the param
    count grows roughly linearly with d_inner.

    Per ARCH_REGISTRY["spiking_expand2"] the d_model is shrunk to 56 so the
    total param count lands within ±10% of the LSTM baseline.
    """
    from fl_oran.models.spiking_forecaster import SpikingForecaster

    schema = _make_schema()
    x_cat, x_cont, _ = _make_inputs()

    # expand=1 baseline (default).
    m1 = SpikingForecaster(
        schema=schema, task="classification", seq_len=5,
        backbone_d_model=56, backbone_expand=1,
    )
    out1 = m1(x_cat, x_cont)
    assert out1.shape == (4, 1)
    for blk in m1.blocks:
        assert blk.d_inner == 56  # expand=1 → d_inner == d_model

    # expand=2 — internal d_inner doubles.
    m2 = SpikingForecaster(
        schema=schema, task="classification", seq_len=5,
        backbone_d_model=56, backbone_expand=2,
    )
    out2 = m2(x_cat, x_cont)
    assert out2.shape == (4, 1)
    for blk in m2.blocks:
        assert blk.d_inner == 112  # 56 × 2 = 112
        assert blk.A_log.shape == (112, blk.d_state)
        assert blk.B.shape == (112, blk.d_state)
        assert blk.out_proj.in_features == 112
        assert blk.out_proj.out_features == 56

    # expand=2 should have more params than expand=1 (same d_model).
    n1 = sum(p.numel() for p in m1.parameters() if p.requires_grad)
    n2 = sum(p.numel() for p in m2.parameters() if p.requires_grad)
    assert n2 > n1


def test_decode_mode_sum_preserves_gradient_at_t_inner_5():
    """The audit-corrected sum-decoder must keep gradient flowing for T_inner > 1.

    The original `majority` decoder uses a hard threshold which is
    non-differentiable, blocking gradients through the LIF for T_inner > 1.
    `sum` divides by t_inner and stays differentiable.
    """
    from fl_oran.models.spiking_forecaster import SpikingForecaster

    torch.manual_seed(11)
    schema = _make_schema()
    sizes = list(schema.categorical_sizes.values())
    x_cat = torch.stack(
        [torch.randint(0, sz + 1, (4, 5)) for sz in sizes], dim=-1,
    ).long()
    x_cont = torch.randn(4, 5, schema.n_continuous)

    # majority decoder with t_inner=5: gradient on A_log is fully blocked
    # (the (>threshold).float() cast is non-differentiable). A_log.grad
    # is therefore None or all-zero.
    m_maj = SpikingForecaster(
        schema=schema, task="classification", seq_len=5,
        t_inner=5, decode_mode="majority",
    )
    out_m = m_maj(x_cat, x_cont)
    out_m.sum().backward()
    for b in m_maj.blocks:
        assert b.A_log.grad is None or b.A_log.grad.abs().sum().item() == 0.0, (
            "majority decoder unexpectedly propagated A_log gradient at t_inner=5"
        )

    # sum decoder with t_inner=5: gradient must flow back to A_log.
    m_sum = SpikingForecaster(
        schema=schema, task="classification", seq_len=5,
        t_inner=5, decode_mode="sum",
    )
    out_s = m_sum(x_cat, x_cont)
    out_s.sum().backward()
    a_log_grad_sum = sum(
        b.A_log.grad.abs().sum().item() for b in m_sum.blocks
        if b.A_log.grad is not None
    )
    assert a_log_grad_sum > 1e-6, (
        f"sum decoder failed to propagate gradient (a_log grad sum = {a_log_grad_sum:.3e})"
    )


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
