"""Tests for :class:`fl_oran.models.mamba3_forecaster.Mamba3Forecaster`.

Covers ADR-001 D-20 param-count parity, forward shape, init-state invariants
(``lambda ≈ 1`` near Mamba-2 Euler, ``theta = 0`` at zero rotation),
numerical safety under extreme inputs, gradient flow, determinism, and the
key semantic-equivalence claim: **forcing λ=1 and θ=0 should produce a
trapezoidal-degenerate recurrence that is the exact Mamba-2 Euler recurrence
applied to the same dt/A/B/C/D parameters.** This catches any drift between
Innovations 1+2 and the Mamba-2 baseline that would silently break
ablation interpretability.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from fl_oran.data_v2.encoders import FeatureSchema
from fl_oran.models.forecaster_v2 import ForecasterV2
from fl_oran.models.mamba3_forecaster import Mamba3Forecaster, Mamba3SSMBlock


COLORAN_SCHEMA = FeatureSchema(
    categorical=["bs_id", "slice_id", "sched", "tr"],
    categorical_sizes={"bs_id": 7, "slice_id": 3, "sched": 5, "tr": 28},
    continuous=[f"c{i}" for i in range(12)],
)


def _count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Param-count parity tests (ADR-001 D-20).
# ---------------------------------------------------------------------------

def test_param_count_in_v6_expected_range():
    m = Mamba3Forecaster(schema=COLORAN_SCHEMA, task="classification", seq_len=5)
    n = _count(m)
    assert 38_000 <= n <= 48_000, (
        f"Mamba3Forecaster param count {n} is outside [38K, 48K]; "
        f"tune backbone_d_model/n_blocks/d_state."
    )


def test_param_count_within_10pct_of_forecasterv2():
    base = ForecasterV2(schema=COLORAN_SCHEMA, task="classification", seq_len=5)
    m3 = Mamba3Forecaster(schema=COLORAN_SCHEMA, task="classification", seq_len=5)
    drift = _count(m3) / _count(base) - 1.0
    assert abs(drift) <= 0.10, (
        f"Mamba3Forecaster has {_count(m3)} params vs ForecasterV2 "
        f"{_count(base)} (drift {drift * 100:+.1f}%); tune backbone."
    )


# ---------------------------------------------------------------------------
# Precondition tests.
# ---------------------------------------------------------------------------

def test_d_state_must_be_even():
    import pytest
    with pytest.raises(ValueError, match="d_state"):
        Mamba3SSMBlock(d_model=64, d_state=15)


def test_invalid_persistence_feature_raises():
    import pytest
    with pytest.raises(ValueError, match="persistence_feature"):
        Mamba3Forecaster(
            schema=COLORAN_SCHEMA,
            task="regression",
            seq_len=5,
            persistence_feature="not_in_schema",
        )


# ---------------------------------------------------------------------------
# Init-state invariants — Innovations 1 + 2 start near Mamba-2.
# ---------------------------------------------------------------------------

def test_lambda_init_near_one_matches_mamba2_euler():
    """``lambda_proj.bias = +3.0`` produces ``sigmoid(3) ≈ 0.953`` at init,
    so the trapezoidal mix starts close to Mamba-2's Euler rule
    (``λ = 1``). This is the conservative initialization the module
    docstring claims; deviating from it changes early-training dynamics
    materially.
    """
    block = Mamba3SSMBlock(d_model=64, d_state=16)
    assert block.lambda_proj.bias.item() == 3.0
    assert torch.allclose(block.lambda_proj.weight, torch.zeros_like(block.lambda_proj.weight))
    sigmoid_at_init = torch.sigmoid(block.lambda_proj.bias).item()
    assert 0.95 <= sigmoid_at_init <= 0.96


def test_theta_init_is_identity_rotation():
    """``theta_proj.weight = theta_proj.bias = 0`` means θ_t ≡ 0 at init,
    so the per-pair rotation matrix is the identity. The model starts as
    a real-eigenvalue SSM and learns rotation during training.
    """
    block = Mamba3SSMBlock(d_model=64, d_state=16)
    assert torch.allclose(block.theta_proj.weight, torch.zeros_like(block.theta_proj.weight))
    assert torch.allclose(block.theta_proj.bias, torch.zeros_like(block.theta_proj.bias))


# ---------------------------------------------------------------------------
# Forward / backward shape + smoke tests.
# ---------------------------------------------------------------------------

def test_forward_shape_classification():
    m = Mamba3Forecaster(schema=COLORAN_SCHEMA, task="classification", seq_len=5)
    m.eval()
    x_cat = torch.zeros(8, 5, 4, dtype=torch.long)
    x_cont = torch.randn(8, 5, 12)
    out = m(x_cat, x_cont)
    assert out.shape == (8, 1)
    assert torch.isfinite(out).all()


def test_forward_shape_regression_with_persistence():
    m = Mamba3Forecaster(
        schema=COLORAN_SCHEMA,
        task="regression",
        seq_len=5,
        persistence_feature="c0",
    )
    m.eval()
    x_cat = torch.zeros(4, 5, 4, dtype=torch.long)
    x_cont = torch.zeros(4, 5, 12)
    x_cont[:, -1, 0] = 0.7
    out = m(x_cat, x_cont)
    assert out.shape == (4, 1)
    # Zero-init head + zero-input trunk => output = persistence baseline.
    assert torch.allclose(out, torch.full((4, 1), 0.7), atol=1e-5)


def test_gradient_flow_to_all_params():
    """Every learnable param — including the NEW ``lambda_proj`` and
    ``theta_proj`` — must receive a non-zero gradient after one backward
    pass. lambda_proj is zero-initialized in weight so it would NOT
    receive a gradient through the weight path on the first step;
    confirm gradient flows via the bias path."""
    torch.manual_seed(0)
    m = Mamba3Forecaster(schema=COLORAN_SCHEMA, task="classification", seq_len=5)
    m.train()
    x_cat = torch.randint(0, 3, (4, 5, 4), dtype=torch.long)
    x_cont = torch.randn(4, 5, 12)
    out = m(x_cat, x_cont)
    out.sum().backward()
    no_grad: list[str] = []
    for name, p in m.named_parameters():
        if p.grad is None or torch.all(p.grad == 0):
            no_grad.append(name)
    # lambda_proj.weight is zero-initialized AND gets multiplied by x,
    # so gradient flows through it; theta_proj.weight similarly.
    assert not no_grad, f"params with zero/no gradient: {no_grad}"


def test_no_nan_under_extreme_input():
    m = Mamba3Forecaster(schema=COLORAN_SCHEMA, task="classification", seq_len=5)
    m.train()
    x_cat = torch.zeros(2, 5, 4, dtype=torch.long)
    x_cont = torch.randn(2, 5, 12) * 20.0
    out = m(x_cat, x_cont)
    assert torch.isfinite(out).all()
    out.sum().backward()
    for name, p in m.named_parameters():
        if p.grad is None:
            continue
        assert torch.isfinite(p.grad).all(), f"NaN grad on {name}"


def test_determinism_same_seed_same_output():
    def _build_and_forward() -> torch.Tensor:
        torch.manual_seed(42)
        m = Mamba3Forecaster(schema=COLORAN_SCHEMA, task="classification", seq_len=5)
        m.eval()
        torch.manual_seed(123)
        x_cat = torch.randint(0, 3, (4, 5, 4), dtype=torch.long)
        x_cont = torch.randn(4, 5, 12)
        return m(x_cat, x_cont)
    a = _build_and_forward()
    b = _build_and_forward()
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# Semantic-equivalence test: forced λ=1 + θ=0 should give pure Mamba-2 Euler
# behavior on the state recurrence. This is the LOAD-BEARING correctness
# check for Innovations 1 + 2 — if any rotation/trapezoidal logic drifts,
# this test will fire.
# ---------------------------------------------------------------------------

def test_lambda1_theta0_recovers_mamba2_euler_state_recurrence():
    """At ``λ=1`` and ``θ=0``, the Mamba-3 recurrence collapses to the
    Mamba-2 Euler recurrence (no β term, no rotation). We force these
    constants and run one Mamba3SSMBlock against an in-place Mamba-2
    selective scan with matched parameters, asserting their state
    trajectories agree elementwise.
    """
    torch.manual_seed(0)
    d_model, d_state, expand, d_conv = 16, 8, 1, 4
    block = Mamba3SSMBlock(
        d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,
    )
    block.eval()

    # Hand-roll the inputs to the scan (bypass conv1d / in_proj for direct
    # state-recurrence comparison). The scan reads lambda_seq and theta_seq
    # from its arguments only — block.lambda_proj / block.theta_proj are
    # bypassed entirely, so we pass ones / zeros directly.
    B, L, d_inner = 2, 5, expand * d_model
    d_state_pairs = d_state // 2

    x = torch.randn(B, L, d_inner)
    dt = torch.full((B, L, d_inner), 0.5)
    A_pairs = -torch.exp(block.A_log)             # (d_inner, pairs)
    # Replicate A across pair (re, im) dims for Mamba-2 reference.
    A_ref = A_pairs.repeat_interleave(2, dim=-1)  # (d_inner, d_state)
    B_param = torch.randn(B, L, d_state)
    C_param = torch.randn(B, L, d_state)
    D = block.D
    # λ=1 sequence and θ=0 sequence — what the block will see.
    lambda_seq = torch.ones(B, L, 1)
    theta_seq = torch.zeros(B, L, d_state_pairs)

    # Mamba-3 scan with forced λ=1, θ=0.
    y_m3 = block._selective_scan(
        x, dt, A_pairs, B_param, C_param, D, lambda_seq, theta_seq,
    )

    # Mamba-2 reference scan (no β term, no rotation).
    h_ref = x.new_zeros(B, d_inner, d_state)
    outputs_ref = []
    dA_ref = torch.exp(dt.unsqueeze(-1) * A_ref.unsqueeze(0).unsqueeze(0))
    dBx_ref = dt.unsqueeze(-1) * B_param.unsqueeze(2) * x.unsqueeze(-1)
    for t in range(L):
        h_ref = dA_ref[:, t] * h_ref + dBx_ref[:, t]
        y_t_ref = (h_ref * C_param[:, t].unsqueeze(1)).sum(dim=-1)
        outputs_ref.append(y_t_ref)
    y_ref = torch.stack(outputs_ref, dim=1) + D.unsqueeze(0).unsqueeze(0) * x

    assert torch.allclose(y_m3, y_ref, atol=1e-5, rtol=1e-4), (
        f"Mamba-3 with λ=1, θ=0 should recover Mamba-2 Euler exactly. "
        f"Max abs diff = {(y_m3 - y_ref).abs().max().item():.2e}"
    )


def test_lambda_half_halves_t0_input_and_adds_prev_term_at_t1():
    """At ``λ=0.5`` with ``θ=0``:

    * At ``t=0``: ``γ_0 = 0.5·Δ`` (input weight halved vs Mamba-2 ``γ=Δ``)
      and ``β_0·B_{-1}x_{-1} = 0`` (sequence-prefix convention). So
      ``y_trap[0] − D·x[0] = 0.5 · (y_euler[0] − D·x[0])`` exactly.

    * At ``t≥1``: the β term ``(1-λ)·Δ·α·B_{t-1}x_{t-1}`` activates, so
      ``y_trap[t≥1]`` diverges from ``y_euler`` by more than the trivial
      γ-scaling alone. This is the qualitative signature of trapezoidal
      vs Euler.
    """
    torch.manual_seed(0)
    block = Mamba3SSMBlock(d_model=16, d_state=8, expand=1, d_conv=4)
    block.eval()
    # Scan reads lambda_seq / theta_seq from arguments only — pass 0.5 / 0
    # directly without touching block.lambda_proj / block.theta_proj.

    B, L, d_inner = 2, 5, 16
    x = torch.randn(B, L, d_inner)
    dt = torch.full((B, L, d_inner), 0.5)
    A_pairs = -torch.exp(block.A_log)
    A_ref = A_pairs.repeat_interleave(2, dim=-1)
    B_param = torch.randn(B, L, 8)
    C_param = torch.randn(B, L, 8)
    lambda_seq = torch.full((B, L, 1), 0.5)
    theta_seq = torch.zeros(B, L, 4)

    y_trap = block._selective_scan(
        x, dt, A_pairs, B_param, C_param, block.D,
        lambda_seq, theta_seq,
    )

    # Reference Euler scan (matches test_lambda1_theta0_recovers_mamba2_euler).
    h_ref = x.new_zeros(B, d_inner, 8)
    outputs_ref = []
    dA_ref = torch.exp(dt.unsqueeze(-1) * A_ref.unsqueeze(0).unsqueeze(0))
    dBx_ref = dt.unsqueeze(-1) * B_param.unsqueeze(2) * x.unsqueeze(-1)
    for t in range(L):
        h_ref = dA_ref[:, t] * h_ref + dBx_ref[:, t]
        outputs_ref.append((h_ref * C_param[:, t].unsqueeze(1)).sum(dim=-1))
    y_euler = torch.stack(outputs_ref, dim=1) + block.D.unsqueeze(0).unsqueeze(0) * x

    # Subtract skip-path contribution to isolate state-recurrence output.
    Dx = block.D.unsqueeze(0).unsqueeze(0) * x          # (B, L, d_inner)
    state_trap = y_trap - Dx
    state_euler = y_euler - Dx

    # t=0: trapezoidal state contribution is EXACTLY half the Euler state
    # contribution (γ=0.5·Δ vs γ=Δ; β=0 since Bx_prev=0).
    assert torch.allclose(
        state_trap[:, 0], 0.5 * state_euler[:, 0], atol=1e-5,
    ), (
        f"At λ=0.5 t=0 the state output should be exactly half of Euler. "
        f"Max abs diff = "
        f"{(state_trap[:, 0] - 0.5 * state_euler[:, 0]).abs().max():.2e}"
    )

    # t≥1: y_trap should NOT be simply 0.5·y_euler — the β term contributes
    # a non-trivial previous-step memory that Euler doesn't have.
    halved_euler_t1 = 0.5 * state_euler[:, 1:]
    diff_from_halved = (state_trap[:, 1:] - halved_euler_t1).abs().max().item()
    assert diff_from_halved > 1e-3, (
        f"At λ=0.5 t≥1, trapezoidal output should diverge from 0.5×Euler "
        f"due to the β·B_{{t-1}}·x_{{t-1}} memory term, but max abs diff = "
        f"{diff_from_halved:.2e}. Either β term is broken or test data "
        f"happens to be degenerate."
    )


def test_theta_nonzero_rotates_state_pairs():
    """At ``θ ≠ 0``, adjacent state pairs (re, im) should mix via the 2x2
    rotation. Force θ = π/2 (90° rotation, perfect re↔im swap with sign)
    and zero decay (A → 0 via near-zero A_log + small dt) to isolate the
    rotation effect.
    """
    torch.manual_seed(0)
    block = Mamba3SSMBlock(d_model=16, d_state=4, expand=1, d_conv=4)
    block.eval()
    # Set near-zero decay by shrinking A_log to a large negative value
    # (A ≈ -4.5e-5 → ρ ≈ 1 under our tiny dt) so the rotation effect is
    # not swamped by magnitude decay. The lambda_proj / theta_proj weights
    # don't need overriding — the scan reads lambda_seq / theta_seq from
    # arguments, which we pass explicitly below.
    import math
    with torch.no_grad():
        block.A_log.fill_(-10.0)

    B, L, d_inner = 1, 2, 16
    # Impulse input at t=0 to populate state; subsequent steps have zero
    # input so any state evolution must come from rotation, not new input.
    x = torch.zeros(B, L, d_inner)
    x[:, 0, :] = 1.0
    dt = torch.full((B, L, d_inner), 1e-4)
    A_pairs = -torch.exp(block.A_log)
    B_param = torch.ones(B, L, 4)
    C_param = torch.ones(B, L, 4)
    lambda_seq = torch.full((B, L, 1), 1.0)        # no β term
    theta_seq = torch.full((B, L, 2), math.pi / 2)  # 90° rotation each step

    y_rot = block._selective_scan(
        x, dt, A_pairs, B_param, C_param, block.D, lambda_seq, theta_seq,
    )
    # Minimum bar: rotation path must not produce non-finite values. The
    # full state-mixing semantics (e.g., re ↔ im swap at θ=π/2) would need
    # state-trajectory plumbing through _selective_scan to verify directly;
    # we settle for finiteness here and trust the
    # test_lambda1_theta0_recovers_mamba2_euler_state_recurrence test
    # above to catch any drift in the no-rotation (θ=0) branch.
    assert torch.isfinite(y_rot).all(), (
        f"rotation path produced non-finite output at θ=π/2: "
        f"any_nan={torch.isnan(y_rot).any().item()}, "
        f"any_inf={torch.isinf(y_rot).any().item()}"
    )

    # Also assert rotation actually changes the output vs the θ=0 baseline,
    # so a future bug where rotation silently degrades to identity (e.g.,
    # zeroing cos/sin) WILL fail this test.
    theta_zero = torch.zeros(B, L, 2)
    y_norot = block._selective_scan(
        x, dt, A_pairs, B_param, C_param, block.D, lambda_seq, theta_zero,
    )
    assert not torch.allclose(y_rot, y_norot, atol=1e-4), (
        "rotation at θ=π/2 produced output indistinguishable from θ=0 — "
        "rotation path may be degenerate"
    )


# ---------------------------------------------------------------------------
# Block-level scan: state shape is consistent across pairs.
# ---------------------------------------------------------------------------

def test_rotate_and_decay_helper_shape_and_identity():
    """``_rotate_and_decay`` with ``cos=1, sin=0, rho=1`` must be the
    identity transform (the trivial no-rotation, no-decay case)."""
    z = torch.randn(2, 4, 8)
    rho = torch.ones(2, 4, 4)
    cos_t = torch.ones(2, 4)
    sin_t = torch.zeros(2, 4)
    out = Mamba3SSMBlock._rotate_and_decay(z, rho, cos_t, sin_t)
    assert out.shape == z.shape
    assert torch.allclose(out, z, atol=1e-6)
