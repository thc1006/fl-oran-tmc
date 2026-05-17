"""Tests for :class:`fl_oran.models.xlstm_forecaster.xLSTMForecaster`.

Covers param-count parity with ForecasterV2 (ADR-001 D-20: ±10%), forward
shape on the ColO-RAN production schema, the sLSTM stabilizer's numerical
safety under extreme inputs (paper eq 15-17 guarantees), gradient flow,
and determinism.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from fl_oran.data_v2.encoders import FeatureSchema
from fl_oran.models.forecaster_v2 import ForecasterV2
from fl_oran.models.xlstm_forecaster import sLSTMCell, xLSTMForecaster


COLORAN_SCHEMA = FeatureSchema(
    categorical=["bs_id", "slice_id", "sched", "tr"],
    categorical_sizes={"bs_id": 7, "slice_id": 3, "sched": 5, "tr": 28},
    continuous=[f"c{i}" for i in range(12)],
)


def _count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_param_count_in_v6_expected_range():
    """xLSTMForecaster should sit in [40K, 50K] like ForecasterV2."""
    m = xLSTMForecaster(schema=COLORAN_SCHEMA, task="classification", seq_len=5)
    n = _count(m)
    assert 40_000 <= n <= 50_000, (
        f"xLSTMForecaster param count {n} is outside [40K, 50K]; "
        f"tune hidden_size or n_layers."
    )


def test_param_count_within_10pct_of_forecasterv2():
    """ADR-001 D-20 parity rule: capacity must match the LSTM baseline."""
    base = ForecasterV2(schema=COLORAN_SCHEMA, task="classification", seq_len=5)
    xl = xLSTMForecaster(schema=COLORAN_SCHEMA, task="classification", seq_len=5)
    n_base, n_xl = _count(base), _count(xl)
    drift = n_xl / n_base - 1.0
    assert abs(drift) <= 0.10, (
        f"xLSTMForecaster has {n_xl} params vs ForecasterV2 {n_base} "
        f"(drift {drift * 100:+.1f}%); tune hidden_size or n_layers."
    )


def test_forward_shape_classification():
    m = xLSTMForecaster(schema=COLORAN_SCHEMA, task="classification", seq_len=5)
    m.eval()
    x_cat = torch.zeros(8, 5, 4, dtype=torch.long)
    x_cont = torch.randn(8, 5, 12)
    out = m(x_cat, x_cont)
    assert out.shape == (8, 1), out.shape
    assert torch.isfinite(out).all()


def test_forward_shape_regression_with_persistence():
    """Persistence-baseline path should add x_cont[:, -1, persistence_idx]."""
    m = xLSTMForecaster(
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
    # task='regression' zero-inits the head, so the trunk contributes 0 and
    # the output should equal the persistence baseline exactly.
    assert torch.allclose(out, torch.full((4, 1), 0.7), atol=1e-6), out


def test_stabilizer_no_nan_under_extreme_input():
    """sLSTMCell paper eq 15-17 guarantees exp(x - m) ≤ 1. Verify NaN-free
    forward + backward when inputs are extreme."""
    cell = sLSTMCell(input_size=16, hidden_size=16)
    state = cell.init_state(4, torch.device("cpu"))
    # Extreme positive input: would overflow `exp(ĩ)` without stabilizer.
    x = torch.full((4, 16), 50.0, requires_grad=True)
    h, new_state = cell(x, state)
    assert torch.isfinite(h).all(), "stabilizer failed under extreme input"
    assert all(torch.isfinite(s).all() for s in new_state)
    loss = h.sum()
    loss.backward()
    assert torch.isfinite(x.grad).all(), "gradient NaN under extreme input"


def test_stabilizer_handles_long_sequence_in_full_model():
    """End-to-end model with extreme continuous inputs must not NaN."""
    m = xLSTMForecaster(schema=COLORAN_SCHEMA, task="classification", seq_len=5)
    m.train()
    x_cat = torch.zeros(2, 5, 4, dtype=torch.long)
    # Extreme continuous values that would overflow without the stabilizer.
    x_cont = torch.randn(2, 5, 12) * 20.0
    out = m(x_cat, x_cont)
    assert torch.isfinite(out).all()
    loss = out.sum()
    loss.backward()
    for name, p in m.named_parameters():
        if p.grad is None:
            continue
        assert torch.isfinite(p.grad).all(), f"NaN grad on {name}"


def test_gradient_flow_to_all_params():
    """Every learnable param should receive a non-zero gradient after one
    backward pass with non-degenerate inputs."""
    torch.manual_seed(0)
    m = xLSTMForecaster(schema=COLORAN_SCHEMA, task="classification", seq_len=5)
    m.train()
    x_cat = torch.randint(0, 3, (4, 5, 4), dtype=torch.long)
    x_cont = torch.randn(4, 5, 12)
    out = m(x_cat, x_cont)
    loss = out.sum()
    loss.backward()
    no_grad: list[str] = []
    for name, p in m.named_parameters():
        if p.grad is None or torch.all(p.grad == 0):
            no_grad.append(name)
    assert not no_grad, f"params with no/zero gradient: {no_grad}"


def test_determinism_same_seed_same_output():
    """Identical seed + identical input → bitwise-equal forward output."""
    def _build_and_forward() -> torch.Tensor:
        torch.manual_seed(42)
        m = xLSTMForecaster(schema=COLORAN_SCHEMA, task="classification", seq_len=5)
        m.eval()
        torch.manual_seed(123)
        x_cat = torch.randint(0, 3, (4, 5, 4), dtype=torch.long)
        x_cont = torch.randn(4, 5, 12)
        return m(x_cat, x_cont)

    a = _build_and_forward()
    b = _build_and_forward()
    assert torch.equal(a, b), "xLSTMForecaster forward is non-deterministic"


def test_drop_categorical_changes_param_count():
    """The D-20 R2 ablation surface: dropping a categorical column should
    remove the corresponding embedding params + shrink in_proj input_dim."""
    base = xLSTMForecaster(schema=COLORAN_SCHEMA, task="classification", seq_len=5)
    dropped = xLSTMForecaster(
        schema=COLORAN_SCHEMA,
        task="classification",
        seq_len=5,
        drop_categorical=["tr"],
    )
    assert _count(dropped) < _count(base)
    # 'tr' embedding (29 × 8) plus reduced in_proj input_dim (44 → 36, 8 fewer
    # cols × 48 hidden = 384 fewer in_proj params).
    delta = _count(base) - _count(dropped)
    assert delta > 29 * 8, f"unexpected param-count delta {delta}"


def test_n_safe_uses_exp_minus_m_not_one():
    """Regression test for the stabilizer-clamp bug (n_safe must be
    ``max(|n|, exp(-m))``, not ``max(|n|, 1)``). Construct a state where
    m_new is forced negative so exp(-m_new) > 1; if the clamp used the
    constant 1, the hidden output would differ from the corrected formula.
    """
    cell = sLSTMCell(input_size=4, hidden_size=4)
    # Zero out all recurrent weights so we can predict m_new exactly from
    # the input projection.
    with torch.no_grad():
        for p in cell.parameters():
            p.zero_()
        # Bias setup: f̃ = 0 (sigmoid → 0.5, log → ~-0.69),
        # ĩ = -2 (so log_i = -2 → m_new = max(-0.69, -2) = -0.69, NOT 0).
        cell.w_i.bias.fill_(-2.0)
    state = cell.init_state(2, torch.device("cpu"))
    x = torch.zeros(2, 4)
    h, (_, c, n, m) = cell(x, state)

    expected_m = torch.full_like(m, torch.log(torch.tensor(0.5)).item())
    assert torch.allclose(m, expected_m, atol=1e-5), m
    # exp(-m_new) ≈ 2.0 > 1.0, so the bug-vs-fix divergence is detectable.
    expected_threshold = torch.exp(-expected_m)
    assert (expected_threshold > 1.0).all()

    expected_n_safe = torch.maximum(n.abs(), expected_threshold)
    # Cell uses sigmoid for output gate (= 0.5 here since o_tilde=0).
    expected_h = 0.5 * (c / expected_n_safe)
    assert torch.allclose(h, expected_h, atol=1e-6), (
        f"h mismatch — stabilizer bug regression? h={h}, expected={expected_h}"
    )


def test_stabilizer_equivalent_to_unstabilized_when_m_is_zero():
    """When m_t = 0, the stabilized recurrence collapses to the unstabilized
    paper eq 8-10 form exactly. We force m=0 by choosing inputs where
    log(i) ≤ 0 and log(f) + m_prev ≤ 0 — so m stays 0 — and check that
    h matches the direct unstabilized computation.
    """
    cell = sLSTMCell(input_size=4, hidden_size=4)
    with torch.no_grad():
        for p in cell.parameters():
            p.zero_()
        cell.w_i.bias.fill_(0.0)   # log(i) = 0 → m_new = max(log(0.5), 0) = 0
        cell.w_f.bias.fill_(0.0)   # log(f) = log(0.5) ≈ -0.69
        cell.w_z.bias.fill_(0.0)   # z = tanh(0) = 0
        cell.w_o.bias.fill_(0.0)   # o = sigmoid(0) = 0.5

    state = cell.init_state(2, torch.device("cpu"))
    x = torch.zeros(2, 4)
    h, (_, c, n, m) = cell(x, state)

    assert torch.allclose(m, torch.zeros_like(m), atol=1e-6), m
    # At m=0, stabilized = unstabilized: i = exp(0) = 1, f = sigmoid(0) = 0.5.
    # c_new = 0.5 * 0 + 1 * 0 = 0
    # n_new = 0.5 * 0 + 1 = 1
    # h_new = 0.5 * (0 / max(1, exp(0))) = 0.5 * 0 / 1 = 0
    assert torch.allclose(c, torch.zeros_like(c), atol=1e-6)
    assert torch.allclose(n, torch.ones_like(n), atol=1e-6)
    assert torch.allclose(h, torch.zeros_like(h), atol=1e-6)


def test_initial_state_is_zero_n_and_distinct_tensors():
    """Bug-regression: n_0 must be 0 per paper eq 9 (n_t = f·n_{t-1} + i_t,
    initialized at 0). Also assert init_state returns four distinct tensors
    so that no aliasing surprises bite under torch.compile or in-place ops.
    """
    cell = sLSTMCell(input_size=4, hidden_size=4)
    h, c, n, m = cell.init_state(2, torch.device("cpu"))
    assert torch.all(h == 0)
    assert torch.all(c == 0)
    assert torch.all(n == 0), f"n_0 should be zero per paper eq 9, got {n}"
    assert torch.all(m == 0)
    # Tensor-identity check: distinct .data_ptr() means distinct memory.
    ptrs = {h.data_ptr(), c.data_ptr(), n.data_ptr(), m.data_ptr()}
    assert len(ptrs) == 4, "init_state aliased two or more state tensors"


def test_invalid_forget_gate_raises():
    import pytest
    with pytest.raises(ValueError, match="forget_gate"):
        sLSTMCell(input_size=8, hidden_size=8, forget_gate="tanh")


def test_invalid_drop_categorical_raises():
    import pytest
    with pytest.raises(ValueError, match="drop_categorical"):
        xLSTMForecaster(
            schema=COLORAN_SCHEMA,
            task="classification",
            seq_len=5,
            drop_categorical=["nonexistent_col"],
        )


def test_invalid_persistence_feature_raises():
    import pytest
    with pytest.raises(ValueError, match="persistence_feature"):
        xLSTMForecaster(
            schema=COLORAN_SCHEMA,
            task="regression",
            seq_len=5,
            persistence_feature="not_in_schema",
        )
