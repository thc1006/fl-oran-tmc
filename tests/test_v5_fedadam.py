"""TDD tests for FedAdam (Reddi et al. 2020, ICLR) — ADR-001 §3.4.

FedAdam = FedOpt with Adam at server. Client is identical to FedAvg; the
difference is the server-side update:

    delta = sum_i p_i * (w_i - w_global)
    m     = beta1 * m + (1 - beta1) * delta
    v     = beta2 * v + (1 - beta2) * delta^2
    w_new = w_global + server_lr * m / (sqrt(v) + tau)

Tests:
1. Registered.
2. Missing ``server_lr`` raises.
3. Client output matches FedAvg bit-wise (FedAdam inherits FedAvg's client).
4. Server with beta1=beta2=0, lr=1.0, small tau reduces to ``w + sign(delta)``
   — distinguishes FedAdam from FedAvg's plain averaging.
5. Moment buffers persist across calls (second call reuses prior m/v).
"""
from __future__ import annotations

import pytest
import torch

from conftest import build_trio as _build_trio


def test_fedadam_registered():
    from fl_oran.federated.algorithms import REGISTRY
    assert "fedadam" in REGISTRY
    assert REGISTRY["fedadam"].name == "fedadam"


def test_fedadam_requires_server_lr():
    from fl_oran.federated.algorithms import REGISTRY
    cls = REGISTRY["fedadam"]
    with pytest.raises(TypeError, match="server_lr"):
        cls(max_steps=1, batch_size=1)


def test_fedadam_client_matches_fedavg():
    from fl_oran.federated.algorithms import REGISTRY
    model_a, tensors, loss_fn = _build_trio(seed=42)
    model_b, _, _ = _build_trio(seed=42)
    fedavg = REGISTRY["fedavg"](max_steps=5, batch_size=4, grad_clip=1.0)
    fedadam = REGISTRY["fedadam"](max_steps=5, batch_size=4, grad_clip=1.0,
                                   server_lr=0.01)
    device = torch.device("cpu")
    torch.manual_seed(11)
    u_a = fedavg.client_update(
        client_id=1, local_model=model_a, client_tensors=tensors,
        loss_fn=loss_fn, current_lr=0.01, device=device, round_idx=1,
    )
    torch.manual_seed(11)
    u_b = fedadam.client_update(
        client_id=1, local_model=model_b, client_tensors=tensors,
        loss_fn=loss_fn, current_lr=0.01, device=device, round_idx=1,
    )
    for k in u_a.state_dict:
        torch.testing.assert_close(u_a.state_dict[k], u_b.state_dict[k])


def test_fedadam_server_beta_zero_reduces_to_sign_delta():
    """beta1=beta2=0, server_lr=1.0, tau -> 0: w_new = w + delta/|delta| = w + sign(delta)."""
    from fl_oran.federated.algorithms import REGISTRY
    from fl_oran.federated.client import ClientUpdate

    algo = REGISTRY["fedadam"](max_steps=1, batch_size=1, server_lr=1.0,
                                beta1=0.0, beta2=0.0, tau=1e-8)
    global_state = {"w": torch.zeros(3)}
    u = ClientUpdate(
        client_id=1,
        state_dict={"w": torch.tensor([2.0, -3.0, 0.5])},
        num_examples=10,
        train_loss=0.0,
    )
    new_state = algo.server_aggregate(global_state=global_state, updates=[u])
    expected = torch.tensor([1.0, -1.0, 1.0])  # sign of the delta
    torch.testing.assert_close(new_state["w"], expected, atol=1e-4, rtol=1e-4)


def test_fedadam_bias_correction_t1_amplifies_update():
    """At t=1 with beta=0.9 and bias_correction=True, m_hat = m / (1 - 0.9)
    = 10 * m = 10 * 0.1 * Delta = Delta (vs m = 0.1*Delta without bias).
    So the round-1 update magnitude is ~10x larger with bias correction.
    """
    from fl_oran.federated.algorithms import REGISTRY
    from fl_oran.federated.client import ClientUpdate

    no_bias = REGISTRY["fedadam"](
        max_steps=1, batch_size=1, server_lr=0.01, beta1=0.9, beta2=0.99,
        tau=1e-3, bias_correction=False,
    )
    with_bias = REGISTRY["fedadam"](
        max_steps=1, batch_size=1, server_lr=0.01, beta1=0.9, beta2=0.99,
        tau=1e-3, bias_correction=True,
    )
    gs = {"w": torch.zeros(3)}
    u = ClientUpdate(
        client_id=1, state_dict={"w": torch.tensor([1.0, 1.0, 1.0])},
        num_examples=1, train_loss=0.0,
    )
    a = no_bias.server_aggregate(global_state=gs, updates=[u])
    b = with_bias.server_aggregate(global_state=gs, updates=[u])
    # With bias correction, update magnitude is larger at t=1.
    assert b["w"].abs().sum() > a["w"].abs().sum(), (
        "bias_correction=True should yield a larger t=1 update than False"
    )


def test_fedadam_moments_persist_across_rounds():
    from fl_oran.federated.algorithms import REGISTRY
    from fl_oran.federated.client import ClientUpdate

    algo = REGISTRY["fedadam"](max_steps=1, batch_size=1, server_lr=0.1)
    global_state = {"w": torch.zeros(3)}
    u = ClientUpdate(
        client_id=1, state_dict={"w": torch.tensor([1.0, 2.0, 3.0])},
        num_examples=1, train_loss=0.0,
    )
    # First call: should populate m, v for "w"
    _ = algo.server_aggregate(global_state=global_state, updates=[u])
    assert "w" in algo.m, "FedAdam.m should have 'w' after first aggregate"
    assert "w" in algo.v
    m_after_round1 = algo.m["w"].clone()
    v_after_round1 = algo.v["w"].clone()
    # Second call with same update: m, v should move (beta*prev + (1-beta)*new)
    _ = algo.server_aggregate(global_state=global_state, updates=[u])
    assert not torch.allclose(algo.m["w"], m_after_round1), "m should evolve"
    assert not torch.allclose(algo.v["w"], v_after_round1), "v should evolve"
