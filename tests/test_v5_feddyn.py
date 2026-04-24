"""TDD tests for FedDyn (Acar et al. 2021, ICLR) — ADR-001 §3.6.

FedDyn augments each client's local objective with a linear term
``-<h_i, w>`` and a quadratic penalty ``(alpha/2)*||w - w_t||^2``. The
gradient correction applied every local step is ``-h_i + alpha*(w - w_t)``,
and at the end of training ``h_i <- h_i - alpha*(w_l - w_t)``. delta_h_i is
reported via ``ClientUpdate.aux`` so the server can keep an accumulator.

Tests:
1. Registered.
2. ``alpha=0`` + first round (h_i=0) ==> grad correction is identically
   zero ==> trajectory is bit-identical to FedAvg.
3. ``ClientUpdate.aux`` contains ``delta_h_i``.
4. Per-client ``h_i`` state persists across rounds.
5. Server's ``h_accum`` accumulates.
"""
from __future__ import annotations

import torch

from conftest import build_trio as _build_trio


def test_feddyn_registered():
    from fl_oran.federated.algorithms import REGISTRY
    assert "feddyn" in REGISTRY
    assert REGISTRY["feddyn"].name == "feddyn"


def test_feddyn_alpha_zero_first_round_matches_fedavg():
    """alpha=0 + h_i=0 ⇒ grad correction is 0 ⇒ identical to FedAvg."""
    from fl_oran.federated.algorithms import REGISTRY
    model_a, tensors, loss_fn = _build_trio(seed=42)
    model_b, _, _ = _build_trio(seed=42)
    fedavg = REGISTRY["fedavg"](max_steps=5, batch_size=4, grad_clip=1.0)
    feddyn = REGISTRY["feddyn"](max_steps=5, batch_size=4, grad_clip=1.0, alpha=0.0)
    device = torch.device("cpu")
    torch.manual_seed(9)
    u_a = fedavg.client_update(
        client_id=1, local_model=model_a, client_tensors=tensors,
        loss_fn=loss_fn, current_lr=0.01, device=device, round_idx=1,
    )
    torch.manual_seed(9)
    u_b = feddyn.client_update(
        client_id=1, local_model=model_b, client_tensors=tensors,
        loss_fn=loss_fn, current_lr=0.01, device=device, round_idx=1,
    )
    for k in u_a.state_dict:
        torch.testing.assert_close(u_a.state_dict[k], u_b.state_dict[k])


def test_feddyn_client_returns_aux_delta_h_i():
    from fl_oran.federated.algorithms import REGISTRY
    model, tensors, loss_fn = _build_trio(seed=42)
    feddyn = REGISTRY["feddyn"](max_steps=5, batch_size=4, alpha=0.01)
    u = feddyn.client_update(
        client_id=1, local_model=model, client_tensors=tensors,
        loss_fn=loss_fn, current_lr=0.01, device=torch.device("cpu"), round_idx=1,
    )
    assert "delta_h_i" in u.aux
    assert len(u.aux["delta_h_i"]) > 0
    any_nonzero = any(float(v.abs().sum()) > 0 for v in u.aux["delta_h_i"].values())
    assert any_nonzero, "expected nonzero delta_h_i after training with alpha > 0"


def test_feddyn_client_h_i_persists():
    from fl_oran.federated.algorithms import REGISTRY
    model_1, tensors, loss_fn = _build_trio(seed=42)
    model_2, _, _ = _build_trio(seed=42)
    feddyn = REGISTRY["feddyn"](max_steps=3, batch_size=4, alpha=0.01)
    device = torch.device("cpu")
    _ = feddyn.client_update(
        client_id=5, local_model=model_1, client_tensors=tensors,
        loss_fn=loss_fn, current_lr=0.01, device=device, round_idx=1,
    )
    assert 5 in feddyn.h_i
    snap = {k: v.clone() for k, v in feddyn.h_i[5].items()}
    _ = feddyn.client_update(
        client_id=5, local_model=model_2, client_tensors=tensors,
        loss_fn=loss_fn, current_lr=0.01, device=device, round_idx=2,
    )
    assert any(
        not torch.allclose(feddyn.h_i[5][k], snap[k]) for k in snap
    ), "h_i should evolve between rounds"


def test_feddyn_server_h_accum_accumulates():
    from fl_oran.federated.algorithms import REGISTRY
    model, tensors, loss_fn = _build_trio(seed=42)
    feddyn = REGISTRY["feddyn"](max_steps=3, batch_size=4, alpha=0.01)
    u = feddyn.client_update(
        client_id=1, local_model=model, client_tensors=tensors,
        loss_fn=loss_fn, current_lr=0.01, device=torch.device("cpu"), round_idx=1,
    )
    global_state = {k: v.clone() for k, v in u.state_dict.items()}
    _ = feddyn.server_aggregate(global_state=global_state, updates=[u])
    assert feddyn.h_accum is not None and len(feddyn.h_accum) > 0
    any_nonzero = any(float(v.abs().sum()) > 0 for v in feddyn.h_accum.values())
    assert any_nonzero, "h_accum should accumulate after the first round"
