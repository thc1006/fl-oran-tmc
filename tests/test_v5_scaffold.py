"""TDD tests for SCAFFOLD (Karimireddy et al. 2020, ICML) — ADR-001 §3.5.

SCAFFOLD carries a global control variate ``c`` and per-client ``c_i`` across
rounds. Client training adds ``c - c_i`` as a variance-reduction correction
to every gradient step. At the end of local training the client computes a
new ``c_i+`` (Option I from the paper): ``c_i - c + (w_g - w_l)/(K·η)``, and
reports ``Δc_i = c_i+ - c_i`` via ``ClientUpdate.aux``. The server aggregates
both the weights (plain FedAvg) and the control-variate delta (mean of Δc_i).

Tests:
1. Registered.
2. First round with ``c=c_i=0`` ⇒ client_update reduces to FedAvg trajectory
   (no grad correction when both control variates are zero).
3. ``aux["delta_c_i"]`` is non-empty after a client_update.
4. After ``server_aggregate``, the algorithm's global ``c`` accumulates
   non-zero entries.
5. Per-client ``c_i`` state persists across rounds (second call sees
   non-zero ``c_i``).
"""
from __future__ import annotations

import copy

import torch
from torch import nn


def _build_trio(seed: int = 42, n: int = 64, seq_len: int = 3,
                n_cat: int = 2, n_cont: int = 2):
    torch.manual_seed(seed)

    class TinyDualInput(nn.Module):
        def __init__(self, n_in: int) -> None:
            super().__init__()
            self.linear = nn.Linear(n_in, 1)

        def forward(self, cat: torch.Tensor, cont: torch.Tensor) -> torch.Tensor:
            cat_f = cat.float().mean(dim=1)
            cont_f = cont.mean(dim=1)
            return self.linear(torch.cat([cat_f, cont_f], dim=-1))

    model = TinyDualInput(n_cat + n_cont)
    cat = torch.randint(0, 5, (n, seq_len, n_cat), dtype=torch.long)
    cont = torch.randn(n, seq_len, n_cont, dtype=torch.float32)
    y = torch.randint(0, 2, (n, 1)).float()
    return model, (cat, cont, y), nn.BCEWithLogitsLoss()


def test_scaffold_registered():
    from fl_oran.federated.algorithms import REGISTRY
    assert "scaffold" in REGISTRY
    assert REGISTRY["scaffold"].name == "scaffold"


def test_scaffold_first_round_reduces_to_fedavg():
    """Round 0: c=0, c_i=0 ⇒ grad correction is 0 ⇒ bit-identical to FedAvg."""
    from fl_oran.federated.algorithms import REGISTRY
    model_a, tensors, loss_fn = _build_trio(seed=42)
    model_b, _, _ = _build_trio(seed=42)
    fedavg = REGISTRY["fedavg"](max_steps=5, batch_size=4, grad_clip=1.0)
    scaffold = REGISTRY["scaffold"](max_steps=5, batch_size=4, grad_clip=1.0)
    device = torch.device("cpu")
    torch.manual_seed(7)
    u_a = fedavg.client_update(
        client_id=1, local_model=model_a, client_tensors=tensors,
        loss_fn=loss_fn, current_lr=0.01, device=device, round_idx=1,
    )
    torch.manual_seed(7)
    u_b = scaffold.client_update(
        client_id=1, local_model=model_b, client_tensors=tensors,
        loss_fn=loss_fn, current_lr=0.01, device=device, round_idx=1,
    )
    for k in u_a.state_dict:
        torch.testing.assert_close(u_a.state_dict[k], u_b.state_dict[k])


def test_scaffold_client_returns_aux_delta_c_i():
    from fl_oran.federated.algorithms import REGISTRY
    model, tensors, loss_fn = _build_trio(seed=42)
    scaffold = REGISTRY["scaffold"](max_steps=5, batch_size=4)
    u = scaffold.client_update(
        client_id=1, local_model=model, client_tensors=tensors,
        loss_fn=loss_fn, current_lr=0.01, device=torch.device("cpu"), round_idx=1,
    )
    assert "delta_c_i" in u.aux, "ClientUpdate.aux must contain delta_c_i"
    assert len(u.aux["delta_c_i"]) > 0, "delta_c_i should have per-parameter entries"
    # At least one tensor should be nonzero (training moved the weights).
    any_nonzero = any(
        float(v.abs().sum()) > 0 for v in u.aux["delta_c_i"].values()
    )
    assert any_nonzero, "delta_c_i is identically zero — did client_update actually train?"


def test_scaffold_server_aggregates_control_variate():
    from fl_oran.federated.algorithms import REGISTRY
    model, tensors, loss_fn = _build_trio(seed=42)
    scaffold = REGISTRY["scaffold"](max_steps=5, batch_size=4)

    u = scaffold.client_update(
        client_id=1, local_model=model, client_tensors=tensors,
        loss_fn=loss_fn, current_lr=0.01, device=torch.device("cpu"), round_idx=1,
    )
    global_state = {k: v.clone() for k, v in u.state_dict.items()}
    # Server's c should be empty / None before aggregate (or all zeros).
    _ = scaffold.server_aggregate(global_state=global_state, updates=[u])
    assert scaffold.c is not None and len(scaffold.c) > 0, \
        "scaffold.c must be populated after first server_aggregate"
    any_nonzero = any(float(v.abs().sum()) > 0 for v in scaffold.c.values())
    assert any_nonzero, "global c should accumulate nonzero after the first round"


def test_scaffold_client_c_i_persists():
    from fl_oran.federated.algorithms import REGISTRY
    model_1, tensors, loss_fn = _build_trio(seed=42)
    model_2, _, _ = _build_trio(seed=42)
    scaffold = REGISTRY["scaffold"](max_steps=3, batch_size=4)
    device = torch.device("cpu")
    # First round
    _ = scaffold.client_update(
        client_id=7, local_model=model_1, client_tensors=tensors,
        loss_fn=loss_fn, current_lr=0.01, device=device, round_idx=1,
    )
    assert 7 in scaffold.c_i
    c_i_after_round1 = {k: v.clone() for k, v in scaffold.c_i[7].items()}
    # Second round (same client)
    _ = scaffold.client_update(
        client_id=7, local_model=model_2, client_tensors=tensors,
        loss_fn=loss_fn, current_lr=0.01, device=device, round_idx=2,
    )
    # c_i for client 7 should evolve between rounds
    changed = any(
        not torch.allclose(scaffold.c_i[7][k], c_i_after_round1[k])
        for k in c_i_after_round1
    )
    assert changed, "c_i should evolve across rounds"
