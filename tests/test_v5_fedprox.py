"""TDD red-phase tests for FedProx (Li et al. 2020) — ADR-001 §3.3.

FedProx adds a proximal term (μ/2)·‖w - w_global‖² to the client's local
objective. The gradient of the prox term is μ·(w - w_global), which we inject
into p.grad directly (cheaper than adding to the loss; standard impl choice).

Tests:
1. ``fedprox`` is registered.
2. Missing ``mu`` raises TypeError (keyword-only, no default).
3. ``mu=0`` is a bit-identical trajectory to FedAvg — pure identity check
   (grad injection of 0 ≡ no injection).
4. ``mu>0`` keeps local weights closer to the global snapshot than ``mu=0``
   for the same init / data / batch sequence.
"""
from __future__ import annotations

import copy

import pytest
import torch
from torch import nn


# --------------------------------------------------------------------------
# Helpers local to this test module.
# --------------------------------------------------------------------------


def _build_trio(seed: int = 42, n: int = 64, seq_len: int = 3,
                n_cat: int = 2, n_cont: int = 2):
    """Build (model, client_tensors, loss_fn) deterministically from ``seed``.

    The model mimics ForecasterV2's dual (cat, cont) input signature so the
    same ``client_update(local_model=..., client_tensors=(cat, cont, y))``
    interface drives it. Tiny so tests run in <1 s.
    """
    torch.manual_seed(seed)

    class TinyDualInput(nn.Module):
        def __init__(self, n_in: int) -> None:
            super().__init__()
            self.linear = nn.Linear(n_in, 1)

        def forward(self, cat: torch.Tensor, cont: torch.Tensor) -> torch.Tensor:
            cat_f = cat.float().mean(dim=1)   # (B, n_cat)
            cont_f = cont.mean(dim=1)          # (B, n_cont)
            x = torch.cat([cat_f, cont_f], dim=-1)  # (B, n_cat + n_cont)
            return self.linear(x)              # (B, 1)

    model = TinyDualInput(n_cat + n_cont)
    cat = torch.randint(0, 5, (n, seq_len, n_cat), dtype=torch.long)
    cont = torch.randn(n, seq_len, n_cont, dtype=torch.float32)
    y = torch.randint(0, 2, (n, 1)).float()
    loss_fn = nn.BCEWithLogitsLoss()
    return model, (cat, cont, y), loss_fn


def _l2_to(sd_a: dict, sd_b: dict) -> float:
    """Sum of squared float-param differences — a scalar drift proxy."""
    total = 0.0
    for k, v in sd_a.items():
        if v.dtype.is_floating_point:
            total += float(((v - sd_b[k]) ** 2).sum())
    return total


# --------------------------------------------------------------------------
# Tests (red before src/fl_oran/federated/algorithms/fedprox.py exists).
# --------------------------------------------------------------------------


def test_fedprox_registered():
    from fl_oran.federated.algorithms import REGISTRY
    assert "fedprox" in REGISTRY, (
        f"REGISTRY missing 'fedprox'; has {sorted(REGISTRY)}"
    )
    assert REGISTRY["fedprox"].name == "fedprox"


def test_fedprox_requires_mu():
    from fl_oran.federated.algorithms import REGISTRY
    cls = REGISTRY["fedprox"]
    with pytest.raises(TypeError, match="mu"):
        cls(max_steps=1, batch_size=1)


def test_fedprox_mu_zero_matches_fedavg_trajectory():
    """mu=0 ⇒ prox gradient is zero ⇒ identical trajectory to FedAvg.

    Both runs share: init weights, data, batch-sampling RNG, optimizer (fresh
    Adam so m/v start zero). With identical grad sequence, Adam is
    deterministic, so state_dicts must agree bit-wise.
    """
    from fl_oran.federated.algorithms import REGISTRY

    model_a, tensors, loss_fn = _build_trio(seed=42)
    model_b, _, _ = _build_trio(seed=42)  # identical init + data

    fedavg = REGISTRY["fedavg"](max_steps=8, batch_size=4, grad_clip=1.0)
    fedprox_zero = REGISTRY["fedprox"](max_steps=8, batch_size=4, grad_clip=1.0, mu=0.0)

    device = torch.device("cpu")

    torch.manual_seed(123)
    u_avg = fedavg.client_update(
        client_id=1, local_model=model_a, client_tensors=tensors,
        loss_fn=loss_fn, current_lr=0.01, device=device, round_idx=1,
    )
    torch.manual_seed(123)
    u_prox = fedprox_zero.client_update(
        client_id=1, local_model=model_b, client_tensors=tensors,
        loss_fn=loss_fn, current_lr=0.01, device=device, round_idx=1,
    )

    assert set(u_avg.state_dict.keys()) == set(u_prox.state_dict.keys())
    for k in u_avg.state_dict:
        torch.testing.assert_close(
            u_avg.state_dict[k], u_prox.state_dict[k],
            msg=lambda m, k=k: f"{m}: mu=0 diverged from FedAvg at key {k!r}",
        )


def test_fedprox_mu_positive_keeps_weights_closer_to_global():
    """Sanity: μ>0 pulls local weights toward the global snapshot.

    With identical init/data/batch-RNG, running FedProx at μ=1.0 should leave
    the local weights closer (in L2) to the pre-training snapshot than μ=0.0.
    """
    from fl_oran.federated.algorithms import REGISTRY

    model_zero, tensors, loss_fn = _build_trio(seed=42)
    model_pos, _, _ = _build_trio(seed=42)
    # Snapshot *before* training (this is what both runs will be pulled toward
    # — identical for both since _build_trio(seed=42) is deterministic).
    global_state = copy.deepcopy(model_zero.state_dict())

    fedprox_zero = REGISTRY["fedprox"](max_steps=30, batch_size=8, mu=0.0)
    fedprox_pos = REGISTRY["fedprox"](max_steps=30, batch_size=8, mu=1.0)

    device = torch.device("cpu")

    torch.manual_seed(777)
    u_zero = fedprox_zero.client_update(
        client_id=1, local_model=model_zero, client_tensors=tensors,
        loss_fn=loss_fn, current_lr=0.05, device=device, round_idx=1,
    )
    torch.manual_seed(777)
    u_pos = fedprox_pos.client_update(
        client_id=1, local_model=model_pos, client_tensors=tensors,
        loss_fn=loss_fn, current_lr=0.05, device=device, round_idx=1,
    )

    d_zero = _l2_to(u_zero.state_dict, global_state)
    d_pos = _l2_to(u_pos.state_dict, global_state)
    assert d_pos < d_zero, (
        f"expected μ=1.0 drift {d_pos:.4g} < μ=0.0 drift {d_zero:.4g} "
        "(prox term should pull weights back toward global)"
    )
