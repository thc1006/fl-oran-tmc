"""TDD tests for MOON (Li et al. 2021, CVPR) — ADR-001 §3.7.

MOON augments the local objective with a model-contrastive term:

    z       = encode(w_local, x)
    z_g     = encode(w_global, x)      # positive, frozen
    z_prev  = encode(w_prev_round, x)  # negative, frozen
    L_total = L_CE + mu * L_contrastive(z, z_g, z_prev; tau)

Representation extraction is abstracted via a caller-provided
``encode_fn(model, cat, cont) -> Tensor`` — MOON is model-agnostic.

Tests:
1. Registered.
2. Required kwargs (``mu``, ``tau``, ``encode_fn``).
3. ``mu=0`` reduces to FedAvg bit-wise (guard short-circuits contrastive).
4. ``mu>0`` produces a trajectory that is NOT identical to FedAvg (the
   contrastive gradient actually contributes).
5. ``self.prev_models[client_id]`` is populated after a client_update and
   evolves across rounds.
"""
from __future__ import annotations

import pytest
import torch

from conftest import build_trio as _build_trio


def _tiny_encode(model, cat: torch.Tensor, cont: torch.Tensor) -> torch.Tensor:
    """Synthesise a multi-dim representation that reuses the model's weight.

    For TinyDualInput (Linear(4, 1)) we cannot use the final 1-dim logit as
    representation because cosine similarity on 1-dim vectors collapses to
    sign(product), whose gradient is zero almost everywhere. Instead we
    element-wise gate the 4-dim input by the model's weight vector — a
    (B, 4) output with gradient flowing back through ``linear.weight``.
    Mimics MOON's production setup where the contrastive head reuses the
    trunk parameters but outputs a multi-dim embedding.
    """
    cat_f = cat.float().mean(dim=1)
    cont_f = cont.mean(dim=1)
    x = torch.cat([cat_f, cont_f], dim=-1)     # (B, 4)
    return x * model.linear.weight.view(-1)    # (B, 4)


def test_moon_registered():
    from fl_oran.federated.algorithms import REGISTRY
    assert "moon" in REGISTRY
    assert REGISTRY["moon"].name == "moon"


def test_moon_requires_mu_and_tau_and_encode_fn():
    from fl_oran.federated.algorithms import REGISTRY
    cls = REGISTRY["moon"]
    # Missing encode_fn
    with pytest.raises(TypeError, match="encode_fn"):
        cls(max_steps=1, batch_size=1, mu=0.1, tau=0.5)
    # Missing mu
    with pytest.raises(TypeError, match="mu"):
        cls(max_steps=1, batch_size=1, tau=0.5, encode_fn=_tiny_encode)
    # Missing tau
    with pytest.raises(TypeError, match="tau"):
        cls(max_steps=1, batch_size=1, mu=0.1, encode_fn=_tiny_encode)


def test_moon_mu_zero_matches_fedavg_trajectory():
    """mu=0 ==> contrastive term skipped entirely ==> bit-identical to FedAvg."""
    from fl_oran.federated.algorithms import REGISTRY
    model_a, tensors, loss_fn = _build_trio(seed=42)
    model_b, _, _ = _build_trio(seed=42)
    fedavg = REGISTRY["fedavg"](max_steps=5, batch_size=4, grad_clip=1.0)
    moon = REGISTRY["moon"](max_steps=5, batch_size=4, grad_clip=1.0,
                             mu=0.0, tau=0.5, encode_fn=_tiny_encode)
    device = torch.device("cpu")
    torch.manual_seed(13)
    u_a = fedavg.client_update(
        client_id=1, local_model=model_a, client_tensors=tensors,
        loss_fn=loss_fn, current_lr=0.01, device=device, round_idx=1,
    )
    torch.manual_seed(13)
    u_b = moon.client_update(
        client_id=1, local_model=model_b, client_tensors=tensors,
        loss_fn=loss_fn, current_lr=0.01, device=device, round_idx=1,
    )
    for k in u_a.state_dict:
        torch.testing.assert_close(u_a.state_dict[k], u_b.state_dict[k])


def test_moon_mu_positive_diverges_from_fedavg():
    """mu>0 with a populated prev_model produces a different trajectory.

    MOON's first round is trivially FedAvg-equivalent (the paper notes that
    when there is no prior local model, the contrastive numerator and
    denominator collapse). The contrastive term only has a nonzero gradient
    once ``prev_model`` differs from ``global_model``. We prime MOON's
    ``prev_models[client_id]`` with a perturbed snapshot to simulate a
    round-2+ setting, then compare one client_update to FedAvg.
    """
    from fl_oran.federated.algorithms import REGISTRY
    model_a, tensors, loss_fn = _build_trio(seed=42)
    model_b, _, _ = _build_trio(seed=42)
    fedavg = REGISTRY["fedavg"](max_steps=10, batch_size=4, grad_clip=1.0)
    moon = REGISTRY["moon"](max_steps=10, batch_size=4, grad_clip=1.0,
                             mu=1.0, tau=0.5, encode_fn=_tiny_encode)
    device = torch.device("cpu")

    # Prime MOON's negative (prev_model) with a perturbed snapshot so the
    # contrastive gradient is nontrivial.
    init_state = {k: v.clone() for k, v in model_b.state_dict().items()}
    moon.prev_models[1] = {
        k: (v + 0.5) if v.dtype.is_floating_point else v.clone()
        for k, v in init_state.items()
    }

    torch.manual_seed(99)
    u_a = fedavg.client_update(
        client_id=1, local_model=model_a, client_tensors=tensors,
        loss_fn=loss_fn, current_lr=0.05, device=device, round_idx=2,
    )
    torch.manual_seed(99)
    u_b = moon.client_update(
        client_id=1, local_model=model_b, client_tensors=tensors,
        loss_fn=loss_fn, current_lr=0.05, device=device, round_idx=2,
    )
    diverged = any(
        not torch.allclose(u_a.state_dict[k], u_b.state_dict[k], atol=1e-6)
        for k in u_a.state_dict
        if u_a.state_dict[k].dtype.is_floating_point
    )
    assert diverged, "mu=1.0 MOON with a populated prev_model should diverge from FedAvg"


def test_moon_prev_models_persists():
    from fl_oran.federated.algorithms import REGISTRY
    model_1, tensors, loss_fn = _build_trio(seed=42)
    model_2, _, _ = _build_trio(seed=42)
    moon = REGISTRY["moon"](max_steps=3, batch_size=4,
                             mu=0.1, tau=0.5, encode_fn=_tiny_encode)
    device = torch.device("cpu")
    _ = moon.client_update(
        client_id=3, local_model=model_1, client_tensors=tensors,
        loss_fn=loss_fn, current_lr=0.01, device=device, round_idx=1,
    )
    assert 3 in moon.prev_models, "prev_models should register client 3 after round 1"
    snap = {k: v.clone() for k, v in moon.prev_models[3].items()}
    _ = moon.client_update(
        client_id=3, local_model=model_2, client_tensors=tensors,
        loss_fn=loss_fn, current_lr=0.01, device=device, round_idx=2,
    )
    changed = any(
        not torch.allclose(moon.prev_models[3][k], snap[k])
        for k in snap
        if snap[k].dtype.is_floating_point
    )
    assert changed, "prev_models[3] should evolve between rounds"
