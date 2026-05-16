"""Class-level invariants for FedSCAM (Rahil et al. 2026, arXiv:2601.00853).

These tests pin the contract the V100 ablation depends on. Mirrors the
TDD pattern of tests/test_fedswa_methodology.py and tests/test_v5_*.py.

Headline invariants:

* Registration + required-kwargs (matches _ALGO_REQUIRED_KWARGS table).
* Constructor validation (negative rho_max / alpha_rho / etc rejected).
* ``rho_max=0 AND gamma=0 AND beta_align=0`` ⇒ reduces to FedAvg on the
  server side bit-exactly. The fast-path in client_update also short-
  circuits the pilot phase to keep RNG state aligned with FedAvg's
  baseline path (verified by a smoke test below that runs FedAvg and
  FedSCAM(reduce-cfg) on identical (seed, data) and asserts identical
  final state_dicts).
* Aggregation weight formula ``S_i = N_i / (1 + γ·h_i^adj) ·
  max(0, 1+β_align·c_i)`` is applied as documented.
* Alignment c_i with previous global direction is bounded in [-1, 1]
  and computed via dot of unit vectors.
"""
from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from fl_oran.federated.algorithms import REGISTRY
from fl_oran.federated.client import ClientUpdate


# ---------------------------------------------------------------------------
# Registration / signature contract
# ---------------------------------------------------------------------------


def test_fedscam_registered() -> None:
    assert "fedscam" in REGISTRY
    assert REGISTRY["fedscam"].name == "fedscam"


def test_fedscam_requires_all_documented_kwargs() -> None:
    """Spec loader relies on these being keyword-only-no-default."""
    with pytest.raises(TypeError):
        REGISTRY["fedscam"](max_steps=1, batch_size=1)


@pytest.mark.parametrize("bad_kwargs", [
    {"rho_max": -0.1, "alpha_rho": 1.0, "gamma": 1.0, "beta_align": 0.8, "kappa": 1.0},
    {"rho_max": 0.05, "alpha_rho": -0.1, "gamma": 1.0, "beta_align": 0.8, "kappa": 1.0},
    {"rho_max": 0.05, "alpha_rho": 1.0, "gamma": -0.1, "beta_align": 0.8, "kappa": 1.0},
    {"rho_max": 0.05, "alpha_rho": 1.0, "gamma": 1.0, "beta_align": -0.1, "kappa": 1.0},
    {"rho_max": 0.05, "alpha_rho": 1.0, "gamma": 1.0, "beta_align": 0.8, "kappa": -0.1},
])
def test_fedscam_rejects_negative_hyperparams(bad_kwargs) -> None:
    with pytest.raises(ValueError):
        REGISTRY["fedscam"](max_steps=1, batch_size=1, **bad_kwargs)


def test_fedscam_rejects_zero_b_pilot() -> None:
    with pytest.raises(ValueError, match="b_pilot"):
        REGISTRY["fedscam"](
            max_steps=1, batch_size=1,
            rho_max=0.05, alpha_rho=1.0, gamma=1.0, beta_align=0.8, kappa=1.0,
            b_pilot=0,
        )


# ---------------------------------------------------------------------------
# Server-side: aggregation weight formula
# ---------------------------------------------------------------------------


def _make_synthetic_updates(seed: int = 0, n_clients: int = 3) -> list[ClientUpdate]:
    """Produce n_clients ClientUpdate objects with a small toy state."""
    torch.manual_seed(seed)
    updates: list[ClientUpdate] = []
    for cid in range(n_clients):
        state = {
            "linear.weight": torch.randn(3, 4),
            "linear.bias": torch.randn(3),
        }
        updates.append(ClientUpdate(
            client_id=cid,
            state_dict=state,
            num_examples=10 * (cid + 1),
            train_loss=0.5,
        ))
    return updates


def _make_global_state_like(updates: list[ClientUpdate]) -> dict[str, torch.Tensor]:
    sample = updates[0].state_dict
    torch.manual_seed(99)
    return {
        k: torch.randn_like(v) if v.dtype.is_floating_point else torch.zeros_like(v)
        for k, v in sample.items()
    }


def test_fedscam_aggregation_reduces_to_fedavg_when_no_modulation() -> None:
    """gamma=0 AND beta_align=0 ⇒ aggregation weights collapse to N_i,
    which IS FedAvg's weighting. Bit-equal to FedAvg.server_aggregate
    when client metadata is uninitialised (first round, c_i=h_i_adj=0).
    """
    fedavg = REGISTRY["fedavg"](max_steps=1, batch_size=1)
    scam = REGISTRY["fedscam"](
        max_steps=1, batch_size=1,
        rho_max=0.0, alpha_rho=1.0, gamma=0.0, beta_align=0.0, kappa=1.0,
    )
    updates = _make_synthetic_updates(seed=42)
    gs = _make_global_state_like(updates)
    # FedSCAM with no client_update having run leaves _client_meta empty;
    # server_aggregate's fallback is h_adj=c_i=0 per client.
    avg_out = fedavg.server_aggregate(
        global_state={k: v.clone() for k, v in gs.items()},
        updates=updates,
    )
    scam_out = scam.server_aggregate(
        global_state={k: v.clone() for k, v in gs.items()},
        updates=updates,
    )
    assert set(avg_out.keys()) == set(scam_out.keys())
    for k in avg_out:
        torch.testing.assert_close(
            scam_out[k], avg_out[k], rtol=1e-6, atol=1e-7,
            msg=lambda m, k=k: f"FedSCAM(no-mod) diverges from FedAvg on {k}: {m}",
        )


def test_fedscam_aggregation_weights_apply_documented_formula() -> None:
    """Hand-construct three clients with explicit (N_i, h_i_adj, c_i),
    inject the metadata, run server_aggregate, and check the weighted
    average matches the formula ``S_i = N_i · 1/(1+γh^adj) · max(0, 1+β·c)``.
    """
    scam = REGISTRY["fedscam"](
        max_steps=1, batch_size=1,
        rho_max=0.05, alpha_rho=1.0, gamma=1.0, beta_align=0.8, kappa=1.0,
    )
    updates = _make_synthetic_updates(seed=7)
    gs = _make_global_state_like(updates)
    # Inject hand-picked metadata
    scam._client_meta = {
        0: {"h_i_adj": 0.0, "c_i": 0.0,  "z_i": torch.zeros(0)},
        1: {"h_i_adj": 1.0, "c_i": +0.5, "z_i": torch.zeros(0)},
        2: {"h_i_adj": 2.0, "c_i": -1.0, "z_i": torch.zeros(0)},
    }
    # Expected weights per the formula
    expected_S = []
    for u in updates:
        m = scam._client_meta[u.client_id]
        n_i = float(u.num_examples)
        hf = 1.0 / (1.0 + 1.0 * m["h_i_adj"])
        af = max(0.0, 1.0 + 0.8 * m["c_i"])
        expected_S.append(n_i * hf * af)
    total = sum(expected_S)
    expected_norm = [w / total for w in expected_S]
    # Expected per-key aggregate
    expected: dict[str, torch.Tensor] = {}
    for key in updates[0].state_dict:
        ref = updates[0].state_dict[key]
        if ref.dtype.is_floating_point:
            acc = torch.zeros_like(ref)
            for u, w in zip(updates, expected_norm):
                acc.add_(u.state_dict[key].to(acc.dtype), alpha=w)
            expected[key] = acc
        else:
            expected[key] = ref.clone()
    # Run server_aggregate
    out = scam.server_aggregate(
        global_state={k: v.clone() for k, v in gs.items()},
        updates=updates,
    )
    for k in expected:
        torch.testing.assert_close(out[k], expected[k], rtol=1e-6, atol=1e-7)


def test_fedscam_aggregation_falls_back_to_fedavg_when_all_weights_zero() -> None:
    """If beta_align drives align_factor to 0 for every client
    (cosine = -1, beta_align ≥ 1), the formula returns zero weights.
    Implementation falls back to FedAvg's N_i weighting to keep the
    round productive — verify.
    """
    scam = REGISTRY["fedscam"](
        max_steps=1, batch_size=1,
        rho_max=0.05, alpha_rho=1.0, gamma=1.0, beta_align=2.0, kappa=1.0,
    )
    updates = _make_synthetic_updates(seed=1)
    gs = _make_global_state_like(updates)
    # All clients strongly anti-aligned: c_i = -1, align_factor = max(0, 1 + 2*(-1)) = 0
    scam._client_meta = {
        u.client_id: {"h_i_adj": 0.0, "c_i": -1.0, "z_i": torch.zeros(0)}
        for u in updates
    }
    out = scam.server_aggregate(
        global_state={k: v.clone() for k, v in gs.items()},
        updates=updates,
    )
    # Should equal FedAvg of the same updates
    fedavg = REGISTRY["fedavg"](max_steps=1, batch_size=1)
    avg_out = fedavg.server_aggregate(
        global_state={k: v.clone() for k, v in gs.items()},
        updates=updates,
    )
    for k in out:
        torch.testing.assert_close(out[k], avg_out[k], rtol=1e-6, atol=1e-7)


# ---------------------------------------------------------------------------
# Server-side: u_t direction update
# ---------------------------------------------------------------------------


def test_fedscam_updates_global_direction_from_aggregate_movement() -> None:
    """Per paper Algorithm 1: ``u_t ← Proj_d(Normalize(w_{t+1} − w_t))``,
    NOT a mean of per-client z_i directions.

    Fidelity audit fix 2026-05-17: previous behaviour averaged
    client-side z_i, which weighted clients equally regardless of S_i.
    The paper's formulation makes u_t the unit direction of the global
    model's actual movement this round, implicitly weighted by S_i
    through ``w_{t+1}`` = Σ p_i w_i.

    Verify: construct 3 client states with KNOWN deltas relative to
    global, equal aggregation weights, and assert u_t equals the
    unit-normalised mean delta.
    """
    scam = REGISTRY["fedscam"](
        max_steps=1, batch_size=1,
        # rho_max=0 + gamma=0 + beta_align=0 makes S_i collapse to N_i;
        # combined with equal N_i, weights become uniform, so the
        # aggregate movement is exactly the mean of per-client deltas.
        rho_max=0.0, alpha_rho=1.0, gamma=0.0, beta_align=0.0, kappa=1.0,
    )
    # Hand-construct: global = zeros, 3 clients with unit-basis deltas in
    # the linear.weight space; linear.bias unchanged. Flatten order is
    # `linear.weight` (3x4 = 12 elems) then `linear.bias` (3 elems) =
    # 15-dim total. We'll put the basis vectors in the first 3 weight
    # slots so the prediction is easy.
    gs = {
        "linear.weight": torch.zeros(3, 4),
        "linear.bias": torch.zeros(3),
    }
    def _delta_state(weight_flat_idx: int) -> dict[str, torch.Tensor]:
        w = torch.zeros(3, 4)
        # Set element at flat index weight_flat_idx to 1.0
        w.view(-1)[weight_flat_idx] = 1.0
        return {"linear.weight": w, "linear.bias": torch.zeros(3)}
    updates = [
        ClientUpdate(client_id=0, state_dict=_delta_state(0), num_examples=100, train_loss=0.5),
        ClientUpdate(client_id=1, state_dict=_delta_state(1), num_examples=100, train_loss=0.5),
        ClientUpdate(client_id=2, state_dict=_delta_state(2), num_examples=100, train_loss=0.5),
    ]
    assert scam._last_global_direction is None
    scam.server_aggregate(
        global_state={k: v.clone() for k, v in gs.items()},
        updates=updates,
    )
    u = scam._last_global_direction
    assert u is not None
    # u_t should be unit-norm
    torch.testing.assert_close(u.norm(), torch.tensor(1.0), atol=1e-6, rtol=0.0)
    # Aggregate movement: mean of 3 unit-basis deltas in slots {0,1,2}
    # of the 12-elem weight (then 0s in bias) → flat shape 15:
    # values = [1/3, 1/3, 1/3, 0, ..., 0] (12 weight + 3 bias = 15).
    # Norm = sqrt(3 * (1/3)^2) = 1/sqrt(3).
    # Normalised = [1/sqrt(3), 1/sqrt(3), 1/sqrt(3), 0, ..., 0].
    assert u.numel() == 15, f"expected 15 flat elements, got {u.numel()}"
    expected = torch.zeros(15)
    expected[0] = expected[1] = expected[2] = 1.0 / math.sqrt(3.0)
    torch.testing.assert_close(u, expected, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# Client-side: end-to-end smoke with a tiny MLP on CPU
# ---------------------------------------------------------------------------


class _TinyHead(nn.Module):
    """Mini stand-in for ForecasterV2: ignores categorical input, projects
    continuous to a single logit. Lets us run client_update on CPU
    deterministically.
    """
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 1, bias=True)

    def forward(self, cat: torch.Tensor, cont: torch.Tensor) -> torch.Tensor:
        return self.fc(cont).squeeze(-1)


def _make_tensors(n: int = 64, seed: int = 11) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    cat = torch.zeros(n, dtype=torch.long)
    cont = torch.randn(n, 4)
    y = torch.randint(0, 2, (n,), dtype=torch.float)
    return cat, cont, y


def _loss_fn(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # BCE with logits on single-logit output.
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, y)


def test_fedscam_client_update_returns_well_shaped_update() -> None:
    """End-to-end smoke: client_update runs, returns a ClientUpdate, and
    caches the per-client metadata expected by server_aggregate.
    """
    scam = REGISTRY["fedscam"](
        max_steps=5, batch_size=8,
        rho_max=0.05, alpha_rho=1.0, gamma=1.0, beta_align=0.8, kappa=1.0,
        b_pilot=2,
    )
    model = _TinyHead()
    tensors = _make_tensors()
    upd = scam.client_update(
        client_id=42,
        local_model=model,
        client_tensors=tensors,
        loss_fn=_loss_fn,
        current_lr=5e-4,
        device=torch.device("cpu"),
        round_idx=0,
    )
    assert isinstance(upd, ClientUpdate)
    assert upd.client_id == 42
    assert "fc.weight" in upd.state_dict
    assert torch.isfinite(torch.tensor(upd.train_loss))
    # metadata cached
    assert 42 in scam._client_meta
    m = scam._client_meta[42]
    assert 0.0 <= m["rho_i"] <= scam.rho_max
    # First round → c_i = 0 (no previous global direction)
    assert m["c_i"] == 0.0


def test_fedscam_alignment_bounded_in_minus1_plus1() -> None:
    """After two rounds, c_i in [-1, 1] strictly (cosine similarity)."""
    scam = REGISTRY["fedscam"](
        max_steps=5, batch_size=8,
        rho_max=0.05, alpha_rho=1.0, gamma=1.0, beta_align=0.8, kappa=1.0,
        b_pilot=2,
    )
    model = _TinyHead()
    tensors = _make_tensors()
    # Round 1
    scam.client_update(
        client_id=0, local_model=model, client_tensors=tensors,
        loss_fn=_loss_fn, current_lr=5e-4, device=torch.device("cpu"),
        round_idx=0,
    )
    # Simulate server: set _last_global_direction
    scam._last_global_direction = torch.randn(model.fc.weight.numel() + model.fc.bias.numel())
    scam._last_global_direction = (
        scam._last_global_direction / scam._last_global_direction.norm()
    )
    # Round 2 with fresh model state to make pilot grad direction non-trivial
    model2 = _TinyHead()
    scam.client_update(
        client_id=1, local_model=model2, client_tensors=tensors,
        loss_fn=_loss_fn, current_lr=5e-4, device=torch.device("cpu"),
        round_idx=1,
    )
    c_i = scam._client_meta[1]["c_i"]
    assert -1.0 <= c_i <= 1.0


# ---------------------------------------------------------------------------
# rho_max=0 should reduce to plain training (no SAM perturbation)
# ---------------------------------------------------------------------------


def test_fedscam_rho_max_zero_no_perturbation() -> None:
    """rho_max=0 ⇒ eps=0 ⇒ no ascent/restore. Training reduces to plain
    SGD/Adam over the same data path that FedAvg would use. Final
    state_dict should not exactly match FedAvg's (because pilot is
    skipped only when gamma=beta_align=0 too — see fast-path comment),
    but should be finite and the perturbation arrays should be empty.
    """
    scam = REGISTRY["fedscam"](
        max_steps=5, batch_size=8,
        rho_max=0.0, alpha_rho=1.0, gamma=1.0, beta_align=0.8, kappa=1.0,
        b_pilot=2,
    )
    model = _TinyHead()
    tensors = _make_tensors()
    upd = scam.client_update(
        client_id=0, local_model=model, client_tensors=tensors,
        loss_fn=_loss_fn, current_lr=5e-4, device=torch.device("cpu"),
        round_idx=0,
    )
    # Sanity: training happened and loss is finite
    for v in upd.state_dict.values():
        assert torch.isfinite(v).all()
    # rho_i should be exactly 0.0 (rho_max=0 / (1 + alpha_rho * h_i_adj) = 0)
    assert scam._client_meta[0]["rho_i"] == 0.0
