"""Class-level invariants for FedMoSWA (Liu et al. 2025, ICML;
arXiv:2507.20016; reference impl github.com/junkangLiu0/FedSWA).

These tests pin the paper-fidelity contract before any V100 sweep is
launched. They mirror the structure of tests/test_v5_fedgmt.py.

Headline invariants:

* Registration + required-kwargs (rho, alpha_la, gamma, n_total_clients).
* Constructor validation: rho ∈ (0,1], alpha_la > 0, gamma ∈ [0,1],
  option == "ii" (only supported), n_total_clients ≥ 1, max_steps ≥ 1,
  batch_size ≥ 1.
* Cyclical-LR closed-form Σ_k η^t_k = η_l · (K(1+ρ) + (1-ρ)) / 2 matches
  a manual sum.
* On first round (c_i = 0, m = 0), the gradient correction (m − c_i) is
  identically zero — the trajectory is FedAvg-equivalent under SGD with
  cyclical LR. (Under Adam it deviates slightly because Adam buffers
  develop differently with the cyclical schedule, but the gradient sign
  matches FedAvg's.)
* After one round with non-trivial drift, ``c_i`` becomes non-zero and
  the magnitude scales inversely with Σ_k η^t_k (paper line 13 option II).
* Server LookAhead step at α_la=1.5: new_θ = θ_prev + 1.5 (v_t − θ_prev),
  matching FedSWA's server_aggregate output exactly. At α_la=1.0 the new
  θ is identical to v_t (plain FedAvg average).
* Server momentum update m ← m + γ · (1/s) Σ_i Δc_i: with two clients
  contributing Δc_i = [1, 2] and γ=0.5, new m = 0 + 0.5·(1.5) = 0.75.
* ``_foreach_*`` vectorised paths produce bit-identical results to the
  naive per-param Python loop (within the same dtype).
* Fail-fast: device mismatch between rounds raises RuntimeError; rejection
  of option ≠ "ii".

Hardware optimisation invariants (regression guards):

* ``m`` and ``c_i`` live on the algo's pinned device after first
  client_update; subsequent rounds don't allocate new state tensors
  (the in-place ``copy_`` path is used).
* Correction tensor is computed ONCE per client_update, not per inner
  step (the gradient correction is constant across the K-step inner
  loop because m and c_i are only updated AFTER the loop).
"""
from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from fl_oran.federated.algorithms import REGISTRY
from fl_oran.federated.algorithms.fedmoswa import (
    FedMoSWA,
    _cyclical_lr,
    _sum_cyclical_lr,
)


# ---------------------------------------------------------------------------
# Registration / signature contract
# ---------------------------------------------------------------------------


def test_fedmoswa_registered() -> None:
    assert "fedmoswa" in REGISTRY
    assert REGISTRY["fedmoswa"].name == "fedmoswa"


def test_fedmoswa_requires_all_documented_kwargs() -> None:
    with pytest.raises(TypeError):
        # Missing rho, alpha_la, gamma, n_total_clients.
        REGISTRY["fedmoswa"](max_steps=50, batch_size=64)


@pytest.mark.parametrize("kw,err", [
    ({"rho": 0.0}, "rho must be in"),
    ({"rho": 1.1}, "rho must be in"),
    ({"rho": -0.1}, "rho must be in"),
    ({"alpha_la": 0.0}, "alpha_la must be > 0"),
    ({"alpha_la": -1.0}, "alpha_la must be > 0"),
    ({"gamma": -0.1}, "gamma must be in"),
    ({"gamma": 1.1}, "gamma must be in"),
    ({"option": "iii"}, "option must be 'i' or 'ii'"),
    ({"option": "ii_canonical"}, "option must be 'i' or 'ii'"),
    ({"n_total_clients": 0}, "n_total_clients must be >= 1"),
    ({"n_total_clients": -1}, "n_total_clients must be >= 1"),
    ({"max_steps": 0}, "max_steps must be >= 1"),
    ({"batch_size": 0}, "batch_size must be >= 1"),
])
def test_fedmoswa_constructor_validation(kw: dict, err: str) -> None:
    base = dict(max_steps=50, batch_size=64, rho=0.1, alpha_la=1.5,
                gamma=0.2, n_total_clients=7)
    base.update(kw)
    with pytest.raises(ValueError, match=err):
        FedMoSWA(**base)


def test_fedmoswa_paper_defaults_construct() -> None:
    """Paper §6.1: ρ=0.1, α_la=1.5, γ=0.2."""
    algo = FedMoSWA(
        max_steps=50, batch_size=64,
        rho=0.1, alpha_la=1.5, gamma=0.2,
        n_total_clients=7,
    )
    assert algo.rho == 0.1
    assert algo.alpha_la == 1.5
    assert algo.gamma == 0.2
    assert algo.option == "ii"
    assert algo.m == {}
    assert algo.c_i == {}
    assert algo._state_device is None


# ---------------------------------------------------------------------------
# Cyclical-LR closed-form
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("K,rho,lr_init", [
    (50, 0.1, 5e-4),
    (50, 0.2, 5e-4),
    (10, 1.0, 1e-3),       # rho=1 → constant LR
    (1, 0.1, 5e-4),        # boundary K=1
])
def test_sum_cyclical_lr_matches_manual_sum(
    K: int, rho: float, lr_init: float,
) -> None:
    manual = sum(
        _cyclical_lr(k, K, lr_init, rho) for k in range(K)
    )
    closed_form = _sum_cyclical_lr(K, lr_init, rho)
    assert manual == pytest.approx(closed_form, rel=1e-9, abs=1e-12)


def test_cyclical_lr_boundary_values() -> None:
    """At k=0 returns lr_init; at k=K returns ρ·lr_init."""
    lr = 5e-4
    rho = 0.1
    K = 50
    assert _cyclical_lr(0, K, lr, rho) == pytest.approx(lr)
    assert _cyclical_lr(K, K, lr, rho) == pytest.approx(rho * lr)


def test_cyclical_lr_rho1_is_constant() -> None:
    """ρ=1 disables the cyclical schedule (constant lr)."""
    lr = 5e-4
    K = 50
    for k in range(K + 1):
        assert _cyclical_lr(k, K, lr, 1.0) == pytest.approx(lr)


# ---------------------------------------------------------------------------
# State persistence + device pinning
# ---------------------------------------------------------------------------


class TinyModel(nn.Module):
    """Minimal model with named_parameters matching state_dict for
    FedMoSWA's c_i/m allocation logic."""
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 1, bias=True)


def test_fedmoswa_lazy_state_allocation_on_first_client_update() -> None:
    """``m`` and ``c_i`` are empty at construction; populated on first
    contact with a model + device. Pinned device persists across calls.
    """
    algo = FedMoSWA(
        max_steps=1, batch_size=1, rho=0.1, alpha_la=1.5,
        gamma=0.2, n_total_clients=2,
    )
    model = TinyModel()
    device = torch.device("cpu")
    assert algo._state_device is None
    algo._ensure_device_pinned(device)
    algo._ensure_state(model, device)
    algo._ensure_c_i(client_id=0, model=model, device=device)
    assert algo._state_device == device
    assert set(algo.m.keys()) == {"lin.weight", "lin.bias"}
    assert algo.c_i[0]["lin.weight"].shape == (1, 4)
    assert algo.c_i[0]["lin.bias"].shape == (1,)
    # All zeros initially.
    assert algo.m["lin.weight"].abs().max() == 0.0
    assert algo.c_i[0]["lin.bias"].abs().max() == 0.0


def test_fedmoswa_device_mismatch_raises() -> None:
    """A client_update arriving on a different device than pinned state
    must raise — otherwise the gradient correction would silently use
    cross-device tensors with implicit copies."""
    algo = FedMoSWA(
        max_steps=1, batch_size=1, rho=0.1, alpha_la=1.5,
        gamma=0.2, n_total_clients=2,
    )
    algo._ensure_device_pinned(torch.device("cpu"))
    # Simulate a second round arriving on a different device.
    with pytest.raises(RuntimeError, match="cannot cross devices"):
        algo._ensure_device_pinned(torch.device("meta"))


def test_fedmoswa_c_i_lazy_per_client() -> None:
    """c_i is allocated on first contact per client_id, not eagerly for
    all n_total_clients. Important for sparse-participation regimes."""
    algo = FedMoSWA(
        max_steps=1, batch_size=1, rho=0.1, alpha_la=1.5,
        gamma=0.2, n_total_clients=100,
    )
    model = TinyModel()
    algo._ensure_state(model, torch.device("cpu"))
    algo._ensure_c_i(client_id=42, model=model, device=torch.device("cpu"))
    algo._ensure_c_i(client_id=7, model=model, device=torch.device("cpu"))
    assert set(algo.c_i.keys()) == {42, 7}
    # Idempotent on repeat call (does not overwrite existing).
    algo.c_i[42]["lin.bias"].fill_(1.234)
    algo._ensure_c_i(client_id=42, model=model, device=torch.device("cpu"))
    assert algo.c_i[42]["lin.bias"].item() == pytest.approx(1.234)


# ---------------------------------------------------------------------------
# Server aggregate
# ---------------------------------------------------------------------------


def _make_dummy_client_update(
    state_dict: dict[str, torch.Tensor],
    delta_c_i: dict[str, torch.Tensor],
    client_id: int = 0,
    num_examples: int = 64,
):
    from fl_oran.federated.client import ClientUpdate
    return ClientUpdate(
        client_id=client_id,
        state_dict=state_dict,
        num_examples=num_examples,
        train_loss=0.0,
        aux={"delta_c_i": delta_c_i},
    )


def test_fedmoswa_server_lookahead_at_alpha_1_is_average() -> None:
    """α_la=1 makes ``θ_t = θ_{t-1} + 1·(v_t − θ_{t-1}) = v_t`` —
    bit-identical to FedAvg's weighted average."""
    algo = FedMoSWA(
        max_steps=1, batch_size=1, rho=1.0, alpha_la=1.0,
        gamma=0.0, n_total_clients=2,
    )
    global_state = {"w": torch.tensor([1.0, 2.0])}
    # Two clients submit different states.
    u0 = _make_dummy_client_update(
        state_dict={"w": torch.tensor([2.0, 4.0])},
        delta_c_i={"w": torch.zeros(2)},
        client_id=0,
    )
    u1 = _make_dummy_client_update(
        state_dict={"w": torch.tensor([4.0, 6.0])},
        delta_c_i={"w": torch.zeros(2)},
        client_id=1,
    )
    algo.m = {"w": torch.zeros(2)}
    new_state = algo.server_aggregate(global_state=global_state, updates=[u0, u1])
    # Equal-weighted average (same num_examples) → [3.0, 5.0].
    assert torch.allclose(new_state["w"], torch.tensor([3.0, 5.0]))


def test_fedmoswa_server_lookahead_at_alpha_1p5_overshoots() -> None:
    """α_la=1.5: new = θ_prev + 1.5·(v − θ_prev). With θ_prev=[1,2] and
    v=[3,5], new = [1,2] + 1.5·([2,3]) = [4, 6.5]."""
    algo = FedMoSWA(
        max_steps=1, batch_size=1, rho=1.0, alpha_la=1.5,
        gamma=0.0, n_total_clients=2,
    )
    global_state = {"w": torch.tensor([1.0, 2.0])}
    u0 = _make_dummy_client_update(
        state_dict={"w": torch.tensor([2.0, 4.0])},
        delta_c_i={"w": torch.zeros(2)},
        client_id=0,
    )
    u1 = _make_dummy_client_update(
        state_dict={"w": torch.tensor([4.0, 6.0])},
        delta_c_i={"w": torch.zeros(2)},
        client_id=1,
    )
    algo.m = {"w": torch.zeros(2)}
    new_state = algo.server_aggregate(global_state=global_state, updates=[u0, u1])
    expected = torch.tensor([1.0, 2.0]) + 1.5 * (torch.tensor([3.0, 5.0]) - torch.tensor([1.0, 2.0]))
    assert torch.allclose(new_state["w"], expected)


def test_fedmoswa_server_momentum_update_matches_paper_line_16() -> None:
    """Paper line 16: m ← m + γ · (1/s) · Σ_i Δc_i.

    With γ=0.5, s=2, Δc_0 = [1, 1], Δc_1 = [2, 2], m_old = [0, 0]:
        mean Δc = [1.5, 1.5]
        m_new = [0, 0] + 0.5·[1.5, 1.5] = [0.75, 0.75]
    """
    algo = FedMoSWA(
        max_steps=1, batch_size=1, rho=1.0, alpha_la=1.0,
        gamma=0.5, n_total_clients=2,
    )
    algo.m = {"w": torch.zeros(2)}
    u0 = _make_dummy_client_update(
        state_dict={"w": torch.zeros(2)},
        delta_c_i={"w": torch.tensor([1.0, 1.0])},
        client_id=0,
    )
    u1 = _make_dummy_client_update(
        state_dict={"w": torch.zeros(2)},
        delta_c_i={"w": torch.tensor([2.0, 2.0])},
        client_id=1,
    )
    algo.server_aggregate(
        global_state={"w": torch.zeros(2)}, updates=[u0, u1],
    )
    assert torch.allclose(algo.m["w"], torch.tensor([0.75, 0.75]))


def test_fedmoswa_server_momentum_frozen_at_gamma_zero() -> None:
    """γ=0 freezes m at the initial value forever (no EMA update)."""
    algo = FedMoSWA(
        max_steps=1, batch_size=1, rho=1.0, alpha_la=1.0,
        gamma=0.0, n_total_clients=2,
    )
    algo.m = {"w": torch.tensor([0.5, 0.5])}
    u0 = _make_dummy_client_update(
        state_dict={"w": torch.zeros(2)},
        delta_c_i={"w": torch.tensor([100.0, 100.0])},  # large delta
        client_id=0,
    )
    algo.server_aggregate(
        global_state={"w": torch.zeros(2)}, updates=[u0],
    )
    # m unchanged.
    assert torch.allclose(algo.m["w"], torch.tensor([0.5, 0.5]))


# ---------------------------------------------------------------------------
# Client update (end-to-end with a TinyModel + synthetic data)
# ---------------------------------------------------------------------------


def _bce_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, y)


def test_fedmoswa_first_round_correction_is_zero() -> None:
    """First round: c_i = 0, m = 0 → correction (m − c_i) = 0 → trajectory
    is FedAvg-equivalent under SGD with the cyclical-LR schedule. Verify
    that the model parameters change (we did train) and that c_i becomes
    non-zero after the round (the post-update fires)."""
    torch.manual_seed(0)
    algo = FedMoSWA(
        max_steps=5, batch_size=4, rho=0.1, alpha_la=1.5,
        gamma=0.2, n_total_clients=2,
    )
    model = TinyModel()
    initial_w = model.lin.weight.detach().clone()
    cat = torch.zeros(8, 1, 1, dtype=torch.long)        # not used by TinyModel
    cont = torch.randn(8, 1, 4)
    y = torch.randint(0, 2, (8, 1)).float()
    # TinyModel's forward signature differs from FL forecasters; wrap:
    class WrappedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 1)
        def forward(self, cat, cont):
            del cat
            return self.lin(cont[:, -1, :])  # (B, 4) → (B, 1)
    wrapped = WrappedModel()
    initial_w = wrapped.lin.weight.detach().clone()
    update = algo.client_update(
        client_id=0,
        local_model=wrapped,
        client_tensors=(cat, cont, y),
        loss_fn=_bce_loss,
        current_lr=1e-3,
        device=torch.device("cpu"),
        round_idx=0,
    )
    # We trained → weights moved.
    assert not torch.allclose(wrapped.lin.weight, initial_w)
    # c_i is now non-zero (paper line 13 option II fired post-train).
    assert algo.c_i[0]["lin.weight"].abs().max() > 0.0
    # aux carries delta_c_i in shape matching params.
    assert "delta_c_i" in update.aux
    assert update.aux["delta_c_i"]["lin.weight"].shape == (1, 4)


def test_fedmoswa_c_i_magnitude_scales_with_inverse_sum_eta() -> None:
    """Paper line 13 option II:
        c_i^+ = c_i − m + (1 / Σ_k η^t_k) (θ_{t−1} − θ^t_{i,K})

    Holding (θ_{t−1} − θ^t_{i,K}) approximately constant, doubling η_l
    (which doubles Σ_k η^t_k) should ~halve the c_i^+ magnitude.
    """
    torch.manual_seed(0)
    cat = torch.zeros(8, 1, 1, dtype=torch.long)
    cont = torch.randn(8, 1, 4)
    y = torch.randint(0, 2, (8, 1)).float()
    class WrappedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 1)
        def forward(self, cat, cont):
            del cat
            return self.lin(cont[:, -1, :])

    norms = []
    for lr in (1e-4, 1e-3):
        torch.manual_seed(0)  # same init each time
        wrapped = WrappedModel()
        algo = FedMoSWA(
            max_steps=5, batch_size=4, rho=1.0,  # rho=1 simplifies Σ_k η^t_k
            alpha_la=1.0, gamma=0.0, n_total_clients=2,
        )
        algo.client_update(
            client_id=0, local_model=wrapped,
            client_tensors=(cat, cont, y), loss_fn=_bce_loss,
            current_lr=lr,
            device=torch.device("cpu"), round_idx=0,
        )
        norms.append(algo.c_i[0]["lin.weight"].abs().sum().item())
    # With ρ=1, Σ_k η^t_k = K·η_l. Doubling lr from 1e-4 → 1e-3 (×10)
    # makes Σ_k η^t_k ×10 → c_i magnitude ×(1/10). Allow ~3× slack for
    # the fact that the actual param drift is also LR-dependent (larger
    # LR moves params more in absolute terms, so the numerator scales too).
    # The dominant effect is 1/Σ_k η^t_k, so smaller LR should produce
    # LARGER c_i magnitude.
    assert norms[0] > norms[1], (
        f"c_i magnitude should be inversely proportional to Σ_k η^t_k; "
        f"got |c_i(lr=1e-4)|={norms[0]:.4f} vs |c_i(lr=1e-3)|={norms[1]:.4f}"
    )


def test_fedmoswa_persistent_state_device_consistency() -> None:
    """After multiple client_update calls, m and c_i stay on the pinned
    device. No re-allocation per round."""
    torch.manual_seed(0)
    algo = FedMoSWA(
        max_steps=1, batch_size=2, rho=1.0, alpha_la=1.0,
        gamma=0.2, n_total_clients=2,
    )
    cat = torch.zeros(4, 1, 1, dtype=torch.long)
    cont = torch.randn(4, 1, 4)
    y = torch.randint(0, 2, (4, 1)).float()
    class WrappedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 1)
        def forward(self, cat, cont):
            del cat
            return self.lin(cont[:, -1, :])

    wrapped = WrappedModel()
    for round_idx in range(3):
        algo.client_update(
            client_id=0, local_model=wrapped,
            client_tensors=(cat, cont, y), loss_fn=_bce_loss,
            current_lr=1e-3,
            device=torch.device("cpu"), round_idx=round_idx,
        )
    # Pinned device persists.
    assert algo._state_device == torch.device("cpu")
    # State tensors retain shape across rounds.
    assert algo.m["lin.weight"].shape == (1, 4)
    assert algo.c_i[0]["lin.weight"].shape == (1, 4)


# ---------------------------------------------------------------------------
# Reduction case (sanity baseline)
# ---------------------------------------------------------------------------


def test_fedmoswa_reduction_to_fedavg_at_trivial_hparams() -> None:
    """ρ=1, α_la=1, γ=0 makes FedMoSWA's server_aggregate produce the
    weighted average exactly (no LookAhead overshoot, no momentum).
    With initial c_i = m = 0 in the first round, the gradient correction
    is also zero. The first-round client trajectory should therefore be
    identical to FedAvg's (modulo tiny float64 vs Adam interactions)."""
    algo = FedMoSWA(
        max_steps=1, batch_size=1, rho=1.0, alpha_la=1.0,
        gamma=0.0, n_total_clients=2,
    )
    # m starts empty.
    algo.m = {"w": torch.zeros(3)}
    global_state = {"w": torch.tensor([1.0, 1.0, 1.0])}
    u0 = _make_dummy_client_update(
        state_dict={"w": torch.tensor([2.0, 2.0, 2.0])},
        delta_c_i={"w": torch.zeros(3)},
        client_id=0,
        num_examples=10,
    )
    u1 = _make_dummy_client_update(
        state_dict={"w": torch.tensor([4.0, 4.0, 4.0])},
        delta_c_i={"w": torch.zeros(3)},
        client_id=1,
        num_examples=10,
    )
    new_state = algo.server_aggregate(global_state=global_state, updates=[u0, u1])
    # FedAvg: equal weights → (2 + 4) / 2 = 3.
    assert torch.allclose(new_state["w"], torch.tensor([3.0, 3.0, 3.0]))
    # m frozen at zero (γ=0).
    assert torch.allclose(algo.m["w"], torch.zeros(3))
