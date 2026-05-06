"""TDD methodology tests for FedSWA (Liu et al. 2025, ICML; arXiv:2507.20016).

R34-A (rebuttal Phase 1, MC7 deep-review). These tests pin the FedSWA
class-level invariants that the §7.5 empirical comparison relies on.
The headline empirical result (FedSWA-vs-FedAdam paired Δ = +0.000379,
CI95 [+0.000206, +0.000553]) is only meaningful if the FedSWA
implementation faithfully reduces to FedAvg at α_LA=1.0; otherwise the
"FedSWA over FedAvg" framing is fragile.

Design
------
FedSWA = FedAvg client + LookAhead-EMA server step::

    v_t = weighted_average(client states)             # FedAvg result
    w_t = w_{t-1} + α_LA * (v_t - w_{t-1})            # LookAhead EMA

At α_LA=1.0 the second line collapses to ``w_t = v_t``, recovering
FedAvg bit-exactly. We therefore test the server step directly
(no training loop needed) by constructing synthetic ClientUpdate
objects and comparing FedSWA(α_LA=1.0).server_aggregate(...) to
FedAvg.server_aggregate(...).
"""
from __future__ import annotations

import pytest
import torch

from fl_oran.federated.algorithms import REGISTRY
from fl_oran.federated.client import ClientUpdate


def _make_updates(seed: int = 0, n_clients: int = 3) -> list[ClientUpdate]:
    """Construct synthetic client updates with a mix of float and integer
    buffers so the ``num_batches_tracked``-style branch is exercised."""
    torch.manual_seed(seed)
    updates: list[ClientUpdate] = []
    for cid in range(n_clients):
        state = {
            "linear.weight": torch.randn(4, 4),
            "linear.bias": torch.randn(4),
            "bn.running_mean": torch.randn(4),
            "bn.num_batches_tracked": torch.tensor(cid + 1, dtype=torch.long),
        }
        updates.append(ClientUpdate(
            client_id=cid,
            state_dict=state,
            num_examples=10 * (cid + 1),  # heterogeneous weighting
            train_loss=0.5,
        ))
    return updates


def _make_global_state(updates: list[ClientUpdate]) -> dict[str, torch.Tensor]:
    """Build a global state with the same keys + shapes as the client updates,
    but distinct values, so the LookAhead step is non-trivial."""
    torch.manual_seed(123)
    sample = updates[0].state_dict
    out: dict[str, torch.Tensor] = {}
    for k, v in sample.items():
        if v.dtype.is_floating_point:
            out[k] = torch.randn_like(v)
        else:
            out[k] = torch.zeros_like(v)
    return out


# ---------------------------------------------------------------------------
# Registration / required-kwarg contract
# ---------------------------------------------------------------------------


def test_fedswa_registered() -> None:
    assert "fedswa" in REGISTRY
    assert REGISTRY["fedswa"].name == "fedswa"


def test_fedswa_requires_alpha_la() -> None:
    """Mirror of test_v7_algo_required_kwargs: ``alpha_la`` is a required
    keyword (no default) so spec_loader rejects under-specified configs."""
    with pytest.raises(TypeError, match="alpha_la"):
        REGISTRY["fedswa"](max_steps=1, batch_size=1)


# ---------------------------------------------------------------------------
# R34-A core invariant: α_LA = 1.0 ⇒ bit-exact FedAvg
# ---------------------------------------------------------------------------


def test_fedswa_alpha_la_one_reduces_to_fedavg_up_to_roundoff() -> None:
    """Headline R34-A invariant. With α_LA=1.0 the LookAhead EMA collapses
    algebraically to ``w_t = v_t``, where ``v_t`` is the FedAvg weighted
    average. The §7.5 narrative ("FedSWA marginally beats FedAvg")
    requires this reduction to hold; if α_LA=1.0 already deviates
    materially from FedAvg, the +0.000379 paired Δ is contaminated by
    an implementation artefact.

    Bit-exact equality does NOT hold because the implementation evaluates
    ``w_g + 1.0 * (v_t - w_g)`` rather than short-circuiting to ``v_t``;
    in float32 the subtraction and re-addition lose the lowest bit. We
    therefore assert "close up to float roundoff" (atol=1e-7) — orders
    of magnitude tighter than the empirical paired Δ ~ 4e-4 the §7.5
    narrative depends on."""
    fedavg = REGISTRY["fedavg"](max_steps=1, batch_size=1)
    fedswa = REGISTRY["fedswa"](max_steps=1, batch_size=1, alpha_la=1.0)
    updates = _make_updates(seed=42)
    global_state = _make_global_state(updates)
    avg_out = fedavg.server_aggregate(
        global_state={k: v.clone() for k, v in global_state.items()},
        updates=updates,
    )
    swa_out = fedswa.server_aggregate(
        global_state={k: v.clone() for k, v in global_state.items()},
        updates=updates,
    )
    assert set(avg_out.keys()) == set(swa_out.keys())
    for k in avg_out:
        torch.testing.assert_close(
            swa_out[k], avg_out[k], rtol=1e-6, atol=1e-7,
            msg=lambda m, k=k: f"FedSWA(α_LA=1.0) diverges from FedAvg on {k}: {m}",
        )


def test_fedswa_alpha_la_paper_default_differs_from_fedavg() -> None:
    """Sanity check that α_LA=1.5 (FedSWA paper eq. 17 default) is
    distinguishable from FedAvg — i.e. the LookAhead step is actually
    being applied. Without this, the previous test could pass trivially
    if server_aggregate simply ignored α_LA."""
    fedavg = REGISTRY["fedavg"](max_steps=1, batch_size=1)
    fedswa = REGISTRY["fedswa"](max_steps=1, batch_size=1, alpha_la=1.5)
    updates = _make_updates(seed=42)
    global_state = _make_global_state(updates)
    avg_out = fedavg.server_aggregate(
        global_state={k: v.clone() for k, v in global_state.items()},
        updates=updates,
    )
    swa_out = fedswa.server_aggregate(
        global_state={k: v.clone() for k, v in global_state.items()},
        updates=updates,
    )
    any_differ = any(
        not torch.allclose(swa_out[k], avg_out[k], atol=1e-6)
        for k in avg_out if avg_out[k].dtype.is_floating_point
    )
    assert any_differ, (
        "FedSWA(α_LA=1.5) must differ from FedAvg on at least one float "
        "parameter; otherwise server_aggregate is silently ignoring α_LA."
    )


def test_fedswa_alpha_la_one_five_matches_explicit_lookahead_formula() -> None:
    """Pin the exact LookAhead update formula::

        new = w_g + α_LA * (v_t - w_g)

    so a future refactor (e.g. swapping to ``w_g * (1-α) + v_t * α`` —
    algebraically equivalent in float64 but not bit-exact in float32)
    fails this test rather than silently shifting numerics."""
    fedswa = REGISTRY["fedswa"](max_steps=1, batch_size=1, alpha_la=1.5)
    updates = _make_updates(seed=42)
    global_state = _make_global_state(updates)
    # Reference: hand-compute v_t = weighted_average then apply LookAhead.
    from fl_oran.federated.aggregation import weighted_average_state_dicts
    v_t = weighted_average_state_dicts(
        [u.state_dict for u in updates],
        [u.num_examples for u in updates],
    )
    swa_out = fedswa.server_aggregate(
        global_state={k: v.clone() for k, v in global_state.items()},
        updates=updates,
    )
    for k, w_g in global_state.items():
        if not w_g.dtype.is_floating_point:
            torch.testing.assert_close(swa_out[k], v_t[k])
            continue
        expected = w_g + 1.5 * (v_t[k] - w_g)
        torch.testing.assert_close(
            swa_out[k], expected, rtol=0.0, atol=0.0,
            msg=lambda m, k=k: f"FedSWA LookAhead formula mismatch on {k}: {m}",
        )


# ---------------------------------------------------------------------------
# Non-float buffers (e.g. BN num_batches_tracked) must be copied not blended
# ---------------------------------------------------------------------------


def test_fedswa_preserves_non_float_buffers_via_copy() -> None:
    """Integer buffers like ``num_batches_tracked`` cannot be blended via
    α_LA arithmetic; the implementation must short-circuit and copy from
    v_t. Without this, BN-bearing models would crash at server_aggregate."""
    fedswa = REGISTRY["fedswa"](max_steps=1, batch_size=1, alpha_la=1.5)
    updates = _make_updates(seed=42)
    global_state = _make_global_state(updates)
    swa_out = fedswa.server_aggregate(
        global_state={k: v.clone() for k, v in global_state.items()},
        updates=updates,
    )
    nbt = swa_out["bn.num_batches_tracked"]
    assert nbt.dtype == torch.long, (
        "Non-float buffer dtype must survive server_aggregate unchanged"
    )
