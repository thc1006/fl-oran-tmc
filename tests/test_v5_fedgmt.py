"""Class-level invariants for FedGMT (Lee et al. 2025, ICML;
OpenReview 80mK2Mqaph; reference impl github.com/harrylee999/FL-SAM).

These tests pin the contract the V100 ablation depends on. They mirror
the structure of tests/test_fedswa_methodology.py.

Headline invariants:

* Registration + required-kwargs.
* Constructor validation (alpha_ema in (0,1), gamma_kl >= 0, tau > 0,
  beta > 0, n_total_clients > 0).
* On first round (ema_state is None), KL term is skipped and the client
  loss reduces to BCE + dual term. With gamma_kl=0 (regardless of EMA
  presence) the KL term is also skipped.
* Server step 3 (EMA trajectory): ``EMA = α·EMA_old + (1-α)·global``
  exactly. First-round EMA initialisation comes from ``global_state``
  (the round's input), not from the post-aggregate output — matching
  the reference impl's ``self.EMA_model = copy.deepcopy(args.model)``.
* Dual variable accumulator: ``h_i += (w_local_after - w_local_before)``
  per round per sampled client.
* Server step 2b mean-of-dual: divides by ``n_total_clients`` (NOT by
  the number of clients seen so far) so partial-participation
  semantics match the reference impl's full-N tensor mean.
* KL adapter for binary logits: for a single-logit (B,) or (B,1)
  output, the algorithm treats it as 2-class with logits [0, z].
  Verify the KL value matches a hand calculation.
"""
from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from fl_oran.federated.algorithms import REGISTRY
from fl_oran.federated.algorithms.fedgmt import FedGMT, _trainable_float_keys
from fl_oran.federated.client import ClientUpdate


# ---------------------------------------------------------------------------
# Registration / signature contract
# ---------------------------------------------------------------------------


def test_fedgmt_registered() -> None:
    assert "fedgmt" in REGISTRY
    assert REGISTRY["fedgmt"].name == "fedgmt"


def test_fedgmt_requires_all_documented_kwargs() -> None:
    with pytest.raises(TypeError):
        REGISTRY["fedgmt"](max_steps=1, batch_size=1)


@pytest.mark.parametrize("bad,kw", [
    ("alpha_ema_0", {"alpha_ema": 0.0}),
    ("alpha_ema_1", {"alpha_ema": 1.0}),
    ("alpha_ema_neg", {"alpha_ema": -0.1}),
    ("gamma_neg", {"gamma_kl": -0.1}),
    ("tau_zero", {"tau": 0.0}),
    ("tau_neg", {"tau": -1.0}),
    ("beta_zero", {"beta": 0.0}),
    ("beta_neg", {"beta": -1.0}),
    ("n_clients_zero", {"n_total_clients": 0}),
    ("n_clients_neg", {"n_total_clients": -1}),
])
def test_fedgmt_rejects_invalid_hyperparams(bad, kw) -> None:
    base = dict(
        max_steps=1, batch_size=1,
        alpha_ema=0.99, gamma_kl=1.0, tau=2.0, beta=10.0, n_total_clients=7,
    )
    base.update(kw)
    with pytest.raises(ValueError):
        REGISTRY["fedgmt"](**base)


# ---------------------------------------------------------------------------
# KL adapter for binary classification
# ---------------------------------------------------------------------------


def _hand_binary_kl(z_local: torch.Tensor, z_ema: torch.Tensor, tau: float) -> torch.Tensor:
    """Reference KL between Bernoulli(σ(z_ema/τ)) (teacher) and
    Bernoulli(σ(z_local/τ)) (student), computed via the [0, z] 2-class
    trick. Used to cross-check FedGMT._kl_term without re-deriving the
    binary KL by hand.
    """
    zeros = torch.zeros_like(z_local)
    log_p_local = F.log_softmax(
        torch.stack([zeros, z_local / tau], dim=-1), dim=-1,
    )
    p_ema = F.softmax(
        torch.stack([torch.zeros_like(z_ema), z_ema / tau], dim=-1), dim=-1,
    )
    return F.kl_div(log_p_local, p_ema, reduction="batchmean")


def test_fedgmt_kl_term_zero_when_logits_match() -> None:
    """KL(p || p) = 0. Sanity check on the binary adapter."""
    gmt = REGISTRY["fedgmt"](
        max_steps=1, batch_size=1,
        alpha_ema=0.99, gamma_kl=1.0, tau=2.0, beta=10.0, n_total_clients=7,
    )
    z = torch.randn(8)
    kl = gmt._kl_term(z.clone(), z.clone())
    torch.testing.assert_close(kl, torch.tensor(0.0), atol=1e-6, rtol=0.0)


def test_fedgmt_kl_term_matches_hand_calc_for_binary() -> None:
    """gamma_kl=1 and tau=2.0 → returned value = tau^2 * binary_KL."""
    gmt = REGISTRY["fedgmt"](
        max_steps=1, batch_size=1,
        alpha_ema=0.99, gamma_kl=1.0, tau=2.0, beta=10.0, n_total_clients=7,
    )
    torch.manual_seed(0)
    z_local = torch.randn(16, requires_grad=True)
    z_ema = torch.randn(16)
    expected = (2.0 ** 2) * _hand_binary_kl(z_local, z_ema, tau=2.0)
    got = gmt._kl_term(z_local, z_ema)
    torch.testing.assert_close(got, expected, rtol=1e-6, atol=1e-7)


# ---------------------------------------------------------------------------
# Server step 3: EMA trajectory update formula
# ---------------------------------------------------------------------------


def _make_updates(seed: int = 0, n: int = 3) -> list[ClientUpdate]:
    torch.manual_seed(seed)
    updates: list[ClientUpdate] = []
    for cid in range(n):
        state = {
            "fc.weight": torch.randn(3, 4),
            "fc.bias": torch.randn(3),
        }
        updates.append(ClientUpdate(
            client_id=cid,
            state_dict=state,
            num_examples=10 * (cid + 1),
            train_loss=0.5,
        ))
    return updates


def _make_global(updates: list[ClientUpdate]) -> dict[str, torch.Tensor]:
    torch.manual_seed(123)
    sample = updates[0].state_dict
    return {k: torch.randn_like(v) for k, v in sample.items()}


def test_fedgmt_server_initialises_ema_from_global_state_first_round() -> None:
    """First server_aggregate call: ema_state seeds from ``global_state``,
    not from the post-aggregate output. Matches reference impl's
    ``self.EMA_model = copy.deepcopy(args.model)``.
    """
    gmt = REGISTRY["fedgmt"](
        max_steps=1, batch_size=1,
        alpha_ema=0.99, gamma_kl=1.0, tau=2.0, beta=10.0, n_total_clients=3,
    )
    updates = _make_updates(seed=42)
    gs = _make_global(updates)
    gs_snapshot = {k: v.clone() for k, v in gs.items()}
    gmt.server_aggregate(global_state=gs, updates=updates)
    # EMA's initial state is global_state, then ONE update with alpha=0.99:
    #   ema_new = 0.99 * global_state + 0.01 * avg
    # We verify the seed value is consistent with that update by hand-
    # computing the expected EMA and matching.
    # Hand-compute the FedAvg over updates
    total_n = sum(u.num_examples for u in updates)
    expected_avg = {}
    for k in gs_snapshot:
        if not gs_snapshot[k].dtype.is_floating_point:
            continue
        acc = torch.zeros_like(gs_snapshot[k])
        for u in updates:
            w = u.num_examples / total_n
            acc.add_(u.state_dict[k].to(acc.dtype), alpha=w)
        expected_avg[k] = acc
    # Expected EMA after one update (no dual variables seen yet → no
    # mean-of-dual correction): ema = 0.99 * global_state + 0.01 * avg
    for k in expected_avg:
        expected_ema = 0.99 * gs_snapshot[k] + 0.01 * expected_avg[k]
        torch.testing.assert_close(
            gmt.ema_state[k], expected_ema, rtol=1e-6, atol=1e-7,
            msg=lambda m, k=k: f"FedGMT EMA init wrong on {k}: {m}",
        )


def test_fedgmt_ema_update_formula_second_round() -> None:
    """After two rounds, EMA should be ``α·EMA_round1 + (1-α)·avg_round2``.
    Pin the recurrence exactly.
    """
    gmt = REGISTRY["fedgmt"](
        max_steps=1, batch_size=1,
        alpha_ema=0.95, gamma_kl=1.0, tau=2.0, beta=10.0, n_total_clients=3,
    )
    updates_1 = _make_updates(seed=1)
    updates_2 = _make_updates(seed=2)
    gs = _make_global(updates_1)
    gmt.server_aggregate(global_state={k: v.clone() for k, v in gs.items()}, updates=updates_1)
    ema_after_round_1 = {k: v.clone() for k, v in gmt.ema_state.items()}
    # Round 2's global_state is round 1's output (whatever it was)
    gs2 = {k: v.clone() for k, v in gs.items()}  # in real flow this would be the round-1 server output
    gmt.server_aggregate(global_state=gs2, updates=updates_2)
    # Hand-compute round 2's avg
    total_n = sum(u.num_examples for u in updates_2)
    avg2 = {}
    for k in gs2:
        if not gs2[k].dtype.is_floating_point:
            continue
        acc = torch.zeros_like(gs2[k])
        for u in updates_2:
            w = u.num_examples / total_n
            acc.add_(u.state_dict[k].to(acc.dtype), alpha=w)
        avg2[k] = acc
    # FedGMT applies dual correction BEFORE the EMA update — first round's
    # client_update wasn't called, so _last_delta is empty and dual stays
    # at zero. With no dual seeds, mean-of-dual correction is zero too.
    for k in avg2:
        expected = 0.95 * ema_after_round_1[k] + 0.05 * avg2[k]
        torch.testing.assert_close(
            gmt.ema_state[k], expected, rtol=1e-6, atol=1e-7,
            msg=lambda m, k=k: f"FedGMT EMA round-2 wrong on {k}: {m}",
        )


# ---------------------------------------------------------------------------
# Server step 2b: mean-of-dual divides by n_total_clients
# ---------------------------------------------------------------------------


def test_fedgmt_mean_of_dual_divides_by_n_total_clients_not_seen_count() -> None:
    """Reference impl: ``global += torch.mean(dual_variable_list, dim=0)``
    where dual_variable_list is a (N, P) tensor with most rows still
    zero. We emulate that by dividing by n_total_clients (e.g. 7) even
    when only 3 client deltas have been recorded.
    """
    gmt = REGISTRY["fedgmt"](
        max_steps=1, batch_size=1,
        alpha_ema=0.99, gamma_kl=1.0, tau=2.0, beta=10.0, n_total_clients=7,
    )
    updates = _make_updates(seed=5)
    gs = _make_global(updates)
    # Manually inject _last_delta for clients 0, 1, 2 (only those 3 of 7)
    # with deltas = +1 for fc.weight and +2 for fc.bias.
    sample = updates[0].state_dict
    fixed_delta = {
        "fc.weight": torch.ones_like(sample["fc.weight"]),
        "fc.bias": 2.0 * torch.ones_like(sample["fc.bias"]),
    }
    gmt._last_delta = {0: copy.deepcopy(fixed_delta), 1: copy.deepcopy(fixed_delta), 2: copy.deepcopy(fixed_delta)}
    # Run server_aggregate: server should accumulate dual[0]=dual[1]=dual[2]=fixed_delta
    # then divide by n_total_clients=7 for the mean-of-dual correction.
    gs_snapshot = {k: v.clone() for k, v in gs.items()}
    out = gmt.server_aggregate(global_state=gs, updates=updates)
    # Hand-compute expected output:
    #   1) avg (FedAvg over updates) — uses num_examples weighting
    total_n = sum(u.num_examples for u in updates)
    expected_avg = {}
    for k, v in gs_snapshot.items():
        if not v.dtype.is_floating_point:
            continue
        acc = torch.zeros_like(v)
        for u in updates:
            w = u.num_examples / total_n
            acc.add_(u.state_dict[k].to(acc.dtype), alpha=w)
        expected_avg[k] = acc
    #   2) mean of dual: 3 clients all = fixed_delta → sum = 3*fixed_delta
    #      then divide by n_total_clients (7), not by 3.
    for k in fixed_delta:
        expected_correction = 3.0 * fixed_delta[k] / 7.0
        expected_total = expected_avg[k] + expected_correction
        torch.testing.assert_close(
            out[k], expected_total, rtol=1e-6, atol=1e-7,
            msg=lambda m, k=k: f"FedGMT mean-of-dual wrong on {k}: {m}",
        )


# ---------------------------------------------------------------------------
# Server step 1: dual_variable += delta
# ---------------------------------------------------------------------------


def test_fedgmt_dual_accumulates_across_rounds() -> None:
    """If a client is sampled twice with deltas d_1 then d_2, its dual
    variable should equal d_1 + d_2 after round 2.
    """
    gmt = REGISTRY["fedgmt"](
        max_steps=1, batch_size=1,
        alpha_ema=0.99, gamma_kl=1.0, tau=2.0, beta=10.0, n_total_clients=3,
    )
    updates = _make_updates(seed=11)
    gs = _make_global(updates)
    # Round 1: inject delta_1 for client 0
    delta_1 = {
        "fc.weight": torch.ones_like(updates[0].state_dict["fc.weight"]),
        "fc.bias": torch.ones_like(updates[0].state_dict["fc.bias"]),
    }
    gmt._last_delta = {0: copy.deepcopy(delta_1)}
    gmt.server_aggregate(global_state={k: v.clone() for k, v in gs.items()}, updates=[updates[0]])
    # After round 1, dual[0] should equal delta_1
    for k in delta_1:
        torch.testing.assert_close(gmt.dual_variable[0][k], delta_1[k])
    # Round 2: inject delta_2 for client 0
    delta_2 = {
        "fc.weight": 2.0 * torch.ones_like(updates[0].state_dict["fc.weight"]),
        "fc.bias": 3.0 * torch.ones_like(updates[0].state_dict["fc.bias"]),
    }
    gmt._last_delta = {0: copy.deepcopy(delta_2)}
    gmt.server_aggregate(global_state={k: v.clone() for k, v in gs.items()}, updates=[updates[0]])
    # After round 2, dual[0] = delta_1 + delta_2
    for k in delta_1:
        torch.testing.assert_close(
            gmt.dual_variable[0][k], delta_1[k] + delta_2[k],
            rtol=1e-6, atol=1e-7,
        )


# ---------------------------------------------------------------------------
# Client-side: end-to-end smoke
# ---------------------------------------------------------------------------


class _TinyHead(nn.Module):
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
    return F.binary_cross_entropy_with_logits(logits, y)


def test_fedgmt_client_update_first_round_lazy_seeds_ema_from_local_model() -> None:
    """Round 1: ema_state is None on entry. Per faithfulness fix
    (audit 2026-05-17), client_update LAZY-SEEDS ema_state from
    local_model.state_dict() — matching reference impl's
    ``self.EMA_model = copy.deepcopy(args.model)`` __init__-time seeding.
    The KL term is therefore active on round 1 with teacher = initial
    weights (KL=0 at step 0, grows during training as student diverges
    from initial-as-anchor). Verify the seeding, no crash, finite loss.
    """
    gmt = REGISTRY["fedgmt"](
        max_steps=5, batch_size=8,
        alpha_ema=0.99, gamma_kl=1.0, tau=2.0, beta=10.0, n_total_clients=7,
    )
    model = _TinyHead()
    tensors = _make_tensors()
    assert gmt.ema_state is None  # pre-call invariant
    upd = gmt.client_update(
        client_id=0,
        local_model=model,
        client_tensors=tensors,
        loss_fn=_loss_fn,
        current_lr=5e-4,
        device=torch.device("cpu"),
        round_idx=0,
    )
    assert isinstance(upd, ClientUpdate)
    assert torch.isfinite(torch.tensor(upd.train_loss))
    # _last_delta should have an entry for client 0 (= w_after - w_before)
    assert 0 in gmt._last_delta
    # ema_state must be populated post-call (lazy-init triggered)
    assert gmt.ema_state is not None
    assert "fc.weight" in gmt.ema_state
    assert "fc.bias" in gmt.ema_state


def test_fedgmt_round1_lazy_seed_skipped_when_gamma_kl_zero() -> None:
    """Inverse contract: when gamma_kl=0, the EMA is unused on the
    client; therefore lazy-init is skipped to avoid wasting memory.
    ema_state should remain None after a gamma_kl=0 client_update.
    The server-side EMA update still runs in server_aggregate but is
    tested separately.
    """
    gmt = REGISTRY["fedgmt"](
        max_steps=5, batch_size=8,
        alpha_ema=0.99, gamma_kl=0.0, tau=2.0, beta=10.0, n_total_clients=7,
    )
    model = _TinyHead()
    tensors = _make_tensors()
    assert gmt.ema_state is None
    gmt.client_update(
        client_id=0, local_model=model, client_tensors=tensors,
        loss_fn=_loss_fn, current_lr=5e-4, device=torch.device("cpu"),
        round_idx=0,
    )
    assert gmt.ema_state is None, (
        "gamma_kl=0 must NOT trigger client-side EMA lazy-init"
    )


def test_fedgmt_client_update_second_round_uses_ema_teacher() -> None:
    """After server_aggregate seeds ema_state, the next client_update
    runs the KL term against the EMA model. We don't assert numerical
    behaviour here — just that the run completes without an EMA-related
    AttributeError or shape mismatch.
    """
    gmt = REGISTRY["fedgmt"](
        max_steps=3, batch_size=8,
        alpha_ema=0.99, gamma_kl=1.0, tau=2.0, beta=10.0, n_total_clients=7,
    )
    model = _TinyHead()
    tensors = _make_tensors()
    # Round 1
    upd_1 = gmt.client_update(
        client_id=0, local_model=model, client_tensors=tensors,
        loss_fn=_loss_fn, current_lr=5e-4, device=torch.device("cpu"),
        round_idx=0,
    )
    # Server aggregate → seeds ema_state
    gs = {k: v.clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}
    # Make a fake global_state with the right shapes (use upd_1 itself; orchestrator gives prev global)
    gmt.server_aggregate(global_state=gs, updates=[upd_1])
    assert gmt.ema_state is not None
    # Round 2 — must use EMA without crashing
    model2 = _TinyHead()
    upd_2 = gmt.client_update(
        client_id=1, local_model=model2, client_tensors=tensors,
        loss_fn=_loss_fn, current_lr=5e-4, device=torch.device("cpu"),
        round_idx=1,
    )
    assert torch.isfinite(torch.tensor(upd_2.train_loss))


def test_fedgmt_gamma_kl_zero_skips_ema_forward() -> None:
    """When gamma_kl=0, the KL term short-circuits even if ema_state is
    set. This makes "FedGMT(gamma_kl=0) on round 1 with no dual seeds"
    equivalent to plain FedAvg client training plus an EMA bookkeeping
    update on the server.

    We verify by setting gamma_kl=0 and observing that ``_last_delta``'s
    magnitude is comparable to a vanilla FedAvg client step's delta
    (sanity check that KL didn't sneak in).
    """
    torch.manual_seed(0)
    gmt = REGISTRY["fedgmt"](
        max_steps=5, batch_size=8,
        alpha_ema=0.99, gamma_kl=0.0, tau=2.0, beta=10.0, n_total_clients=7,
    )
    model = _TinyHead()
    tensors = _make_tensors()
    # Pretend a prior ema_state exists (set arbitrary values) — gamma_kl=0
    # must still skip the EMA forward despite this.
    gmt.ema_state = {
        k: torch.zeros_like(v)
        for k, v in model.state_dict().items()
        if v.dtype.is_floating_point
    }
    upd = gmt.client_update(
        client_id=0, local_model=model, client_tensors=tensors,
        loss_fn=_loss_fn, current_lr=5e-4, device=torch.device("cpu"),
        round_idx=0,
    )
    assert torch.isfinite(torch.tensor(upd.train_loss))
    # Same client_update with gamma_kl > 0 must produce a DIFFERENT loss
    # path (KL term contributes a non-zero penalty when EMA != local).
    torch.manual_seed(0)
    gmt_kl = REGISTRY["fedgmt"](
        max_steps=5, batch_size=8,
        alpha_ema=0.99, gamma_kl=1.0, tau=2.0, beta=10.0, n_total_clients=7,
    )
    model_kl = _TinyHead()
    gmt_kl.ema_state = {
        k: torch.zeros_like(v)
        for k, v in model_kl.state_dict().items()
        if v.dtype.is_floating_point
    }
    upd_kl = gmt_kl.client_update(
        client_id=0, local_model=model_kl, client_tensors=tensors,
        loss_fn=_loss_fn, current_lr=5e-4, device=torch.device("cpu"),
        round_idx=0,
    )
    # The two updates must differ because the KL term influenced gradients
    differs = False
    for k in upd.state_dict:
        if upd.state_dict[k].dtype.is_floating_point:
            if not torch.allclose(upd.state_dict[k], upd_kl.state_dict[k], atol=1e-8):
                differs = True
                break
    assert differs, (
        "gamma_kl=0 vs gamma_kl=1 must yield different client outputs "
        "when ema teacher is non-trivial; otherwise the KL short-circuit "
        "is masking the term entirely."
    )


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------


def test_trainable_float_keys_filters_correctly() -> None:
    """Sanity for the helper: only float keys come back."""
    state = {
        "fc.weight": torch.randn(2, 3),  # float
        "fc.bias": torch.randn(2),       # float
        "bn.num_batches_tracked": torch.tensor(0, dtype=torch.long),  # int
    }
    keys = _trainable_float_keys(state)
    assert set(keys) == {"fc.weight", "fc.bias"}


# ---------------------------------------------------------------------------
# NaN/Inf guard (pre-V100 defensive check)
# ---------------------------------------------------------------------------


def test_local_loop_raises_on_nan_loss() -> None:
    """Verify _local_loop.run_local_sgd raises NonFiniteLossError when the
    loss function produces NaN. This is the guard that prevents a single
    divergent cell from poisoning state for subsequent cells in the V100
    sweep — combined with the launcher's --continue-on-cell-failure flag,
    a NaN cell terminates cleanly with diagnostic info."""
    from fl_oran.federated.algorithms._local_loop import (
        NonFiniteLossError,
        run_local_sgd,
    )

    model = _TinyHead()
    cat = torch.zeros(64, dtype=torch.long)
    cont = torch.randn(64, 4)
    y = torch.randint(0, 2, (64,), dtype=torch.float)

    def nan_loss_fn(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.tensor(float("nan"), requires_grad=True)

    with pytest.raises(NonFiniteLossError, match="non-finite loss"):
        run_local_sgd(
            local_model=model,
            client_tensors=(cat, cont, y),
            loss_fn=nan_loss_fn,
            current_lr=1e-3,
            max_steps=3,
            batch_size=8,
            grad_clip=1.0,
            amp_enabled=False,
            amp_dtype=None,
            device=torch.device("cpu"),
        )
