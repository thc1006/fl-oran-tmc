"""TDD tests for FedDyn canonical (paper-faithful Acar 2021) mode.

Phase 1.5j (Stage B Option E, 2026-04-28).

Background
----------

Our existing FedDyn (commit landing 2026-04-25 final-2) supported two
update modes — Option-I (paper-faithful for SGD) and Option-II
(Adam-friendly variant). After verifying via raw GitHub fetch of
``alpemreacar/FedDyn/utils_methods.py`` (the paper authors' reference
implementation), we discovered the actual canonical formula differs
from BOTH Option-I and Option-II:

* **Local h update**: ``h_i += (w_l - w_g)`` — NO ``alpha`` multiplier,
  POSITIVE sign on the drift term. Our Option-I uses ``- alpha *
  (w_l - w_t)``; Option-II uses ``- alpha * grad_at_w_t``. Both differ
  in sign and in alpha-scaling from canonical.

* **Server step**: ``w_new = avg(client_states) + mean(h over ALL N
  clients)``. Our existing server returns plain FedAvg with a TODO
  saying ``h_accum / (alpha * N)`` — the alpha division is wrong vs
  reference (which has no alpha at the server side).

This file specifies the canonical mode and pins it as the new default
so the paper genuinely benchmarks "FedDyn (Acar 2021)" rather than a
variant. Option-I and Option-II are preserved as opt-in modes for
§appendix C ablation if canonical-Adam fails downstream.

Reference
---------

* Acar et al. 2021, "Federated Learning Based on Dynamic
  Regularization" (ICLR 2021). https://openreview.net/forum?id=B7v4QMR6Z9w
* Author reference impl:
  https://github.com/alpemreacar/FedDyn/blob/master/utils_methods.py
  - Init: ``local_param_list = np.zeros((n_clnt, n_par))``
  - h update: ``local_param_list[clnt] += curr_model_par - cld_mdl_param``
  - Server: ``cld_mdl_param = avg_mdl_param + np.mean(local_param_list, axis=0)``
"""
from __future__ import annotations

import pytest
import torch

from conftest import build_trio as _build_trio


# ---------------------------------------------------------------------------
# 1. Default-mode contract
# ---------------------------------------------------------------------------


def test_feddyn_default_mode_is_canonical():
    """Stage B (2026-04-28) flips default from option_ii (Adam-friendly
    variant) to canonical (paper-faithful Acar 2021). Ensures the
    paper claim 'we evaluate FedDyn' is honest."""
    from fl_oran.federated.algorithms import REGISTRY
    feddyn = REGISTRY["feddyn"](
        max_steps=1, batch_size=1, alpha=0.01, n_total_clients=7,
    )
    assert feddyn.update_mode == "canonical", (
        "FedDyn default must be 'canonical' (paper-ref formula), "
        "not option_i / option_ii"
    )


# ---------------------------------------------------------------------------
# 2. Canonical h update has no alpha and positive sign
# ---------------------------------------------------------------------------


def test_feddyn_canonical_h_update_uses_no_alpha_positive_sign():
    """Per alpemreacar/FedDyn reference impl, the per-round per-client
    h update is::

        local_param_list[clnt] += curr_model_par - cld_mdl_param

    No alpha multiplier; positive sign on (w_l - w_g). Our Option-I
    uses ``- alpha * (w_l - w_t)`` (sign + alpha differ); Option-II
    uses ``- alpha * grad_at_wt`` (a different formula entirely).

    This test pins the canonical formula by checking ``delta_h_i ==
    (w_local - w_global)`` after one client_update.
    """
    from fl_oran.federated.algorithms import REGISTRY
    model, tensors, loss_fn = _build_trio(seed=42)
    # Snapshot initial weights = w_global (before training).
    w_global = {
        name: p.detach().clone() for name, p in model.named_parameters()
    }
    feddyn = REGISTRY["feddyn"](
        max_steps=3, batch_size=4, alpha=0.5,
        n_total_clients=7, update_mode="canonical",
    )
    u = feddyn.client_update(
        client_id=0, local_model=model, client_tensors=tensors,
        loss_fn=loss_fn, current_lr=0.01, device=torch.device("cpu"),
        round_idx=1,
    )
    # Post-training weights = w_local.
    w_local = {
        name: p.detach().clone() for name, p in model.named_parameters()
    }
    # Canonical formula: delta_h_i == (w_local - w_global)
    # (h_i started at zero, so h_new = 0 + (w_l - w_g) = (w_l - w_g)).
    delta = u.aux["delta_h_i"]
    for name in w_global:
        if name not in delta:
            continue
        expected = (w_local[name] - w_global[name]).detach().cpu()
        torch.testing.assert_close(
            delta[name], expected, rtol=1e-5, atol=1e-6,
            msg=lambda m, n=name: f"canonical h update mismatch on {n}: {m}",
        )


# ---------------------------------------------------------------------------
# 3. Canonical server step applies h_accum / N_total
# ---------------------------------------------------------------------------


def test_feddyn_canonical_server_applies_h_over_n_total():
    """Per reference impl::

        cld_mdl_param = avg_mdl_param + np.mean(local_param_list, axis=0)

    where ``np.mean(local_param_list, axis=0)`` averages h over ALL N
    clients (zeros for unvisited). Equivalent to ``h_accum / N_total``
    since unvisited clients contribute zeros to both formulations.

    No alpha division at the server side (the previous TODO comment
    saying ``h_accum / (alpha * N)`` was incorrect).
    """
    from fl_oran.federated.algorithms import REGISTRY
    from fl_oran.federated.aggregation import weighted_average_state_dicts
    model, tensors, loss_fn = _build_trio(seed=42)
    feddyn = REGISTRY["feddyn"](
        max_steps=3, batch_size=4, alpha=0.5,
        n_total_clients=4, update_mode="canonical",
    )
    u = feddyn.client_update(
        client_id=0, local_model=model, client_tensors=tensors,
        loss_fn=loss_fn, current_lr=0.01, device=torch.device("cpu"),
        round_idx=1,
    )
    # Reconstruct the FedAvg-only baseline.
    fedavg_only = weighted_average_state_dicts(
        [u.state_dict], [u.num_examples],
    )
    # Canonical server returns avg + h_accum / N_total.
    global_state = {k: v.clone() for k, v in u.state_dict.items()}
    new_w = feddyn.server_aggregate(
        global_state=global_state, updates=[u],
    )
    # Difference (new_w - fedavg_only) should equal h_accum / 4.
    # Since only client 0 ran, h_accum = delta_h_i for client 0 =
    # w_local - w_global.
    h_accum = feddyn.h_accum
    for name in fedavg_only:
        if name not in h_accum or not fedavg_only[name].dtype.is_floating_point:
            continue
        diff = (new_w[name] - fedavg_only[name]).detach().cpu()
        expected = (h_accum[name] / 4.0).detach().cpu()
        torch.testing.assert_close(
            diff, expected, rtol=1e-5, atol=1e-6,
            msg=lambda m, n=name: (
                f"canonical server step diff mismatch on {n}: {m}; "
                f"expected (new_w - fedavg) == h_accum / N_total"
            ),
        )


# ---------------------------------------------------------------------------
# 4. Canonical at alpha=0 does NOT reduce to FedAvg (h still grows)
# ---------------------------------------------------------------------------


def test_feddyn_canonical_alpha_zero_diverges_from_fedavg():
    """Unlike Option-I/II (which return FedAvg trajectory at alpha=0
    because h stays at zero), canonical FedDyn's h update has NO
    alpha factor, so h grows by ``(w_l - w_g)`` even at alpha=0. The
    server then applies ``h/N`` to the new global weight, diverging
    from FedAvg.

    This test pins that semantic difference — important for §method
    where we describe canonical vs Option-I/II ablation.
    """
    from fl_oran.federated.algorithms import REGISTRY
    model_a, tensors, loss_fn = _build_trio(seed=42)
    model_b, _, _ = _build_trio(seed=42)
    fedavg = REGISTRY["fedavg"](max_steps=5, batch_size=4, grad_clip=1.0)
    feddyn = REGISTRY["feddyn"](
        max_steps=5, batch_size=4, grad_clip=1.0,
        alpha=0.0, n_total_clients=4, update_mode="canonical",
    )
    device = torch.device("cpu")
    torch.manual_seed(9)
    u_avg = fedavg.client_update(
        client_id=1, local_model=model_a, client_tensors=tensors,
        loss_fn=loss_fn, current_lr=0.01, device=device, round_idx=1,
    )
    torch.manual_seed(9)
    u_dyn = feddyn.client_update(
        client_id=1, local_model=model_b, client_tensors=tensors,
        loss_fn=loss_fn, current_lr=0.01, device=device, round_idx=1,
    )
    # Client trajectory matches FedAvg at alpha=0 (no local correction).
    for k in u_avg.state_dict:
        torch.testing.assert_close(
            u_avg.state_dict[k], u_dyn.state_dict[k],
        )
    # But canonical FedDyn's server step diverges because h_i_new =
    # 0 + (w_l - w_g) is non-zero, and server adds h_accum / N_total.
    avg_state = {k: v.clone() for k, v in u_avg.state_dict.items()}
    fedavg_w = fedavg.server_aggregate(
        global_state=avg_state, updates=[u_avg],
    )
    feddyn_w = feddyn.server_aggregate(
        global_state=avg_state, updates=[u_dyn],
    )
    # Some parameter must differ.
    any_differ = any(
        not torch.allclose(fedavg_w[k], feddyn_w[k], atol=1e-7)
        for k in fedavg_w if fedavg_w[k].dtype.is_floating_point
    )
    assert any_differ, (
        "Canonical FedDyn at alpha=0 should still diverge from FedAvg "
        "via h_accum / N at server step (h grows because no alpha)"
    )


# ---------------------------------------------------------------------------
# 5. Option-I / Option-II preserved as opt-in (regression)
# ---------------------------------------------------------------------------


def test_feddyn_option_i_alpha_zero_still_reduces_to_fedavg():
    """Regression: when explicitly using update_mode='option_i' with
    alpha=0, FedDyn must still reduce bit-exactly to FedAvg. This was
    the invariant the original test_v5_feddyn.py pinned; we preserve
    it via opt-in mode now that default flipped to canonical."""
    from fl_oran.federated.algorithms import REGISTRY
    model_a, tensors, loss_fn = _build_trio(seed=42)
    model_b, _, _ = _build_trio(seed=42)
    fedavg = REGISTRY["fedavg"](max_steps=5, batch_size=4, grad_clip=1.0)
    feddyn = REGISTRY["feddyn"](
        max_steps=5, batch_size=4, grad_clip=1.0,
        alpha=0.0, n_total_clients=4, update_mode="option_i",
    )
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


# ---------------------------------------------------------------------------
# 6. n_total_clients propagation contract
# ---------------------------------------------------------------------------


def test_feddyn_canonical_n_total_clients_changes_server_scaling():
    """Different n_total_clients values must produce proportionally
    different server-step adjustments. With h_accum constant and
    ``new_w = avg + h_accum / N``, doubling N halves the adjustment."""
    from fl_oran.federated.algorithms import REGISTRY
    from copy import deepcopy
    model, tensors, loss_fn = _build_trio(seed=42)

    def run_one(n_total):
        m = deepcopy(model)
        feddyn = REGISTRY["feddyn"](
            max_steps=3, batch_size=4, alpha=0.5,
            n_total_clients=n_total, update_mode="canonical",
        )
        torch.manual_seed(7)
        u = feddyn.client_update(
            client_id=0, local_model=m, client_tensors=tensors,
            loss_fn=loss_fn, current_lr=0.01, device=torch.device("cpu"),
            round_idx=1,
        )
        global_state = {k: v.clone() for k, v in u.state_dict.items()}
        return feddyn.server_aggregate(
            global_state=global_state, updates=[u],
        ), u, feddyn.h_accum

    new_w_4, u_4, h_accum_4 = run_one(4)
    new_w_8, u_8, h_accum_8 = run_one(8)
    # h_accum identical (same client trajectory, same h-update formula).
    for name in h_accum_4:
        torch.testing.assert_close(h_accum_4[name], h_accum_8[name])
    # Server adjustment should be h_accum/4 vs h_accum/8 → adjustment
    # at N=4 is 2× the adjustment at N=8.
    avg_state = {k: v.clone() for k, v in u_4.state_dict.items()}
    for name in new_w_4:
        if not new_w_4[name].dtype.is_floating_point:
            continue
        adj_4 = (new_w_4[name] - avg_state[name]).detach()
        adj_8 = (new_w_8[name] - avg_state[name]).detach()
        # adj_4 should be ~2 * adj_8.
        if float(adj_8.abs().sum()) < 1e-9:
            continue  # zero adjustment, skip ratio check
        torch.testing.assert_close(
            adj_4, 2.0 * adj_8, rtol=1e-4, atol=1e-7,
            msg=lambda m, n=name: (
                f"adj at N=4 should equal 2 * adj at N=8 on {n}: {m}"
            ),
        )


# ---------------------------------------------------------------------------
# 7. Validation
# ---------------------------------------------------------------------------


def test_feddyn_canonical_invalid_update_mode_raises():
    from fl_oran.federated.algorithms import REGISTRY
    with pytest.raises(ValueError, match=r"update_mode"):
        REGISTRY["feddyn"](
            max_steps=1, batch_size=1, alpha=0.01,
            n_total_clients=7, update_mode="bogus_mode",
        )


def test_feddyn_canonical_n_total_clients_must_be_positive():
    from fl_oran.federated.algorithms import REGISTRY
    with pytest.raises(ValueError, match=r"n_total_clients"):
        REGISTRY["feddyn"](
            max_steps=1, batch_size=1, alpha=0.01,
            n_total_clients=0, update_mode="canonical",
        )
