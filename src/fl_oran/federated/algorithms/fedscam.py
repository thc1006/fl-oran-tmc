"""FedSCAM (Rahil et al., arXiv:2601.00853, Jan 2026) —
"Federated Sharpness-Aware Minimization with Clustered Aggregation and
Modulation: Scam-resistant SAM for Robust Federated Optimization in
Heterogeneous Environments".

Paper: https://arxiv.org/abs/2601.00853

Mechanism (Algorithm 1 of the paper):

Each round, every sampled client (a) estimates its heterogeneity from B
pilot mini-batch gradient norms, (b) computes alignment with the previous
global direction, (c) trains locally with a SAM perturbation whose
radius rho_i is modulated **inversely** to its (alignment-adjusted)
heterogeneity, (d) returns its weight update plus a direction summary
to the server. The server reweights clients in aggregation: low-
heterogeneity, well-aligned clients get more credit; high-heterogeneity
or anti-aligned clients get less (the "credit reduction" rule).

Equations (paper §3, reproduced in our notation):

    h_i        = (1/B) sum_b || nabla L(w_t; B_{i,b}) ||_2      # B pilot batches
    c_i        = cos(s_i, u_{t-1})                              # alignment with last
                                                                 #   global direction summary
    h_i^adj    = h_i * max(0, 1 - kappa * c_i)                  # heterogeneity-alignment
                                                                 #   coupling
    rho_i      = rho_max / (1 + alpha_rho * h_i^adj)            # per-client SAM radius

    Standard SAM inner step (Foret et al. 2021 — for ref; eq. is paper-
    standard and not re-derived in FedSCAM):
        w'  = w + rho_i * g_1 / ||g_1||                         # ascent at w
        w_new = w - eta * g_2     where  g_2 = nabla L(w'; batch)

    Server aggregation weight:
        S_i        = N_i * (1 / (1 + gamma * h_i^adj))
                         * max(0, 1 + beta_align * c_i)
        w_{t+1}    = sum_i (S_i / sum_j S_j) * w_i

    Update direction memory (server-side, paper Algorithm 1):
        u_t = Proj_d(Normalize(w_{t+1} - w_t))                  # FROM AGGREGATE
                                                                # (NOT from mean of z_i)
    Client direction summary (used only for OPTIONAL clustering — skipped):
        z_i = Proj_d(Normalize(w_i - w_t))

Two paper-side knobs we DO NOT implement in this minimum-viable version
(see "Deferred (paper-faithful follow-up)" below):

    1. K-means clustering on the {z_i} server-side for "conflict
       dampening". The paper presents this as an OPTIONAL refinement
       (paper §3 Algorithm 1 lines marked "optional"). Our v1 ships the
       core insight (per-client modulation + alignment-aware aggregation)
       and matches paper §3 modulo this refinement.
    2. Random projection of pilot direction summaries to dimension
       d in {256, 512}. With our forecaster_v2 having only ~44K
       parameters total, we keep the full-dimensional s_i / u_t and
       absorb the comparison-cost difference into the per-step compute
       budget — at 44K params * 4 bytes * 5 clients = 880 KB the
       projection saves nothing.

Both are accept-deferral choices, documented here and in tests so a
future paper-faithful PR can lift them.

Hyperparameters (paper-pinned where stated; defaults chosen to match
typical SAM-on-FL settings where the paper leaves them task-dependent):

    rho_max     : 0.05  (paper headline; also tested 0.01, 0.1)
    alpha_rho   : 1.0   (per-client radius modulation strength;
                          paper notes "task-dependent")
    gamma       : 1.0   (aggregation hetero-penalty; tested 0, 1, 5)
    beta_align  : 0.8   (aggregation alignment-boost; tested 0, 0.8, 2)
    kappa       : 1.0   (hetero-alignment coupling)
    B_pilot     : 3     (number of pilot batches for heterogeneity est.)

Test reduction contract (tests/test_v5_fedscam.py):
* rho_max=0 AND gamma=0 AND beta_align=0  -> reduces to FedAvg bitwise
  (no SAM perturbation; aggregation weights collapse to N_i).
* alpha_rho=0  -> rho_i = rho_max uniformly = FedSAM (all clients
  same radius, no per-client modulation).

This file does NOT inherit from FedAvg — the SAM perturbation requires a
custom two-backward-pass local loop that ``_local_loop.run_local_sgd``
does not expose. The aggregation, in contrast, is just a different
choice of weights for ``weighted_average_state_dicts``.
"""
from __future__ import annotations

import math
from typing import Callable

import torch
from torch import nn

from ..aggregation import weighted_average_state_dicts
from ..client import ClientUpdate, _make_optimizer
from ...logging_utils import get_logger
from . import register
from ._local_loop import NonFiniteLossError

log = get_logger(__name__)


def _flatten_grads(model: nn.Module) -> torch.Tensor:
    """Concatenate all param gradients into a single 1-D tensor.

    Used by the pilot phase to compute gradient norm and pilot direction.
    Params with .grad == None contribute nothing (Spiking forecasters can
    have a few non-firing params at init).
    """
    parts: list[torch.Tensor] = []
    for p in model.parameters():
        if p.grad is not None:
            parts.append(p.grad.detach().reshape(-1))
    if not parts:
        return torch.zeros(0)
    return torch.cat(parts)


def _flatten_param_delta(
    post_state: dict[str, torch.Tensor],
    pre_state: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Flat ``w_after - w_before`` over the FLOAT keys of ``pre_state``.

    Skips integer / boolean buffers (e.g. ``num_batches_tracked``) so the
    concat doesn't mix dtypes. ``pre_state`` can be either a
    ``named_parameters()`` dict (params only, all float) or a
    ``state_dict()`` dict (params + buffers) — the float filter makes
    both interchangeable.

    For our 3 backbones (LSTM, Mamba, Spiking-SSM) there are no float
    buffers, so a state_dict() filtered to float keys produces the same
    key set and iteration order as ``named_parameters()`` — required for
    the cosine similarity in FedSCAM's c_i to be well-defined across
    rounds (pilot s_i uses ``model.parameters()`` order; server u_t uses
    this helper's order; both must match).

    Both states are coerced to CPU before subtraction so the result is
    server-storable across rounds.
    """
    parts: list[torch.Tensor] = []
    for name, pre in pre_state.items():
        if not pre.dtype.is_floating_point:
            continue
        if name not in post_state:
            continue
        post_cpu = post_state[name].detach().cpu()
        parts.append((post_cpu - pre.detach().cpu()).reshape(-1))
    if not parts:
        return torch.zeros(0)
    return torch.cat(parts)


@register
class FedSCAM:
    """FedSCAM — per-client SAM radius modulation + alignment-aware aggregation.

    Persistent server state across rounds:
        ``self._last_global_direction``  : flat CPU tensor (or None on
                                            round 1 — alignment defaults to 0)
        ``self._client_meta[cid]``        : per-client {h_i^adj, c_i, z_i}
                                            cached by ``client_update`` and
                                            consumed by ``server_aggregate``
                                            in the same round
    """

    name = "fedscam"

    def __init__(
        self,
        *,
        max_steps: int,
        batch_size: int,
        rho_max: float,
        alpha_rho: float,
        gamma: float,
        beta_align: float,
        kappa: float,
        b_pilot: int = 3,
        grad_clip: float = 1.0,
        amp_enabled: bool = False,
        amp_dtype: torch.dtype | None = None,
    ) -> None:
        if rho_max < 0.0:
            raise ValueError(f"FedSCAM rho_max must be >= 0; got {rho_max}")
        if alpha_rho < 0.0:
            raise ValueError(f"FedSCAM alpha_rho must be >= 0; got {alpha_rho}")
        if gamma < 0.0:
            raise ValueError(f"FedSCAM gamma must be >= 0; got {gamma}")
        if beta_align < 0.0:
            raise ValueError(f"FedSCAM beta_align must be >= 0; got {beta_align}")
        if kappa < 0.0:
            raise ValueError(f"FedSCAM kappa must be >= 0; got {kappa}")
        if b_pilot < 1:
            raise ValueError(f"FedSCAM b_pilot must be >= 1; got {b_pilot}")

        self.max_steps = max_steps
        self.batch_size = batch_size
        self.rho_max = float(rho_max)
        self.alpha_rho = float(alpha_rho)
        self.gamma = float(gamma)
        self.beta_align = float(beta_align)
        self.kappa = float(kappa)
        self.b_pilot = int(b_pilot)
        self.grad_clip = grad_clip
        self.amp_enabled = amp_enabled
        self.amp_dtype = amp_dtype

        # Persistent server-side state. _last_global_direction is CPU-
        # resident so it's safe to keep across rounds even if devices
        # change between rounds (unusual but defensible against multi-
        # GPU rotation in future).
        self._last_global_direction: torch.Tensor | None = None
        # Per-client metadata cached by client_update for server_aggregate.
        # Overwritten each round; stale entries from non-sampled clients
        # are simply not consumed by the matching round's server_aggregate.
        self._client_meta: dict[int, dict] = {}

    # ------------------------------------------------------------------
    # Client side
    # ------------------------------------------------------------------

    def _pilot_phase(
        self,
        local_model: nn.Module,
        cat_g: torch.Tensor,
        cont_g: torch.Tensor,
        y_g: torch.Tensor,
        loss_fn: Callable,
        device: torch.device,
    ) -> tuple[float, torch.Tensor]:
        """Run B pilot mini-batches; return ``(h_i_raw, s_i)``.

        Per FedSCAM paper Algorithm 1:
        * ``h_i_raw`` = (1/B) · Σ_b ||∇L(w; B_{i,b})||_2 — heterogeneity
          estimate, averages the gradient NORMS over B batches.
        * ``v_i`` = the pilot direction, computed on **ONE** batch
          (paper: "compute a low-cost pilot direction v_i on one batch
          (e.g., projected gradient or one-step update)"). We use the
          LAST of the B batches' gradient as v_i — its norm has already
          been counted toward h_i, so reusing it is cost-free.
        * ``s_i`` = Normalize(v_i). We skip the Proj_d random projection
          per the module-level docstring deferral; for ~44K params the
          comparison cost is negligible.

        Fidelity audit fix 2026-05-17: previous version averaged unit
        gradients over all B batches into s_i, which is NOT what the
        paper specifies. Paper Algorithm 1 explicitly uses ONE batch
        for the pilot direction.
        """
        n_local = cat_g.shape[0]
        norms: list[float] = []
        last_flat: torch.Tensor | None = None
        amp_ctx = torch.autocast(
            device_type=device.type,
            dtype=self.amp_dtype or torch.bfloat16,
            enabled=self.amp_enabled,
        )
        for _ in range(self.b_pilot):
            idx = torch.randint(0, n_local, (self.batch_size,), device=device)
            local_model.zero_grad(set_to_none=True)
            with amp_ctx:
                logits = local_model(cat_g[idx], cont_g[idx])
                loss = loss_fn(logits, y_g[idx])
            loss.backward()
            flat = _flatten_grads(local_model)
            n = float(flat.norm().item())
            norms.append(n)
            last_flat = flat  # paper: use ONE batch's grad as v_i
        local_model.zero_grad(set_to_none=True)
        h_i_raw = sum(norms) / max(len(norms), 1)
        # s_i = Normalize(v_i) where v_i = gradient on the last pilot batch.
        # "Normalize" = unit-L2 norm so cosine similarity downstream is
        # well-defined. CPU-resident for cross-round persistence in
        # ``self._last_global_direction``.
        if last_flat is None or last_flat.numel() == 0:
            return h_i_raw, torch.zeros(0)
        last_norm = float(last_flat.norm().item())
        if last_norm > 0:
            s_i = (last_flat / last_norm).cpu()
        else:
            s_i = last_flat.cpu()
        return h_i_raw, s_i

    def _sam_train(
        self,
        local_model: nn.Module,
        cat_g: torch.Tensor,
        cont_g: torch.Tensor,
        y_g: torch.Tensor,
        loss_fn: Callable,
        current_lr: float,
        rho_i: float,
        device: torch.device,
    ) -> float:
        """Run ``self.max_steps`` SAM steps with per-client radius ``rho_i``.

        Standard SAM (Foret et al. 2021):
            forward + loss_1 + backward       # g_1 at w
            ascent: w' = w + rho_i * g_1 / ||g_1||
            forward + loss_2 + backward       # g_2 at w'
            restore: w = w' - rho_i * g_1 / ||g_1||  (subtract SAME ascent step)
            optimizer.step()                  # applies g_2 at restored w

        With ``rho_i == 0`` the ascent step is a no-op (eps=0), reducing
        to plain SGD/Adam — used by the FedAvg-reduction test.

        Returns avg loss across the inner steps (using loss_2).
        """
        n_local = cat_g.shape[0]
        optimizer = _make_optimizer(local_model.parameters(), current_lr, device)
        total_loss = 0.0
        amp_ctx = torch.autocast(
            device_type=device.type,
            dtype=self.amp_dtype or torch.bfloat16,
            enabled=self.amp_enabled,
        )
        local_model.train()
        for step_idx in range(self.max_steps):
            idx = torch.randint(0, n_local, (self.batch_size,), device=device)
            cb = cat_g[idx]
            ob = cont_g[idx]
            yb = y_g[idx]

            # SAM step 1: gradient g_1 at current w
            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                logits_1 = local_model(cb, ob)
                loss_1 = loss_fn(logits_1, yb)
            loss_1.backward()

            # Compute global L2 norm of g_1 across all params.
            # Perf: accumulate per-param sum-of-squares as GPU tensors,
            # stack + sum + .item() once (1 GPU->CPU sync per SAM step)
            # instead of N syncs (one .item() per param). At max_steps=50
            # × ~7 params, this saves ~300 sync points per round per
            # client (audit fix 2026-05-17).
            with torch.no_grad():
                sq_sums: list[torch.Tensor] = []
                for p in local_model.parameters():
                    if p.grad is not None:
                        sq_sums.append(p.grad.detach().pow(2).sum())
                if sq_sums:
                    grad_norm_sq = float(torch.stack(sq_sums).sum().item())
                else:
                    grad_norm_sq = 0.0
            grad_norm = (grad_norm_sq ** 0.5) + 1e-12
            eps = rho_i / grad_norm if rho_i > 0 else 0.0

            # Ascent: w <- w + eps * g_1. Save the ASCENT STEP per param
            # so the restore subtracts the same tensor (bit-exact restore).
            # If eps == 0 (rho_i == 0), skip both ascent and restore to
            # match the FedAvg-reduction contract.
            ascent: list[tuple[nn.Parameter, torch.Tensor | None]] = []
            if eps > 0:
                with torch.no_grad():
                    for p in local_model.parameters():
                        if p.grad is not None:
                            step_p = eps * p.grad.detach()
                            p.data.add_(step_p)
                            ascent.append((p, step_p))
                        else:
                            ascent.append((p, None))

            # SAM step 2: gradient g_2 at perturbed point (or at w if eps==0)
            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                logits_2 = local_model(cb, ob)
                loss_2 = loss_fn(logits_2, yb)
            loss_2.backward()

            # Restore w (subtract the saved ascent step). Bit-exact when
            # tensors are float32; ~1e-7 roundoff possible in bf16 amp,
            # but params live in fp32 and the ascent stayed in fp32.
            if eps > 0:
                with torch.no_grad():
                    for p, step_p in ascent:
                        if step_p is not None:
                            p.data.sub_(step_p)

            torch.nn.utils.clip_grad_norm_(local_model.parameters(), self.grad_clip)
            optimizer.step()
            loss_2_val = float(loss_2.item())
            # NaN/Inf guard: mirrors _local_loop.run_local_sgd's check.
            # FedSCAM's SAM perturbation with adaptive ρ_i can theoretically
            # blow up if ρ_i × ||g₁||⁻¹ is very large (tiny gradient at
            # init) AND the perturbed point produces a runaway loss.
            if not math.isfinite(loss_2_val):
                raise NonFiniteLossError(
                    f"FedSCAM non-finite loss at SAM step "
                    f"{step_idx}/{self.max_steps}: loss_2={loss_2_val!r} "
                    f"(rho_i={rho_i:.4f}). Check rho_max, alpha_rho, "
                    f"kappa hyperparameters."
                )
            total_loss += loss_2_val
        avg_loss = total_loss / max(self.max_steps, 1)
        return avg_loss

    def client_update(
        self,
        *,
        client_id: int,
        local_model: nn.Module,
        client_tensors: tuple[torch.Tensor, ...],
        loss_fn: Callable,
        current_lr: float,
        device: torch.device,
        round_idx: int,
    ) -> ClientUpdate:
        del round_idx
        local_model.to(device)

        # Snapshot pre-training params (= the global state distributed
        # this round). Used to compute z_i = w_after - w_before for the
        # server-side direction update.
        pre_state = {
            name: p.detach().clone()
            for name, p in local_model.named_parameters()
        }

        # Move client tensors to device once. ``_local_loop.run_local_sgd``
        # does this internally; here we duplicate the pattern because we
        # don't reuse the helper (SAM needs a custom inner loop).
        cat_c, cont_c, y_c = client_tensors
        cat_g = cat_c.to(device, non_blocking=True)
        cont_g = cont_c.to(device, non_blocking=True)
        y_g = y_c.to(device, non_blocking=True)

        # --- Phase A: pilot heterogeneity + direction summary ---
        # Fast-path: when rho_max=0 AND gamma=0 AND beta_align=0 the
        # pilot outputs (h_i, s_i) are unused everywhere downstream
        # (rho_i ≡ 0 regardless of h_i; aggregation weights collapse to
        # N_i regardless of h_i and c_i). Skipping the pilot's B random
        # batch draws preserves RNG state alignment with the FedAvg
        # baseline, enabling the bit-exact FedAvg-reduction test in
        # tests/test_v5_fedscam.py. Without this short-circuit, the
        # pilot would consume b_pilot extra RNG draws and the SAM
        # phase's mini-batch indices would diverge from FedAvg's.
        if self.rho_max == 0.0 and self.gamma == 0.0 and self.beta_align == 0.0:
            h_i_raw = 0.0
            s_i = torch.zeros(0)
        else:
            h_i_raw, s_i = self._pilot_phase(
                local_model, cat_g, cont_g, y_g, loss_fn, device,
            )

        # --- Phase B: alignment with previous global direction ---
        last_u = self._last_global_direction
        if (
            last_u is not None
            and last_u.numel() == s_i.numel()
            and s_i.numel() > 0
        ):
            # Both s_i and last_u are unit-norm by construction; their dot
            # product is the cosine similarity. Clamp to [-1, 1] to absorb
            # accumulated float roundoff.
            dot = float(torch.dot(s_i, last_u).item())
            c_i = max(-1.0, min(1.0, dot))
        else:
            c_i = 0.0

        # --- Phase C: per-client SAM radius ---
        h_i_adj = h_i_raw * max(0.0, 1.0 - self.kappa * c_i)
        rho_i = self.rho_max / (1.0 + self.alpha_rho * h_i_adj)

        # --- Phase D: SAM training with rho_i ---
        avg_loss = self._sam_train(
            local_model, cat_g, cont_g, y_g, loss_fn,
            current_lr, rho_i, device,
        )

        # Emit final state (CPU) and direction summary z_i (CPU).
        state = {
            k: v.detach().cpu().clone()
            for k, v in local_model.state_dict().items()
        }
        delta_flat = _flatten_param_delta(state, pre_state)
        delta_norm = float(delta_flat.norm().item())
        if delta_norm > 0:
            z_i = delta_flat / delta_norm
        else:
            # Pathological: training produced no movement at all. Could
            # happen with rho_i=0 + max_steps=0 (smoke test). Fall back
            # to zero direction; server's mean direction will simply not
            # be updated by this client this round.
            z_i = delta_flat

        # Stash per-client metadata for server_aggregate.
        self._client_meta[client_id] = {
            "h_i_adj": h_i_adj,
            "c_i": c_i,
            "z_i": z_i,
            "n_i_pilot": float(h_i_raw),
            "rho_i": rho_i,
        }

        log.debug(
            "fedscam client %s: steps=%d batch=%d h_i=%.4f c_i=%+.3f "
            "rho_i=%.4f loss=%.4f",
            client_id, self.max_steps, self.batch_size,
            h_i_raw, c_i, rho_i, avg_loss,
        )
        return ClientUpdate(
            client_id=client_id,
            state_dict=state,
            num_examples=self.max_steps * self.batch_size,
            train_loss=avg_loss,
        )

    # ------------------------------------------------------------------
    # Server side
    # ------------------------------------------------------------------

    def server_aggregate(
        self,
        *,
        global_state: dict[str, torch.Tensor],
        updates: list[ClientUpdate],
    ) -> dict[str, torch.Tensor]:
        """Aggregation weights per the paper:

            S_i = N_i * 1/(1 + gamma * h_i^adj) * max(0, 1 + beta_align * c_i)

        Plus the global direction update u_t = mean_i z_i for next round.
        """
        # Collect per-client (N_i, h_i^adj, c_i) from the cache.
        # Defensive fallback: if a client's metadata is missing, treat it
        # as h_i^adj=0, c_i=0 — recovers FedAvg semantics for that client.
        weights_S: list[float] = []
        z_list: list[torch.Tensor] = []
        for u in updates:
            n_i = float(u.num_examples)
            m = self._client_meta.get(u.client_id)
            if m is None:
                h_adj = 0.0
                c_i = 0.0
                z_i = None
            else:
                h_adj = float(m["h_i_adj"])
                c_i = float(m["c_i"])
                z_i = m["z_i"]
            hetero_factor = 1.0 / (1.0 + self.gamma * h_adj)
            align_factor = max(0.0, 1.0 + self.beta_align * c_i)
            S = n_i * hetero_factor * align_factor
            weights_S.append(S)
            if z_i is not None and z_i.numel() > 0:
                z_list.append(z_i)

        # Fallback: all weights zero (e.g. all clients strongly anti-
        # aligned with beta_align large enough to zero them out). Fall
        # back to FedAvg's N_i weighting to keep the round productive.
        if sum(weights_S) <= 0.0:
            log.warning(
                "fedscam: all aggregation weights collapsed to zero "
                "(beta_align=%.2f drove align_factor to 0 for every "
                "client); falling back to FedAvg weighting this round.",
                self.beta_align,
            )
            weights_S = [float(u.num_examples) for u in updates]

        new_state = weighted_average_state_dicts(
            [u.state_dict for u in updates],
            weights_S,
        )

        # Sanity check on aggregator keys.
        if set(new_state.keys()) != set(global_state.keys()):
            missing = set(global_state.keys()) - set(new_state.keys())
            extra = set(new_state.keys()) - set(global_state.keys())
            raise ValueError(
                f"FedSCAM: client state_dict keys diverge from global_state. "
                f"missing={sorted(missing)} extra={sorted(extra)}"
            )

        # Update direction memory u_t per FedSCAM paper Algorithm 1:
        #     u_t ← Proj_d(Normalize(w_{t+1} − w_t))
        # i.e. unit-normalised direction of the AGGREGATED global movement
        # this round, NOT the unweighted mean of per-client unit deltas.
        #
        # Fidelity audit fix 2026-05-17: previous version used
        # ``u_t = mean(z_i_unit) / ||mean||`` over the cached client
        # directions, which gives equal weight to every client regardless
        # of S_i. The paper formulation is implicitly S_i-weighted via
        # the new_state itself (= Σ p_i w_i ⇒ Σ p_i (w_i - w_t) ⇒ the
        # S_i-weighted aggregate movement). Use that directly.
        delta_flat = _flatten_param_delta(new_state, global_state)
        un = float(delta_flat.norm().item())
        if un > 0:
            self._last_global_direction = delta_flat / un

        log.debug(
            "fedscam aggregate: weights=[%s], rho_max=%.3f gamma=%.2f beta_align=%.2f "
            "u_t_norm=%.3e",
            ",".join(f"{w:.2e}" for w in weights_S),
            self.rho_max, self.gamma, self.beta_align, un,
        )
        return new_state
