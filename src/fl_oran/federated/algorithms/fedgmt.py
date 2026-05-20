"""FedGMT (Li et al., ICML 2025) — "One Arrow, Two Hawks: Sharpness-aware
Minimization for Federated Learning via Global Model Trajectory".

OpenReview: https://openreview.net/forum?id=80mK2Mqaph
Reference PyTorch implementation:
    https://github.com/harrylee999/FL-SAM
    system/flcore/clients/clientgmt.py
    system/flcore/servers/servergmt.py

Mechanism (from the reference impl, not paper-paraphrased):

Server keeps an EMA of the global parameters across rounds — the "Global
Model Trajectory" (GMT). Each round:

    1. Sample clients, send (a) current global parameters, (b) current EMA
       parameters, (c) per-client dual variable h_i (FedDyn-style).
    2. Each client trains with a single-backward-pass loss::

           L_total = L_task(z, y)
                   + gamma * tau^2 * KL( softmax(z_local / tau) ||
                                          softmax(z_ema   / tau) )
                   + (1 / beta) * <local_params, dual_variable_i>

       The KL term is a knowledge-distillation regulariser that pulls
       local logits toward the (no-grad) EMA-teacher logits. The dual
       term is the FedDyn alignment correction. Only ONE backward through
       the local model is required — the EMA model runs no-grad.
    3. Server updates dual_variable_list[i] += (w_local_after - w_local_before).
    4. Server FedAvg → global; then global += mean(dual_variable_list);
       then EMA = alpha * EMA + (1 - alpha) * global.

Adaptation for our binary classification setting (single logit head):

The reference impl assumes multi-class softmax. For BCE-with-logits with a
single logit `z`, we treat the prediction as 2-class with logits `[0, z]`
so `F.kl_div(F.log_softmax(...), F.softmax(...))` works unmodified.
Equivalent to the binary KL `p_teacher · log(p_teacher / p_student)`.

Round 1 EMA seeding (faithfulness fix 2026-05-17):

The reference impl ``servergmt.py.__init__`` does
``self.EMA_model = copy.deepcopy(args.model)`` — i.e. EMA is seeded to
the INITIAL random model before any FL round begins. Round-1 clients
therefore train with the KL term active (teacher = initial weights), so
the KL acts as an anchor-to-initial-state regulariser for round 1.

Without access to the orchestrator's pre-round-1 model, we lazy-seed
``self.ema_state`` on the FIRST call to ``client_update`` from the
``local_model`` itself — the orchestrator has just loaded the round's
global parameters into it, which on round 1 IS the initial random
model. Subsequent ``client_update`` calls within round 1 see the
already-seeded ``ema_state`` (matches reference's ``client.EMA =
copy.deepcopy(server.EMA_model)`` broadcast in ``send_models``).

When ``gamma_kl == 0`` we skip the seeding entirely — EMA is never
read by the client, so initialising it would waste memory. The server-
side EMA update still runs in ``server_aggregate`` so that if a future
round changes the algorithm config (it can't, but defensively),
``ema_state`` is still maintained.

Hyperparameters (reference defaults in args.tau / args.gama / args.beta /
args.alpha; reproduced verbatim with our naming):

    alpha_ema:  EMA decay for the trajectory. Paper README: {0.95, 0.995, 0.998}.
                ``ema_new = alpha_ema * ema_old + (1 - alpha_ema) * global``.
                Higher = smoother trajectory = more inertia.
    gamma_kl:   KL weight. README: {0.5, 1.0, 2.0}.
    tau:        KL softmax temperature. Standard distillation: 1-4.
    beta:       Inverse penalty coefficient on the dual term. README: 10.
                Internal coefficient = 1 / beta, so the dual term has
                weight 1/beta (e.g. beta=10 → 0.1).
    n_total_clients: total client count, used to size dual_variable buffer.
                Same semantics as FedDyn.

Test contract: at gamma_kl=0 and beta=inf (effectively dual_coef=0), this
reduces to FedAvg over clients with no server-side trajectory. We assert
this in tests/test_v5_fedgmt.py.
"""
from __future__ import annotations

import copy
from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F

from ..aggregation import weighted_average_state_dicts
from ..client import ClientUpdate
from ...logging_utils import get_logger
from . import register
from ._local_loop import run_local_sgd

log = get_logger(__name__)


def _trainable_float_keys(state: dict[str, torch.Tensor]) -> list[str]:
    """Return keys whose values are float tensors — the ones EMA and dual
    variables should track. Non-float buffers (e.g. num_batches_tracked,
    spiking LIF integer state) are excluded.
    """
    return [k for k, v in state.items() if v.dtype.is_floating_point]


@register
class FedGMT:
    """FedGMT — Global Model Trajectory SAM via single-backward distillation.

    Implementation closely follows harrylee999/FL-SAM (ICML 2025 reference).
    The orchestrator-facing surface matches FedDyn (we share the
    n_total_clients + per-client dual state pattern).
    """

    name = "fedgmt"

    def __init__(
        self,
        *,
        max_steps: int,
        batch_size: int,
        alpha_ema: float,
        gamma_kl: float,
        tau: float,
        beta: float,
        n_total_clients: int,
        grad_clip: float = 1.0,
        amp_enabled: bool = False,
        amp_dtype: torch.dtype | None = None,
    ) -> None:
        if not (0.0 < alpha_ema < 1.0):
            raise ValueError(
                f"FedGMT alpha_ema must be in (0, 1); got {alpha_ema}"
            )
        if gamma_kl < 0.0:
            raise ValueError(f"FedGMT gamma_kl must be >= 0; got {gamma_kl}")
        if tau <= 0.0:
            raise ValueError(f"FedGMT tau must be > 0; got {tau}")
        if beta <= 0.0:
            raise ValueError(f"FedGMT beta must be > 0; got {beta}")
        if n_total_clients <= 0:
            raise ValueError(
                f"FedGMT n_total_clients must be > 0; got {n_total_clients}"
            )

        self.max_steps = max_steps
        self.batch_size = batch_size
        self.alpha_ema = float(alpha_ema)
        self.gamma_kl = float(gamma_kl)
        self.tau = float(tau)
        self.beta = float(beta)
        self.n_total_clients = int(n_total_clients)
        self.grad_clip = grad_clip
        self.amp_enabled = amp_enabled
        self.amp_dtype = amp_dtype

        # Persistent server-side state across rounds.
        # ema_state: dict[name -> tensor] OR None on the first round (no GMT yet).
        # dual_variable: dict[client_id -> dict[name -> tensor]] initialised
        # lazily on first sight of each client (matches the reference impl which
        # pre-allocates a (N, P) zero tensor — we use a sparse-by-client dict for
        # memory efficiency when only a subset of clients participates).
        # _last_delta: dict[client_id -> dict[name -> tensor]] holds the most
        # recent (w_after - w_before) emitted by client_update so the next
        # server_aggregate call can update dual_variable[i] += delta_i.
        self.ema_state: dict[str, torch.Tensor] | None = None
        self.dual_variable: dict[int, dict[str, torch.Tensor]] = {}
        self._last_delta: dict[int, dict[str, torch.Tensor]] = {}

    # ------------------------------------------------------------------
    # Client side
    # ------------------------------------------------------------------

    def _kl_term(
        self,
        logits_local: torch.Tensor,
        logits_ema: torch.Tensor,
    ) -> torch.Tensor:
        """Knowledge-distillation KL between local logits (with grad) and
        EMA-teacher logits (no grad).

        For multi-class outputs (``logits.shape[-1] > 1``) this is the
        standard softmax KL ``KL(softmax(z_local / tau) || softmax(z_ema / tau))``
        used by the reference impl.

        For a single-logit BCE head (``logits.shape[-1] == 1`` or a 1-D
        tensor of shape ``(B,)``) we treat the prediction as 2-class with
        logits ``[0, z]`` so the same softmax-KL machinery applies. This
        is mathematically equivalent to the binary KL
        ``p_t · log(p_t / p_s) + (1 - p_t) · log((1 - p_t) / (1 - p_s))``
        evaluated with Bernoulli(σ(z_ema / tau)) as teacher and
        Bernoulli(σ(z_local / tau)) as student, where σ is the sigmoid.

        Multiplied by tau**2 to match Hinton-distillation conventions
        (gradient magnitude is invariant to tau).
        """
        # Coerce both to (B, C>=2). For our forecaster_v2, the head emits
        # (B, 1) or sometimes (B,). Convert single-logit to 2-class.
        if logits_local.dim() == 1:
            logits_local = logits_local.unsqueeze(-1)
            logits_ema = logits_ema.unsqueeze(-1)
        if logits_local.shape[-1] == 1:
            zeros_local = torch.zeros_like(logits_local)
            zeros_ema = torch.zeros_like(logits_ema)
            logits_local = torch.cat([zeros_local, logits_local], dim=-1)
            logits_ema = torch.cat([zeros_ema, logits_ema], dim=-1)
        log_p_local = F.log_softmax(logits_local / self.tau, dim=-1)
        p_ema = F.softmax(logits_ema / self.tau, dim=-1)
        kl = F.kl_div(log_p_local, p_ema, reduction="batchmean")
        return self.gamma_kl * (self.tau ** 2) * kl

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

        # Snapshot pre-training params so we can emit ``local_update = w_after - w_before``
        # for the server-side dual update. Match reference impl's
        # ``param_to_vector`` semantics by stashing per-name copies.
        pre_state = {
            name: p.detach().clone()
            for name, p in local_model.named_parameters()
        }

        # Round-1 EMA seeding (faithfulness fix per audit 2026-05-17):
        # match reference impl's ``self.EMA_model = copy.deepcopy(args.model)``
        # by lazy-seeding ``ema_state`` from the current ``local_model`` —
        # the orchestrator just loaded round-1's distributed global state
        # (= the initial random model on round 1) into it. Subsequent
        # clients in the same round and across rounds use the same
        # seeded/updated trajectory. Skip when gamma_kl=0 (KL term off).
        if self.ema_state is None and self.gamma_kl > 0.0:
            self.ema_state = {
                name: v.detach().cpu().clone()
                for name, v in local_model.state_dict().items()
                if v.dtype.is_floating_point
            }

        # Build the (no-grad) KL teacher. With the lazy-seed above,
        # ema_state is guaranteed non-None whenever gamma_kl > 0; when
        # gamma_kl == 0 the teacher is unused and we skip the build.
        ema_model: nn.Module | None
        if self.gamma_kl > 0.0 and self.ema_state is not None:
            ema_model = copy.deepcopy(local_model)
            with torch.no_grad():
                for name, p in ema_model.named_parameters():
                    if name in self.ema_state:
                        p.copy_(self.ema_state[name].to(device, non_blocking=True))
            # Do NOT call ema_model.eval(): on Spiking backbones this flips
            # self.training=False, which activates the spike-counter buffer
            # update in spiking_forecaster.SpikingSSMBlock._scan_emit_spikes
            # (line 144 "if not self.training: self.spike_count += ...") for
            # every no-grad teacher forward inside the KL closure. Wasteful
            # (~7 LIF timesteps × 50 inner steps × per-cell allocations) and
            # the counter is never read on the teacher anyway. LSTM and Mamba
            # have no train/eval-mode-sensitive code (dropout=0, no BN) so
            # leaving training=True is a no-op there. Reference impl
            # harrylee999/FL-SAM also does not eval the EMA teacher.
            for p in ema_model.parameters():
                p.requires_grad_(False)
        else:
            ema_model = None

        # Precompute device-resident dual for THIS client so the per-step
        # closure doesn't trigger a fresh CPU->GPU transfer on every step
        # (P0 perf fix: ~2200 H2D transfers / client / round otherwise).
        dual_cpu = self.dual_variable.get(client_id)
        if dual_cpu is None:
            dual_dev: dict[str, torch.Tensor] | None = None
        else:
            dual_dev = {
                name: t.to(device, non_blocking=True)
                for name, t in dual_cpu.items()
            }

        # Loss modifier closure: runs inside ``run_local_sgd``'s autocast
        # context. NOTE on cost: we re-run a forward on ``model`` here to
        # get logits_s. ``run_local_sgd`` already did the forward to
        # compute ``base_loss``, but currently only passes the scalar
        # loss to the modifier — there is no hook to receive the logits
        # directly. That makes our client per-step cost = 2 local forwards
        # + 1 backward + 1 EMA forward (no_grad) vs the reference impl's
        # 1 local forward + 1 backward + 1 EMA forward. For our binary-
        # classification setup with dropout=0, both forwards are
        # deterministic so this is a perf overhead only, not a correctness
        # issue. Future optimisation: extend _local_loop.run_local_sgd's
        # modifier signature to pass logits as well — additive change,
        # gated on a separate task.
        gamma_kl = self.gamma_kl
        beta = self.beta
        inv_beta = 1.0 / beta
        kl_term_fn = self._kl_term

        def loss_modifier(
            model: nn.Module,
            cat_batch: torch.Tensor,
            cont_batch: torch.Tensor,
            base_loss: torch.Tensor,
        ) -> torch.Tensor:
            total = base_loss
            # KL distillation against EMA teacher. Skip entirely when
            # gamma_kl=0 (ablation baseline) OR when no EMA exists yet
            # (first round) — both branches save a local forward + a KL
            # compute.
            if ema_model is not None and gamma_kl > 0.0:
                with torch.no_grad():
                    logits_t = ema_model(cat_batch, cont_batch)
                logits_s = model(cat_batch, cont_batch)
                total = total + kl_term_fn(logits_s, logits_t)
            # Dual term: (1/beta) * <local_params, dual_i>. Uses CURRENT
            # model params (gradient flows through them). Skip when this
            # client's dual is still uninitialised (first sight) — the
            # zero seed contributes nothing but would still pay a
            # named_parameters() walk per step.
            if dual_dev is not None:
                acc = None
                for name, p in model.named_parameters():
                    h = dual_dev.get(name)
                    if h is None:
                        continue
                    term = torch.sum(p * h)
                    acc = term if acc is None else acc + term
                if acc is not None:
                    total = total + inv_beta * acc
            return total

        state, avg_loss = run_local_sgd(
            local_model=local_model,
            client_tensors=client_tensors,
            loss_fn=loss_fn,
            current_lr=current_lr,
            max_steps=self.max_steps,
            batch_size=self.batch_size,
            grad_clip=self.grad_clip,
            amp_enabled=self.amp_enabled,
            amp_dtype=self.amp_dtype,
            device=device,
            loss_modifier=loss_modifier,
        )

        # Emit delta = w_after - w_before (per-named-parameter). The server
        # consumes this in server_aggregate to update dual_variable[i].
        # Use the state_dict returned by run_local_sgd (CPU) for w_after,
        # subtract the pre-snapshot (was on device — bring to CPU first).
        delta = {}
        for name, pre in pre_state.items():
            if name not in state:
                continue
            post_cpu = state[name]
            delta[name] = post_cpu - pre.detach().cpu()
        self._last_delta[client_id] = delta

        # Free the deepcopy ASAP — important on 4060 Ti (16 GiB VRAM).
        del ema_model

        log.debug(
            "fedgmt client %s: steps=%d batch=%d gamma_kl=%.3f tau=%.2f beta=%.2f loss=%.4f",
            client_id, self.max_steps, self.batch_size,
            self.gamma_kl, self.tau, self.beta, avg_loss,
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
        """Three-step server: (1) update per-client dual, (2) FedAvg +
        mean-of-dual correction, (3) update EMA trajectory.
        """
        # Step 1: update dual_variable[i] += local_update_i.
        #
        # ``self._last_delta`` is populated by ``client_update`` calls in the
        # same round, keyed by client_id. We DO NOT clear it here — a future
        # round's client_update overwrites the entry. Stale entries from
        # clients not sampled this round simply aren't consumed.
        #
        # Storage convention: dual_variable[cid] is CPU-resident
        # (client_update brings it to ``device`` per-round in a closure).
        # _last_delta is already CPU (computed from CPU-returning state_dict
        # minus a .cpu() snapshot of pre-state).
        float_keys = _trainable_float_keys(global_state)
        for u in updates:
            cid = u.client_id
            delta = self._last_delta.get(cid)
            if delta is None:
                # Defensive: client_update wasn't called this round, or was
                # called with a different ClientUpdate identity. Skip dual
                # update but don't crash — the FedAvg term is still valid.
                log.warning(
                    "fedgmt: no delta cached for client %s; dual var unchanged",
                    cid,
                )
                continue
            existing = self.dual_variable.get(cid)
            if existing is None:
                # First sight of this client — seed dual at delta (equivalent
                # to ``existing = 0; existing += delta``). Force CPU + clone
                # so subsequent in-place updates don't alias the delta we
                # stash in _last_delta.
                self.dual_variable[cid] = {
                    k: v.detach().cpu().clone() for k, v in delta.items()
                }
            else:
                for k in float_keys:
                    if k in delta and k in existing:
                        existing[k] = existing[k] + delta[k].detach().cpu()

        # Step 2a: standard weighted FedAvg.
        avg = weighted_average_state_dicts(
            [u.state_dict for u in updates],
            [u.num_examples for u in updates],
        )
        # Sanity check on keys (catch silent state_dict skew across archs).
        if set(avg.keys()) != set(global_state.keys()):
            missing = set(global_state.keys()) - set(avg.keys())
            extra = set(avg.keys()) - set(global_state.keys())
            raise ValueError(
                f"FedGMT: client state_dict keys diverge from global_state. "
                f"missing={sorted(missing)} extra={sorted(extra)}"
            )

        # Step 2b: mean-of-dual correction.
        # Reference impl: ``global_para += torch.mean(self.dual_variable_list, dim=0)``
        # i.e. mean over ALL n_total_clients rows (most of which may still be
        # zero seeds). Reproduce that semantics: divide by self.n_total_clients,
        # not by len(self.dual_variable) — a different denominator would
        # double-count under partial participation.
        if self.dual_variable:
            for k in float_keys:
                accum = None
                for dv in self.dual_variable.values():
                    if k in dv:
                        accum = dv[k] if accum is None else accum + dv[k]
                if accum is not None:
                    avg[k] = avg[k] + accum / float(self.n_total_clients)

        # Step 3: EMA trajectory update.
        # Reference: ``EMA = alpha * EMA + (1 - alpha) * global`` where
        # ``EMA`` is initialised in ``__init__`` to ``copy.deepcopy(args.model)``
        # — i.e. the model's initial parameters BEFORE round 1. To match that
        # without changing the orchestrator interface, we seed the EMA on
        # first server_aggregate from ``global_state`` (the parameters the
        # server distributed this round, = the pre-round-1 model when this
        # is the first round) and then perform the standard update.
        # Storage convention: ema_state is CPU-resident — client_update
        # brings it to ``device`` per-round.
        if self.ema_state is None:
            self.ema_state = {
                k: global_state[k].detach().cpu().clone()
                for k in float_keys
                if k in global_state
            }
        for k in float_keys:
            if k not in self.ema_state or k not in avg:
                continue
            ema_old = self.ema_state[k]
            avg_cpu = avg[k].detach().cpu() if avg[k].is_cuda else avg[k].detach()
            self.ema_state[k] = self.alpha_ema * ema_old + (1.0 - self.alpha_ema) * avg_cpu

        log.debug(
            "fedgmt aggregate: %d updates, %d clients in dual buffer, alpha_ema=%.3f",
            len(updates), len(self.dual_variable), self.alpha_ema,
        )
        return avg
