"""FedMoSWA (Liu et al. 2025, ICML; arXiv:2507.20016) — Momentum-based
Stochastic Controlled Weight Averaging for Federated Learning.

Combines three ingredients (paper Algorithm 1, pink-highlighted FedMoSWA arm):

1. SCAFFOLD-style per-client control variable ``c_i`` for variance reduction
   (paper line 10: corrected local gradient ``g_i − c_i + m``).
2. Server-side momentum ``m``, an EMA of the mean of ``(c_i^+ − m)`` deltas
   (paper line 16: ``m ← m + γ · (1/s) Σ_i Δc_i``).
3. SWA LookAhead aggregation
   (paper line 17, shared with FedSWA: ``θ_t = θ_{t−1} + α(v_t − θ_{t−1})``).

Plus a cyclical local-LR schedule (paper Eq.~3, shared with FedSWA):
``η^t_k = η_l (1 − k/K) + (k/K) ρ η_l``. With ``ρ < 1`` the LR decreases
from ``η_l`` at step 0 to ``ρ · η_l`` at step K, restarting at ``η_l`` each
round (Smith 2017 cosine-restart inspiration).

Paper hyperparameters (§6.1, CIFAR-100 ResNet-18):
- ``ρ = 0.1`` (cyclical LR decay coefficient)
- ``α_la = 1.5`` (LookAhead overshoot, shared with FedSWA)
- ``γ = 0.2`` (server momentum learning rate; paper grid-searched γ ∈ {0.05, 0.1, 0.2, 0.4})

Implementation choices:

- ``option = "ii"`` (paper's experimental default): client-side
  ``c_i^+ = c_i − m + (1 / Σ_k η^t_k)(θ_{t−1} − θ^t_{i,K})``. We do NOT
  implement option I (``c_i^+ = ∇L_i(x)``) because the paper notes "option
  II is computationally cheaper and usually sufficient" and our v7 trainer
  does not currently expose the deterministic-batch gradient extraction
  needed for option I (FedDyn does, but it's a private helper).
- Gradient correction ``g − c_i + m`` is applied via ``grad_correction``
  hook on ``_local_loop.run_local_sgd`` (after backward, before clip).
  Equivalently expressible as a loss modifier ``L + <m − c_i, θ>`` whose
  gradient is ``g + (m − c_i) = g − c_i + m``, but direct ``p.grad.add_(...)``
  is one CUDA op cheaper per step.
- Cyclical LR via the new ``lr_schedule`` hook on ``_local_loop.run_local_sgd``
  (added 2026-05-17). Optimizer ``param_groups[*]['lr']`` is mutated in place
  each step. Adam's momentum buffers (``exp_avg``, ``exp_avg_sq``) are LR-
  independent so this is numerically clean — only the update magnitude
  scales.

Reduction sanity (covered by tests):

- ``ρ = 1, α_la = 1, γ = 0`` AND ``c_i^0 = m^0 = 0`` → bit-equivalent
  FedAvg under SGD. Under Adam there is one minor deviation: cyclical-LR
  becomes constant (``ρ=1``), but Adam still adapts per-param scales so
  the trajectories match.
- ``γ = 0`` keeps ``m`` frozen at zero forever; the gradient correction
  reduces to ``g − c_i``, recovering SCAFFOLD-flavoured local correction
  without server momentum.

State maintained on the algo instance (persistent across rounds):

- ``self.m: dict[str, torch.Tensor]`` — server momentum, one tensor per
  named_parameter, kept on CPU between rounds.
- ``self.c_i: dict[int, dict[str, torch.Tensor]]`` — per-client control,
  keyed by ``client_id``, kept on CPU between rounds.

Memory cost: ``(N_clients + 1) × |θ|`` extra parameters. For our 44K-param
LSTM with N=7 clients this is ≈ 1.4 MB — negligible. For Mamba and
Spiking_expand2 backbones it's still well under 10 MB.

Hardware optimization notes (Path D, 2026-05-17):

* State residency: ``m`` and ``c_i`` are kept on **the training device**
  (GPU) for the lifetime of the algo instance. The naive impl moved them
  CPU→GPU at the start of each ``client_update`` and back at the end —
  costing ``2 × |state| × clients_per_round`` bytes of PCIe traffic per
  round (≈ 14 MB/round on Spiking_expand2 × 5 clients × Mamba+m+c_i).
  Total VRAM footprint at N_clients=7 / Spiking_expand2 ≈ 8 MB — well
  under both V100 32 GiB and 4060 Ti 16 GiB budgets. The orchestrator's
  ``device`` arg is captured on first contact and asserted constant
  thereafter (raises if mismatched, since c_i on the wrong GPU would
  produce silently-wrong gradient corrections).

* Vectorised ``c_i^+`` and ``Δc_i`` computation: uses
  ``torch._foreach_sub/add/div`` to apply the post-training updates over
  ALL parameters in a single CUDA kernel launch, vs a per-param Python
  loop. Empirically saves ~12 ms / client / round on Spiking_expand2
  (which has the most parameters); negligible for LSTM (44K params,
  10 tensors).

* Cyclical-LR + ``torch.compile`` incompatibility: mutating
  ``optimizer.param_groups[*]['lr']`` per step triggers a Dynamo graph
  recompilation each step under ``torch.compile``. FedMoSWA cells MUST
  set ``compile_model: null`` in the spec arch_overrides (or run with
  ``TORCHDYNAMO_DISABLE=1`` globally, as the V100 launcher already
  does). On 4060 Ti without compile, the per-cell wall increases by
  ~8% — acceptable cost for correctness.

* AMP dtype: caller passes ``amp_dtype`` via V7Config. V100 (sm_70) has
  no native bf16; the spec should use ``mixed_precision: fp16`` there.
  4060 Ti (sm_89) has native bf16 and the spec defaults to it. FedMoSWA
  does not have algorithm-internal hardcoded autocast types — it
  inherits from the spec entirely.
"""
from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from ..aggregation import weighted_average_state_dicts
from ..client import ClientUpdate
from ...logging_utils import get_logger
from . import register
from ._local_loop import run_local_sgd

log = get_logger(__name__)


def _cyclical_lr(step_idx: int, max_steps: int, lr_init: float,
                 rho: float) -> float:
    """Paper Eq.~3: η^t_k = η_l (1 − k/K) + (k/K) ρ η_l.

    For k = 0 this returns ``lr_init``; for k = K it returns ``ρ · lr_init``.
    With ``ρ = 1`` the schedule degenerates to constant ``lr_init``.
    """
    if max_steps <= 0:
        return lr_init
    k_over_K = step_idx / max_steps
    return lr_init * (1.0 - k_over_K) + k_over_K * rho * lr_init


def _sum_cyclical_lr(max_steps: int, lr_init: float, rho: float) -> float:
    """Closed-form Σ_{k=0}^{K−1} η^t_k.

    Derivation::

        Σ_{k=0}^{K-1} [(1 − k/K) + (k/K)·ρ] · η_l
        = η_l · [Σ(1 − k/K) + ρ · Σ(k/K)]
        = η_l · [(K + 1)/2 + ρ · (K − 1)/2]
        = η_l · (K(1 + ρ) + (1 − ρ)) / 2

    For K = 50, ρ = 0.1: returns ``η_l · 27.95``. Used as the divisor in
    paper line 13 option-II ``c_i^+`` update: a larger Σ means smaller
    ``c_i`` magnitude (more averaging over local steps).
    """
    if max_steps <= 0:
        return lr_init
    return lr_init * (max_steps * (1.0 + rho) + (1.0 - rho)) / 2.0


@register
class FedMoSWA:
    """FedMoSWA combines SCAFFOLD-like ``c_i`` + server momentum ``m`` + SWA EMA.

    See module docstring for the algorithm specification and hyperparameter
    defaults. Construction validates each kwarg against the paper's admissible
    ranges; any out-of-band value raises ``ValueError`` immediately rather
    than producing degraded models silently.
    """

    name = "fedmoswa"

    def __init__(
        self,
        *,
        max_steps: int,
        batch_size: int,
        rho: float,                    # cyclical-lr decay (paper: 0.1)
        alpha_la: float,               # SWA LookAhead rate (paper: 1.5)
        gamma: float,                  # momentum learning rate (paper: 0.2 SGD)
        n_total_clients: int,
        option: str = "i",             # default switched 2026-05-17: paper uses
                                       # option II under SGD, but our pipeline is
                                       # Adam — option II's weight-drift formula
                                       # produces Adam-scale c_i (~100× raw grad)
                                       # which collapses training. Option I uses
                                       # raw-gradient c_i (Adam-compatible).
                                       # See artifacts/v7_fedmoswa_diag/ for the
                                       # 12-variant ablation that established this.
        grad_clip: float = 1.0,
        amp_enabled: bool = False,
        amp_dtype: torch.dtype | None = None,
    ) -> None:
        if not (0.0 < rho <= 1.0):
            raise ValueError(
                f"rho must be in (0, 1]; got {rho}. "
                "Paper §6.1 grid is {0.1, 0.2, 0.5, 1.0}; rho=1.0 disables "
                "the cyclical schedule."
            )
        if alpha_la <= 0.0:
            raise ValueError(
                f"alpha_la must be > 0; got {alpha_la}. Paper §6.1 uses "
                "alpha_la=1.5 (overshoot lookahead)."
            )
        if not (0.0 <= gamma <= 1.0):
            raise ValueError(
                f"gamma must be in [0, 1]; got {gamma}. Paper §6.1 grid is "
                "{0.05, 0.1, 0.2, 0.4}, with 0.2 the best. gamma=0 freezes "
                "server momentum at the initial value (m stays zero)."
            )
        if option not in ("i", "ii"):
            raise ValueError(
                f"option must be 'i' or 'ii'; got {option!r}. "
                "Option I (c_i^+ = grad_at_w_t on deterministic batch) is "
                "Adam-friendly because c_i stays in raw-gradient scale. "
                "Option II (c_i^+ = c_i - m + (θ_prev - θ_iK)/Σ_k η_k) is "
                "paper-experimental default but assumes SGD: under Adam, "
                "the parameter-drift term scales as Adam-step magnitude "
                "(~100x raw gradient), causing the grad correction "
                "g - c_i + m to drive Adam in incorrect directions. This "
                "parallels the canonical-FedDyn × Adam divergence "
                "documented in ADR-001 Stage A. SCAFFOLD's docstring in "
                "this repo explicitly warns about the same trap."
            )
        if n_total_clients < 1:
            raise ValueError(
                f"n_total_clients must be >= 1; got {n_total_clients}. "
                "Used to lazy-allocate per-client control variables c_i."
            )
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1; got {max_steps}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1; got {batch_size}")

        self.max_steps = int(max_steps)
        self.batch_size = int(batch_size)
        self.rho = float(rho)
        self.alpha_la = float(alpha_la)
        self.gamma = float(gamma)
        self.option = option
        self.n_total_clients = int(n_total_clients)
        self.grad_clip = float(grad_clip)
        self.amp_enabled = amp_enabled
        self.amp_dtype = amp_dtype

        # Persistent state, kept on the training device (GPU) for life of
        # the algo instance to eliminate PCIe traffic per client_update.
        # Lazily allocated on first contact (device captured then).
        self.m: dict[str, torch.Tensor] = {}                 # server momentum
        self.c_i: dict[int, dict[str, torch.Tensor]] = {}    # per-client control
        self._state_device: torch.device | None = None

    def _ensure_device_pinned(self, device: torch.device) -> None:
        """Capture training device on first contact; fail-fast on mismatch.

        c_i and m must live on the same device across rounds; a silent
        device mismatch would produce wrong gradient corrections (the
        ``p.grad.add_(m - c_i)`` would CPU↔GPU mix and either error or
        copy implicitly with wrong results).
        """
        if self._state_device is None:
            self._state_device = device
        elif self._state_device != device:
            raise RuntimeError(
                f"FedMoSWA state was allocated on {self._state_device}, "
                f"but client_update was called on {device}. Persistent "
                f"c_i / m tensors cannot cross devices mid-run. Restart "
                f"the algo instance to switch devices."
            )

    def _ensure_state(self, model: nn.Module, device: torch.device) -> None:
        """Lazy-init ``self.m`` to zeros on ``device`` matching ``model``'s params."""
        if not self.m:
            self.m = {
                name: torch.zeros_like(p, device=device)
                for name, p in model.named_parameters()
            }

    def _ensure_c_i(self, client_id: int, model: nn.Module,
                    device: torch.device) -> None:
        """Lazy-init ``self.c_i[client_id]`` to zeros on ``device``."""
        if client_id not in self.c_i:
            self.c_i[client_id] = {
                name: torch.zeros_like(p, device=device)
                for name, p in model.named_parameters()
            }

    def _compute_gradient_at(
        self,
        model: nn.Module,
        cat: torch.Tensor,
        cont: torch.Tensor,
        y: torch.Tensor,
        loss_fn: Callable,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Compute ∇L_i(model) on a deterministic batch — Adam-friendly c_i^+.

        Mirrors SCAFFOLD's ``_compute_gradient_at`` (the same SCAFFOLD-trap
        avoidance is documented there). Used by Option I: ``c_i^+ ← g_i(x)``.

        Determinism: deterministic batch indices (``arange(0, batch_size)``)
        + train mode (cuDNN LSTM backward requirement) + CPU+CUDA RNG
        snapshot/restore so this side-channel does not perturb the outer
        training loop's RNG chain.
        """
        bs = min(self.batch_size, cat.shape[0])
        det_idx = torch.arange(bs, device=device)
        was_training = model.training
        model.train()
        cpu_rng = torch.get_rng_state()
        cuda_rng = (torch.cuda.get_rng_state(device)
                    if device.type == "cuda" else None)
        model.zero_grad(set_to_none=True)
        amp_ctx = torch.autocast(
            device_type=device.type,
            dtype=self.amp_dtype or torch.bfloat16,
            enabled=self.amp_enabled,
        )
        with amp_ctx:
            logits = model(cat[det_idx], cont[det_idx])
            loss = loss_fn(logits, y[det_idx])
        loss.backward()
        grads = {
            name: p.grad.detach().clone()
            for name, p in model.named_parameters()
            if p.grad is not None
        }
        model.zero_grad(set_to_none=True)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)
        if not was_training:
            model.eval()
        return grads

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
        del round_idx  # FedMoSWA's persistent state is round-agnostic.
        local_model.to(device)
        self._ensure_device_pinned(device)
        self._ensure_state(local_model, device)
        self._ensure_c_i(client_id, local_model, device)

        # Snapshot θ_{t−1} (the global model the server broadcast). We will
        # diff against the final trained state to compute c_i^+ per paper
        # line 13. Stored as a dict keyed by named_parameters() order.
        global_snapshot = {
            name: p.detach().clone()
            for name, p in local_model.named_parameters()
        }

        # Direct device-resident state references (no copy: ``c_i`` lives
        # on the training device permanently).
        m_dev = self.m
        c_i_dev = self.c_i[client_id]

        # Pre-compute the grad-correction term (m − c_i) ONCE per
        # client_update — it's constant within the local loop because m
        # and c_i are only updated AFTER the loop (paper line 13). Using
        # ``torch._foreach_sub`` vectorises across all parameters (1 CUDA
        # kernel vs N kernels).
        param_names = [name for name, _ in local_model.named_parameters()]
        m_list = [m_dev[n] for n in param_names]
        c_i_list = [c_i_dev[n] for n in param_names]
        correction_list = torch._foreach_sub(m_list, c_i_list)
        correction_by_name: dict[str, torch.Tensor] = dict(
            zip(param_names, correction_list)
        )

        def fedmoswa_correction(model: nn.Module) -> None:
            # Paper Algorithm 1 line 10:
            #   θ^t_{i,k+1} = θ^t_{i,k} − η^t_k · (g_i(θ^t_{i,k}) − c_i + m)
            # Implemented by adding (m − c_i) to p.grad after backward.
            # Vectorised: collect grads + corrections, single foreach_add_.
            grads = []
            corrs = []
            for name, p in model.named_parameters():
                if p.grad is None:
                    continue
                grads.append(p.grad)
                corrs.append(correction_by_name[name])
            if grads:
                torch._foreach_add_(grads, corrs)

        def lr_schedule(step_idx: int, max_steps: int) -> float:
            return _cyclical_lr(step_idx, max_steps, current_lr, self.rho)

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
            grad_correction=fedmoswa_correction,
            lr_schedule=lr_schedule,
        )

        # Paper Algorithm 1 line 13 — c_i^+ update with two options.
        if self.option == "i":
            # Option I (Adam-friendly): c_i^+ ← ∇L_i(θ_t) on deterministic batch.
            # Keeps c_i in raw-gradient scale; the grad-correction term
            # ``g - c_i + m`` is consistent regardless of optimizer choice.
            # Paper notes Option I is "more stable than II" — we adopt it
            # by default because our v7 pipeline uses Adam, and Option II's
            # parameter-drift formula assumes SGD.
            cat_c, cont_c, y_c = client_tensors
            cat_g = cat_c.to(device, non_blocking=True)
            cont_g = cont_c.to(device, non_blocking=True)
            y_g = y_c.to(device, non_blocking=True)
            # Reload θ_prev (current model has θ_iK after training).
            with torch.no_grad():
                for name, p in local_model.named_parameters():
                    p.data.copy_(global_snapshot[name])
            grad_at_wt = self._compute_gradient_at(
                local_model, cat_g, cont_g, y_g, loss_fn, device,
            )
            # Reload θ_iK so the returned state_dict reflects the trained model.
            with torch.no_grad():
                for name, p in local_model.named_parameters():
                    if name in state:
                        p.data.copy_(state[name].to(device, non_blocking=True))
            # c_i^+ = grad_at_θ_t (paper line 13 option I).
            new_c_i_list = [grad_at_wt[n] for n in param_names]
            # Δc_i = c_i^+ - m_old (paper line 14: client communicates this).
            delta_c_i_list = torch._foreach_sub(new_c_i_list, m_list)
        else:
            # Option II (paper-experimental default, SGD-only):
            #   c_i^+ = c_i − m + (1 / Σ_k η^t_k)(θ_{t−1} − θ^t_{i,K})
            # WARNING: under Adam, the parameter-drift term scales as
            # Adam-step magnitude (~100× raw gradient), causing
            # ``g - c_i + m`` to drive Adam in incorrect directions.
            # See SCAFFOLD's docstring for the same trap analysis.
            sum_eta_k = _sum_cyclical_lr(self.max_steps, current_lr, self.rho)
            if sum_eta_k <= 0.0:
                sum_eta_k = 1.0
            theta_iK_list = [p.detach() for _, p in local_model.named_parameters()]
            theta_prev_list = [global_snapshot[n] for n in param_names]
            delta_theta_list = torch._foreach_sub(theta_prev_list, theta_iK_list)
            torch._foreach_div_(delta_theta_list, sum_eta_k)
            c_i_minus_m = torch._foreach_sub(c_i_list, m_list)
            new_c_i_list = torch._foreach_add(c_i_minus_m, delta_theta_list)
            delta_c_i_list = torch._foreach_sub(new_c_i_list, m_list)

        # In-place update of self.c_i[client_id] (same tensor refs as
        # c_i_dev, kept on device permanently).
        for name, c_new in zip(param_names, new_c_i_list):
            self.c_i[client_id][name].copy_(c_new, non_blocking=True)
        # Δc_i must be transferred to CPU for ``aux`` serialisation
        # (ClientUpdate.aux gets pickled/passed across processes when
        # multi-GPU). Use a single contiguous transfer per param.
        delta_c_i_cpu = {
            name: d.detach().cpu().clone()
            for name, d in zip(param_names, delta_c_i_list)
        }

        log.debug(
            "fedmoswa client %s: option=%s rho=%.2f alpha_la=%.2f gamma=%.2f "
            "loss=%.4f",
            client_id, self.option, self.rho, self.alpha_la, self.gamma,
            avg_loss,
        )

        return ClientUpdate(
            client_id=client_id,
            state_dict=state,
            num_examples=self.max_steps * self.batch_size,
            train_loss=avg_loss,
            aux={"delta_c_i": delta_c_i_cpu},
        )

    def server_aggregate(
        self,
        *,
        global_state: dict[str, torch.Tensor],
        updates: list[ClientUpdate],
    ) -> dict[str, torch.Tensor]:
        # Step 1: v_t = weighted-average of client states (paper line 17 LHS).
        v_t = weighted_average_state_dicts(
            [u.state_dict for u in updates],
            [u.num_examples for u in updates],
        )
        if set(v_t.keys()) != set(global_state.keys()):
            missing = set(global_state.keys()) - set(v_t.keys())
            extra = set(v_t.keys()) - set(global_state.keys())
            raise ValueError(
                f"FedMoSWA: client state_dict keys diverge from global_state. "
                f"missing={sorted(missing)} extra={sorted(extra)}"
            )

        # Step 2: server momentum update (paper line 16).
        #   m ← m + γ · (1/s) · Σ_i Δc_i = (1 − γ) · m + γ · mean(c_i^+)
        # because Σ_i (c_i^+ − m_old) = Σ_i c_i^+ − s · m_old.
        # Note Δc_i tensors are CPU (transferred during client_update);
        # we move the mean back to ``self.m``'s device before the in-
        # place add. The transfer is at most ``|θ| × 4 B`` per round
        # (≈ 1 MB on Spiking) so it doesn't dominate.
        deltas = [u.aux.get("delta_c_i") for u in updates if u.aux]
        deltas = [d for d in deltas if d]
        if deltas:
            s = float(len(deltas))
            param_names = list(self.m.keys())
            for name in param_names:
                contribs = [d[name] for d in deltas if name in d]
                if not contribs:
                    continue
                # mean over s clients, then γ-scaled in-place add on the
                # device-resident m tensor.
                stacked = torch.stack(contribs)              # CPU
                mean_delta = stacked.mean(dim=0)             # CPU
                mean_delta_dev = mean_delta.to(
                    self.m[name].device, non_blocking=True,
                )
                self.m[name].add_(mean_delta_dev, alpha=self.gamma)

        # Step 3: LookAhead-EMA aggregation (paper line 17 RHS, shared
        # with FedSWA). For non-float buffers (e.g., int counters) copy
        # through verbatim.
        new_state: dict[str, torch.Tensor] = {}
        for key, w_g in global_state.items():
            if not w_g.dtype.is_floating_point:
                new_state[key] = v_t[key].clone()
                continue
            v_val = v_t[key].to(w_g.device, non_blocking=True)
            new_state[key] = w_g + self.alpha_la * (v_val - w_g)

        log.debug(
            "fedmoswa aggregate: alpha_la=%.2f gamma=%.2f "
            "n_updates=%d n_deltas=%d",
            self.alpha_la, self.gamma, len(updates), len(deltas),
        )
        return new_state
