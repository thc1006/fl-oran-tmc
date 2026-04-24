"""FedDyn (Acar et al. 2021, ICLR) — dynamic regularization for FL.

Each client's local objective is::

    L_i(w) - <h_i, w> + (alpha / 2) * ||w - w_t|| ** 2

yielding the gradient correction ``-h_i + alpha*(w - w_t)`` applied on every
local step. Two modes for the ``h_i`` update after local training:

- ``update_mode="option_ii"`` (**default, Adam-friendly**)::

      h_i <- h_i - alpha * grad_{L_i}(w_t)

  Uses a deterministic mini-batch gradient at ``w_t`` so ``h_i`` scales
  with the true data gradient — magnitude is invariant to the local
  optimiser choice. Recommended whenever the client uses Adam (as we do).

- ``update_mode="option_i"`` (**paper-faithful, assumes SGD**)::

      h_i <- h_i - alpha * (w_l - w_t)

  Accumulates weight drift. Under Adam, drift scales as ``eta`` per
  step (Adam normalises per-dim), which is ~100x larger than gradient
  magnitude; ``h_i`` then dominates the gradient correction and the
  client effectively minimises the regulariser, not the loss. Kept for
  paper-faithful SGD baselines only.

The server maintains a cumulative ``h_accum`` (summed delta_h_i across
rounds) that would enter the canonical server update
``w_new = mean(w_i) + h_accum / (alpha * N)``. For M2 we report delta_h_i
via ``ClientUpdate.aux["delta_h_i"]`` and accumulate on the server; wiring
the full ``h_accum`` correction into the returned weights is deferred
until the v5 orchestrator decides N.
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


@register
class FedDyn:
    """FedDyn with dynamic-regularization coefficient ``alpha``."""

    name = "feddyn"

    def __init__(
        self,
        *,
        max_steps: int,
        batch_size: int,
        alpha: float,
        update_mode: str = "option_ii",
        grad_clip: float = 1.0,
        amp_enabled: bool = False,
        amp_dtype: torch.dtype | None = None,
    ) -> None:
        if update_mode not in ("option_i", "option_ii"):
            raise ValueError(
                f"update_mode must be 'option_i' or 'option_ii', got {update_mode!r}"
            )
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.alpha = float(alpha)
        self.update_mode = update_mode
        self.grad_clip = grad_clip
        self.amp_enabled = amp_enabled
        self.amp_dtype = amp_dtype
        self.h_i: dict[int, dict[str, torch.Tensor]] = {}
        self.h_accum: dict[str, torch.Tensor] = {}

    def _ensure_h_i(self, client_id: int, model: nn.Module,
                     device: torch.device) -> None:
        if client_id not in self.h_i:
            self.h_i[client_id] = {
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
        """Grad of loss_fn at model's current weights on a deterministic batch.

        Eval mode + arange index → does not perturb global torch RNG.
        """
        bs = min(self.batch_size, cat.shape[0])
        det_idx = torch.arange(bs, device=device)
        was_training = model.training
        model.eval()
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
        if was_training:
            model.train()
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
        del round_idx
        local_model.to(device)
        global_snapshot = {
            name: p.detach().clone()
            for name, p in local_model.named_parameters()
        }
        global_snapshot_state = {
            k: v.detach().clone()
            for k, v in local_model.state_dict().items()
        }
        self._ensure_h_i(client_id, local_model, device)
        h_i = {k: v.to(device) for k, v in self.h_i[client_id].items()}
        alpha = self.alpha

        def feddyn_correction(model: nn.Module) -> None:
            # grad <- grad - h_i + alpha*(w - w_t).
            # With alpha=0 AND h_i=0 this is a no-op (= FedAvg trajectory).
            for name, p in model.named_parameters():
                if p.grad is None:
                    continue
                if alpha != 0.0:
                    p.grad.add_(p.data - global_snapshot[name], alpha=alpha)
                p.grad.sub_(h_i[name])

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
            grad_correction=feddyn_correction,
        )

        # Update h_i.
        new_h_i: dict[str, torch.Tensor] = {}
        delta_h_i: dict[str, torch.Tensor] = {}
        if self.update_mode == "option_ii":
            # h_i <- h_i - alpha * grad_{L_i}(w_t). Compute grad at w_t on a
            # deterministic batch (optimizer-agnostic magnitude).
            cat_c, cont_c, y_c = client_tensors
            cat_g = cat_c.to(device, non_blocking=True)
            cont_g = cont_c.to(device, non_blocking=True)
            y_g = y_c.to(device, non_blocking=True)
            local_model.load_state_dict(global_snapshot_state, strict=True)
            grad_at_wt = self._compute_gradient_at(
                local_model, cat_g, cont_g, y_g, loss_fn, device,
            )
            local_model.load_state_dict(
                {k: v.to(device) for k, v in state.items()}, strict=True,
            )
            for name in grad_at_wt:
                h_i_plus = h_i[name] - alpha * grad_at_wt[name]
                new_h_i[name] = h_i_plus.detach().cpu()
                delta_h_i[name] = (h_i_plus - h_i[name]).detach().cpu()
        else:  # option_i — paper-faithful for SGD only
            for name, p in local_model.named_parameters():
                diff = p.detach() - global_snapshot[name]  # w_l - w_t
                h_i_plus = h_i[name] - alpha * diff
                new_h_i[name] = h_i_plus.detach().cpu()
                delta_h_i[name] = (h_i_plus - h_i[name]).detach().cpu()
        self.h_i[client_id] = new_h_i

        log.debug("feddyn client %s: steps=%d batch=%d alpha=%.4f loss=%.4f",
                  client_id, self.max_steps, self.batch_size, self.alpha, avg_loss)
        return ClientUpdate(
            client_id=client_id,
            state_dict=state,
            num_examples=self.max_steps * self.batch_size,
            train_loss=avg_loss,
            aux={"delta_h_i": delta_h_i},
        )

    def server_aggregate(
        self,
        *,
        global_state: dict[str, torch.Tensor],
        updates: list[ClientUpdate],
    ) -> dict[str, torch.Tensor]:
        del global_state
        new_w = weighted_average_state_dicts(
            [u.state_dict for u in updates],
            [u.num_examples for u in updates],
        )
        # Accumulate delta_h_i across participating clients.
        deltas = [u.aux.get("delta_h_i") for u in updates if u.aux]
        deltas = [d for d in deltas if d]
        if not deltas:
            return new_w
        if not self.h_accum:
            # Lazy-init to match delta shape/dtype/device (CPU).
            self.h_accum = {k: torch.zeros_like(v) for k, v in deltas[0].items()}
        for name in self.h_accum:
            for d in deltas:
                if name in d:
                    self.h_accum[name] = self.h_accum[name] + d[name]
        log.debug("feddyn aggregate: accumulated %d delta_h_i into h_accum",
                  len(deltas))
        # TODO(M3): paper-faithful server step is
        #   new_w[name] += h_accum[name] / (alpha * N)
        # where N is the total number of clients ever seen. That requires the
        # sweep orchestrator to thread through N. Today we return plain FedAvg
        # weights and keep h_accum as a side accumulator that downstream
        # analysis can inspect.
        return new_w
