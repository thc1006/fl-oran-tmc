"""FedDyn (Acar et al. 2021, ICLR) — dynamic regularization for FL.

Each client's local objective is

    L_i(w) - ⟨h_i, w⟩ + (α/2)·‖w - w_t‖²

yielding the gradient correction ``-h_i + α·(w - w_t)`` applied on every
local step. At the end of local training:

    h_i ← h_i - α·(w_l - w_t)

The server maintains a cumulative ``h_accum`` (summed Δh_i across rounds)
that would enter the canonical server update ``w_new = mean(w_i) + h_accum/(α·N)``.
For M2 we report Δh_i via ``ClientUpdate.aux["delta_h_i"]`` and accumulate
on the server; wiring the full ``h_accum`` correction into the returned
weights is deferred until the v5 orchestrator lands (it also decides N).
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
        grad_clip: float = 1.0,
        amp_enabled: bool = False,
        amp_dtype: torch.dtype | None = None,
    ) -> None:
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.alpha = float(alpha)
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
        self._ensure_h_i(client_id, local_model, device)
        h_i = {k: v.to(device) for k, v in self.h_i[client_id].items()}
        alpha = self.alpha

        def feddyn_correction(model: nn.Module) -> None:
            # grad ← grad - h_i + α·(w - w_t).
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

        # Update h_i: h_i ← h_i - α·(w_l - w_t).
        new_h_i: dict[str, torch.Tensor] = {}
        delta_h_i: dict[str, torch.Tensor] = {}
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
        # Accumulate Δh_i across participating clients.
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
        return new_w
