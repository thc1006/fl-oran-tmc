"""FedProx (Li et al. 2020, MLSys) — adds a proximal term
``(mu / 2) * ||w - w_g|| ** 2`` to each client's local objective.

Implementation choice: inject ``mu * (w - w_g)`` directly into ``p.grad``
after ``loss.backward()`` and before ``clip_grad_norm_`` +
``optimizer.step()``. This is mathematically equivalent to adding the prox
term to the loss but avoids building autograd nodes over every parameter —
for a ~1M-param model this is visibly faster per step.
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
class FedProx:
    """FedProx with proximal coefficient ``mu``.

    When ``mu == 0`` this is identical to FedAvg (bit-wise, given the same
    RNG / data / optimizer init). ``run_local_sgd`` expects the caller to
    have moved the model to ``device``; we do so before snapshotting the
    initial weights.
    """

    name = "fedprox"

    def __init__(
        self,
        *,
        max_steps: int,
        batch_size: int,
        mu: float,
        grad_clip: float = 1.0,
        amp_enabled: bool = False,
        amp_dtype: torch.dtype | None = None,
    ) -> None:
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.mu = float(mu)
        self.grad_clip = grad_clip
        self.amp_enabled = amp_enabled
        self.amp_dtype = amp_dtype

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
        # Move model to device before snapshotting so the snapshot is co-resident
        # with p.grad during the correction step.
        local_model.to(device)
        global_snapshot = {
            name: p.detach().clone()
            for name, p in local_model.named_parameters()
        }
        mu = self.mu

        def prox_correction(model: nn.Module) -> None:
            if mu == 0.0:
                return  # bit-identical to FedAvg
            for name, p in model.named_parameters():
                if p.grad is not None:
                    p.grad.add_(p.data - global_snapshot[name], alpha=mu)

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
            grad_correction=prox_correction,
        )
        log.debug("fedprox client %s: steps=%d batch=%d mu=%.4f loss=%.4f",
                  client_id, self.max_steps, self.batch_size, self.mu, avg_loss)
        return ClientUpdate(
            client_id=client_id,
            state_dict=state,
            num_examples=self.max_steps * self.batch_size,
            train_loss=avg_loss,
        )

    def server_aggregate(
        self,
        *,
        global_state: dict[str, torch.Tensor],
        updates: list[ClientUpdate],
    ) -> dict[str, torch.Tensor]:
        del global_state  # FedProx keeps FedAvg's aggregation
        return weighted_average_state_dicts(
            [u.state_dict for u in updates],
            [u.num_examples for u in updates],
        )
