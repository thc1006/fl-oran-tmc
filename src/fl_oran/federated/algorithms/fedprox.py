"""FedProx (Li et al. 2020, MLSys) — adds a proximal term ``(μ/2)·‖w-w_g‖²``
to each client's local objective.

Implementation choice: inject ``μ·(w - w_g)`` directly into ``p.grad`` after
``loss.backward()`` and before ``clip_grad_norm_`` + ``optimizer.step()``.
This is mathematically equivalent to adding the prox term to the loss but
avoids building autograd nodes over every parameter — for a ~1M-param model
this is visibly faster per step.
"""
from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from ..aggregation import weighted_average_state_dicts
from ..client import ClientUpdate
from ...logging_utils import get_logger
from . import register

log = get_logger(__name__)


@register
class FedProx:
    """FedProx with proximal coefficient ``mu``.

    When ``mu == 0`` this is identical to FedAvg (bit-wise, given the same
    RNG / data / optimizer init).
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
        cat_c, cont_c, y_c = client_tensors
        cat_g = cat_c.to(device, non_blocking=True)
        cont_g = cont_c.to(device, non_blocking=True)
        y_g = y_c.to(device, non_blocking=True)

        local_model.to(device).train()
        # Snapshot global weights *after* .to(device) so the prox term's
        # arithmetic stays on the same device as p.grad.
        global_snapshot = {
            name: p.detach().clone()
            for name, p in local_model.named_parameters()
        }

        optimizer = torch.optim.Adam(local_model.parameters(), lr=current_lr)
        n_local = cat_g.shape[0]
        total_loss = 0.0

        amp_ctx = torch.autocast(
            device_type=device.type,
            dtype=self.amp_dtype or torch.bfloat16,
            enabled=self.amp_enabled,
        )
        for _ in range(self.max_steps):
            idx = torch.randint(0, n_local, (self.batch_size,), device=device)
            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                logits = local_model(cat_g[idx], cont_g[idx])
                loss = loss_fn(logits, y_g[idx])
            loss.backward()
            # Prox term: ∇_w (μ/2)‖w - w_g‖² = μ·(w - w_g).
            # Skip when μ==0 so FedProx(mu=0) is numerically identical to FedAvg.
            if self.mu != 0.0:
                for name, p in local_model.named_parameters():
                    if p.grad is not None:
                        p.grad.add_(p.data - global_snapshot[name], alpha=self.mu)
            torch.nn.utils.clip_grad_norm_(local_model.parameters(), self.grad_clip)
            optimizer.step()
            total_loss += loss.item()

        state = {k: v.detach().cpu() for k, v in local_model.state_dict().items()}
        avg_loss = total_loss / max(self.max_steps, 1)
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
