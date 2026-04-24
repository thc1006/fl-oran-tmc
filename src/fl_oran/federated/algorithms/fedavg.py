"""FedAvg (McMahan et al. 2017), wrapped as an ``FLAlgorithm`` for the v5 sweep.

The per-client training loop mirrors the behaviour of ``training/fl_v3.py``'s
inline loop (v3 is intentionally left untouched; CLAUDE.md rule #2).
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
class FedAvg:
    """Vanilla FedAvg.

    Client: ``max_steps`` Adam updates starting from the global weights.
    Server: weighted average of client state_dicts, weighted by num_examples.
    """

    name = "fedavg"

    def __init__(
        self,
        *,
        max_steps: int,
        batch_size: int,
        grad_clip: float = 1.0,
        amp_enabled: bool = False,
        amp_dtype: torch.dtype | None = None,
    ) -> None:
        self.max_steps = max_steps
        self.batch_size = batch_size
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
        del round_idx  # FedAvg doesn't vary behavior by round
        cat_c, cont_c, y_c = client_tensors
        cat_g = cat_c.to(device, non_blocking=True)
        cont_g = cont_c.to(device, non_blocking=True)
        y_g = y_c.to(device, non_blocking=True)

        local_model.to(device).train()
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
            torch.nn.utils.clip_grad_norm_(local_model.parameters(), self.grad_clip)
            optimizer.step()
            total_loss += loss.item()

        state = {k: v.detach().cpu() for k, v in local_model.state_dict().items()}
        avg_loss = total_loss / max(self.max_steps, 1)
        log.debug("fedavg client %s: steps=%d batch=%d loss=%.4f",
                  client_id, self.max_steps, self.batch_size, avg_loss)
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
        del global_state  # FedAvg's new state doesn't depend on the previous one
        return weighted_average_state_dicts(
            [u.state_dict for u in updates],
            [u.num_examples for u in updates],
        )
