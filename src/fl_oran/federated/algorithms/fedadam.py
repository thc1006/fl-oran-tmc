"""FedAdam (Reddi et al. 2020, ICLR) — Adam at the server over the aggregated
client delta.

Client side is identical to FedAvg (inheritance). Server side:

    Δ      = Σ_i p_i · (w_i - w_g)           (weighted mean of client deltas)
    m      = β1·m + (1-β1)·Δ
    v      = β2·v + (1-β2)·Δ²                (element-wise square)
    w_new  = w_g + server_lr · m / (√v + τ)

``m`` and ``v`` are persistent across rounds and allocated lazily on the
first ``server_aggregate`` call so they match the global-state shape.
"""
from __future__ import annotations

import torch

from ..aggregation import weighted_average_state_dicts
from ..client import ClientUpdate
from ...logging_utils import get_logger
from . import register
from .fedavg import FedAvg

log = get_logger(__name__)


@register
class FedAdam(FedAvg):
    """FedAdam — FedAvg client + Adam-on-delta server."""

    name = "fedadam"

    def __init__(
        self,
        *,
        max_steps: int,
        batch_size: int,
        server_lr: float,
        beta1: float = 0.9,
        beta2: float = 0.99,
        tau: float = 1e-3,
        grad_clip: float = 1.0,
        amp_enabled: bool = False,
        amp_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            max_steps=max_steps,
            batch_size=batch_size,
            grad_clip=grad_clip,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        self.server_lr = float(server_lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.tau = float(tau)
        self.m: dict[str, torch.Tensor] = {}
        self.v: dict[str, torch.Tensor] = {}

    def server_aggregate(
        self,
        *,
        global_state: dict[str, torch.Tensor],
        updates: list[ClientUpdate],
    ) -> dict[str, torch.Tensor]:
        # Weighted average of client states → implied delta = avg_state - global_state.
        avg_state = weighted_average_state_dicts(
            [u.state_dict for u in updates],
            [u.num_examples for u in updates],
        )
        new_state: dict[str, torch.Tensor] = {}
        for key, w_g in global_state.items():
            if not w_g.dtype.is_floating_point:
                # Non-float buffers (e.g. BN num_batches_tracked) — copy through.
                new_state[key] = avg_state[key].clone()
                continue
            delta = avg_state[key] - w_g
            # Lazy-init moments to match the global-state shape/dtype/device.
            if key not in self.m:
                self.m[key] = torch.zeros_like(w_g)
                self.v[key] = torch.zeros_like(w_g)
            self.m[key] = self.beta1 * self.m[key] + (1.0 - self.beta1) * delta
            self.v[key] = self.beta2 * self.v[key] + (1.0 - self.beta2) * (delta * delta)
            new_state[key] = w_g + self.server_lr * self.m[key] / (self.v[key].sqrt() + self.tau)
        log.debug("fedadam aggregate: %d updates, keys=%d", len(updates), len(new_state))
        return new_state
