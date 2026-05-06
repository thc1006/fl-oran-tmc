"""FedSWA (Liu et al. 2025, ICML; arXiv:2507.20016) — LookAhead-style server EMA.

Client side is identical to FedAvg (inheritance). Server side per
the FedSWA paper eq.~17::

    v_t   = weighted_average_state_dicts(client updates)   # FedAvg result
    θ_t   = θ_{t-1} + α_LA * (v_t - θ_{t-1})

α_LA is the LookAhead extrapolation rate; the FedSWA paper uses
α_LA=1.5 (over-extrapolation past the FedAvg target). α_LA=1.0
recovers FedAvg exactly; α_LA<1.0 dampens; α_LA>1.0 over-shoots.

This minimum-viable implementation OMITS the cyclical local LR
schedule (FedSWA paper eq.~3) and uses our standard warmup-then-
constant client LR. The ablation is documented in the commit
message; the paper's §2.6 mechanism argument applies to the
LookAhead-EMA component, which is the dominant component in
small-N high-participation regimes per Liu et al.~2025.

Per the paper §2.6 + §7.5 prediction (FedSWA underperforms FedAdam
in our regime due to FedAdam's adaptive variance damping), this
implementation is for empirical refutation: a single 5-cell
sweep × LSTM × natural-by-BS that compares paired vs Phase 5 FedAdam.
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
class FedSWA(FedAvg):
    """FedSWA — FedAvg client + LookAhead-EMA server step."""

    name = "fedswa"

    def __init__(
        self,
        *,
        max_steps: int,
        batch_size: int,
        alpha_la: float,                 # required; FedSWA paper uses 1.5 (eq. 17)
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
        self.alpha_la = float(alpha_la)
        # No persistent server-side state needed for plain LookAhead EMA;
        # global_state is provided fresh each round by the orchestrator.

    def server_aggregate(
        self,
        *,
        global_state: dict[str, torch.Tensor],
        updates: list[ClientUpdate],
    ) -> dict[str, torch.Tensor]:
        # Step 1: standard FedAvg averaging → v_t
        v_t = weighted_average_state_dicts(
            [u.state_dict for u in updates],
            [u.num_examples for u in updates],
        )
        # Sanity check matching FedAdam's pattern
        if set(v_t.keys()) != set(global_state.keys()):
            missing = set(global_state.keys()) - set(v_t.keys())
            extra = set(v_t.keys()) - set(global_state.keys())
            raise ValueError(
                f"FedSWA: client state_dict keys diverge from global_state. "
                f"missing={sorted(missing)} extra={sorted(extra)}"
            )
        # Step 2: LookAhead EMA update per key
        new_state: dict[str, torch.Tensor] = {}
        for key, w_g in global_state.items():
            if not w_g.dtype.is_floating_point:
                # Non-float buffers (e.g., num_batches_tracked) — copy through
                new_state[key] = v_t[key].clone()
                continue
            v_val = v_t[key].to(w_g.device, non_blocking=True)
            new_state[key] = w_g + self.alpha_la * (v_val - w_g)
        log.debug("fedswa aggregate (alpha_la=%.2f): %d updates",
                  self.alpha_la, len(updates))
        return new_state
