"""FedBN (Li et al. 2021, ICLR; arXiv:2102.07623) wrapped as an FLAlgorithm.

The FedBN modification to FedAvg: server-side aggregation SKIPS per-client
personalised parameters — canonically, BatchNorm weights + running
statistics. Each client keeps its own BN params locally across rounds.

On our 3 architectures (LSTM, Mamba, Spiking-SSM), no normalisation layers
are present (verified via grep on src/fl_oran/models/*.py — only the v1
baseline mlp_deep.py uses BatchNorm). Therefore FedBN's parameter-skipping
aggregation reduces to FedAvg's complete aggregation by construction.

We document the reduction in artifacts/audit/fedbn_reduces_to_fedavg.md
and verify it empirically in tests/test_fedbn_methodology.py. Reviewer MC3
ask is answered by mechanism (no BN → no skipped params), with FedAvg
already present in the 5-algorithm baseline.
"""
from __future__ import annotations

import re
from typing import Callable

import torch
from torch import nn

from ..aggregation import weighted_average_state_dicts
from ..client import ClientUpdate
from ...logging_utils import get_logger
from . import register
from ._local_loop import run_local_sgd

log = get_logger(__name__)


# Patterns identifying personalised (= NOT aggregated by server) parameters.
# Matches the canonical FedBN definition (BN params + running stats) plus
# common modern normalisation layers (LayerNorm, RMSNorm, GroupNorm,
# InstanceNorm). Conservative — better to skip a borderline param than to
# aggregate a per-client distribution-statistic param.
_PERSONALISED_NAME_PATTERNS = (
    # Match `bn`, `bn1`, `bn2`, etc. when separated by . or _ from neighbours.
    # Excludes false positives like `embeddings.bs_id.weight` (no `bn` token).
    re.compile(r"(?:^|[._])bn\d*(?:[._]|$)"),
    re.compile(r"batchnorm", re.IGNORECASE),
    re.compile(r"(?:^|[._])norm\d*(?:[._]|$)", re.IGNORECASE),
    re.compile(r"layernorm", re.IGNORECASE),
    re.compile(r"rmsnorm", re.IGNORECASE),
    re.compile(r"groupnorm", re.IGNORECASE),
    re.compile(r"instancenorm", re.IGNORECASE),
    re.compile(r"running_mean$"),
    re.compile(r"running_var$"),
    re.compile(r"num_batches_tracked$"),
)


def _is_personalised_param(name: str) -> bool:
    """Return True iff this parameter name should NOT participate in server
    aggregation (per FedBN's per-client personalisation rule)."""
    return any(p.search(name) for p in _PERSONALISED_NAME_PATTERNS)


@register
class FedBN:
    """FedBN: FedAvg + skip personalised (norm-layer) params on server.

    For our 3 backbones with no norm layers, this reduces to FedAvg
    bit-exactly. Documented in artifacts/audit/fedbn_reduces_to_fedavg.md.
    """

    name = "fedbn"

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
        del round_idx  # FedBN doesn't vary client behaviour by round
        local_model.to(device)
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
            grad_correction=None,
        )
        log.debug("fedbn client %s: steps=%d batch=%d loss=%.4f",
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
        """Per-key aggregation: standard FedAvg for non-personalised params,
        retain previous global_state value for personalised params."""
        # Standard FedAvg over all params first
        avg = weighted_average_state_dicts(
            [u.state_dict for u in updates],
            [u.num_examples for u in updates],
        )
        # Then for any personalised key, fall back to previous global_state.
        # (Our 3 archs have no such keys; this branch is a no-op for them.)
        for k in list(avg.keys()):
            if _is_personalised_param(k) and k in global_state:
                avg[k] = global_state[k]
        return avg
