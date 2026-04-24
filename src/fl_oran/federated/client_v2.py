"""Step-capped client trainer: runs exactly ``max_steps`` gradient updates.

Fixes the main instability source in trainer_v2: for clients with 2M sequences,
2 local epochs meant 62k local gradient steps between aggregations — far too
many, hence the wild val-loss oscillation. This trainer decouples "amount of
local training" from "amount of local data".
"""
from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from ..logging_utils import get_logger
from .client import ClientUpdate, _make_optimizer

log = get_logger(__name__)


def train_one_client_capped(
    client_id: int,
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    loss_fn: Callable,
    device: torch.device,
    *,
    lr: float,
    max_steps: int,
    batch_size: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    grad_clip_norm: float | None = 1.0,
    seed: int | None = None,
) -> ClientUpdate:
    """Run exactly ``max_steps`` SGD steps. Each step samples ``batch_size``
    rows from (X, y) — with replacement if needed.

    Supports both GPU-resident tensors and CPU tensors (will move per-batch).
    """
    model.to(device).train()
    optimizer = _make_optimizer(model.parameters(), lr, device)
    n = X.shape[0]
    gen = torch.Generator(device=device if device.type == "cuda" and X.is_cuda else "cpu")
    if seed is not None:
        gen.manual_seed(seed)

    autocast_ctx = torch.autocast(
        device_type=device.type, dtype=amp_dtype or torch.bfloat16, enabled=amp_enabled
    )
    total_loss = 0.0
    for step in range(max_steps):
        # Sample batch_size indices uniformly (with replacement for simplicity).
        idx = torch.randint(0, n, (batch_size,), generator=gen, device=gen.device)
        xb = X[idx].to(device, non_blocking=True) if not X.is_cuda else X[idx]
        yb = y[idx].to(device, non_blocking=True) if not y.is_cuda else y[idx]

        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx:
            # Unpack tuple if the model takes (cat, cont) inputs.
            if isinstance(xb, (tuple, list)):
                pred = model(*xb)
            else:
                pred = model(xb)
            loss = loss_fn(pred, yb)
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / max(max_steps, 1)
    state = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
    log.debug("client %s capped: steps=%d batch=%d loss=%.4f",
              client_id, max_steps, batch_size, avg_loss)
    return ClientUpdate(
        client_id=client_id,
        state_dict=state,
        num_examples=max_steps * batch_size,
        train_loss=avg_loss,
    )
