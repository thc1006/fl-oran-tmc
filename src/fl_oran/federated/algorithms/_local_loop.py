"""Shared client-side training loop.

Rule-of-three trigger: FedAvg, FedProx, SCAFFOLD, FedDyn, (FedAdam inherits)
all share the same per-step structure (sample batch → forward → backward →
optional grad correction → clip → Adam step). Extracted to avoid drift
across copies as M2 algorithms land.
"""
from __future__ import annotations

from typing import Callable

import torch
from torch import nn


def run_local_sgd(
    *,
    local_model: nn.Module,
    client_tensors: tuple[torch.Tensor, ...],
    loss_fn: Callable,
    current_lr: float,
    max_steps: int,
    batch_size: int,
    grad_clip: float,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    device: torch.device,
    grad_correction: Callable[[nn.Module], None] | None = None,
) -> tuple[dict[str, torch.Tensor], float]:
    """Run ``max_steps`` Adam steps on ``local_model`` against ``client_tensors``.

    ``grad_correction(model)`` is invoked after ``loss.backward()`` and before
    ``clip_grad_norm_`` on every step. Pass ``None`` for vanilla FedAvg. Used
    by FedProx (prox term), SCAFFOLD (c - c_i) and FedDyn (α·(w-w_g) - h_i).

    Returns ``(cpu_state_dict, avg_loss)``. Model is left on ``device`` in
    train mode at the end.
    """
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
        dtype=amp_dtype or torch.bfloat16,
        enabled=amp_enabled,
    )
    for _ in range(max_steps):
        idx = torch.randint(0, n_local, (batch_size,), device=device)
        optimizer.zero_grad(set_to_none=True)
        with amp_ctx:
            logits = local_model(cat_g[idx], cont_g[idx])
            loss = loss_fn(logits, y_g[idx])
        loss.backward()
        if grad_correction is not None:
            grad_correction(local_model)
        torch.nn.utils.clip_grad_norm_(local_model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()

    state = {k: v.detach().cpu() for k, v in local_model.state_dict().items()}
    avg_loss = total_loss / max(max_steps, 1)
    return state, avg_loss
