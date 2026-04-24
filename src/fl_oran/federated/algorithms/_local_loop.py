"""Shared client-side training loop.

Rule-of-three trigger: FedAvg, FedProx, SCAFFOLD, FedDyn, (FedAdam inherits)
and MOON all share the same per-step structure (sample batch -> forward
-> optional loss modification -> backward -> optional grad correction ->
clip -> Adam step). Two hook points:

- ``loss_modifier(model, cat_batch, cont_batch, base_loss) -> new_loss``
  runs inside the autocast context before ``backward()``. MOON uses this
  for the model-contrastive term.
- ``grad_correction(model)`` runs after ``backward()`` and before
  ``clip_grad_norm_``. FedProx / SCAFFOLD / FedDyn use this to inject
  proximal / variance-reduction / dynamic-reg corrections into ``p.grad``.

FedAvg passes both as ``None`` and gets the vanilla loop.
"""
from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from ..client import _make_optimizer


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
    loss_modifier: Callable[
        [nn.Module, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
    ] | None = None,
    grad_correction: Callable[[nn.Module], None] | None = None,
) -> tuple[dict[str, torch.Tensor], float]:
    """Run ``max_steps`` Adam steps on ``local_model`` against ``client_tensors``.

    ``loss_modifier(model, cat_batch, cont_batch, base_loss)`` is invoked
    inside the autocast context; return the modified loss. Pass ``None``
    for algorithms that only need cross-entropy. Used by MOON for the
    contrastive term.

    ``grad_correction(model)`` is invoked after ``loss.backward()`` and
    before ``clip_grad_norm_`` on every step. Pass ``None`` for vanilla
    FedAvg. Used by FedProx (prox term), SCAFFOLD (c - c_i), and FedDyn
    (alpha*(w-w_g) - h_i).

    Preconditions: caller has already moved ``local_model`` to ``device``
    (FedProx/SCAFFOLD/FedDyn/MOON do this to snapshot parameters before
    training starts; FedAvg also moves so the move happens exactly once
    per round).

    Returns ``(cpu_state_dict, avg_loss)``. Model is left on ``device`` in
    train mode at the end.
    """
    cat_c, cont_c, y_c = client_tensors
    cat_g = cat_c.to(device, non_blocking=True)
    cont_g = cont_c.to(device, non_blocking=True)
    y_g = y_c.to(device, non_blocking=True)

    local_model.train()
    # Fused Adam on CUDA (bit-equivalent to non-fused; 1.5-2x faster on
    # small models). Falls back to plain Adam on CPU / older PyTorch.
    optimizer = _make_optimizer(local_model.parameters(), current_lr, device)
    n_local = cat_g.shape[0]
    total_loss = 0.0

    amp_ctx = torch.autocast(
        device_type=device.type,
        dtype=amp_dtype or torch.bfloat16,
        enabled=amp_enabled,
    )
    for _ in range(max_steps):
        idx = torch.randint(0, n_local, (batch_size,), device=device)
        cb = cat_g[idx]
        ob = cont_g[idx]
        yb = y_g[idx]
        optimizer.zero_grad(set_to_none=True)
        with amp_ctx:
            logits = local_model(cb, ob)
            loss = loss_fn(logits, yb)
            if loss_modifier is not None:
                loss = loss_modifier(local_model, cb, ob, loss)
        loss.backward()
        if grad_correction is not None:
            grad_correction(local_model)
        torch.nn.utils.clip_grad_norm_(local_model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()

    # ``.detach().cpu()`` on a CPU tensor does NOT copy — it returns a view on
    # the same storage. Callers that subsequently mutate ``local_model`` (e.g.
    # SCAFFOLD reloads w_g for its Option-II gradient) would then see their
    # "saved" state change under them. Explicit clone breaks the aliasing.
    state = {k: v.detach().cpu().clone() for k, v in local_model.state_dict().items()}
    avg_loss = total_loss / max(max_steps, 1)
    return state, avg_loss
