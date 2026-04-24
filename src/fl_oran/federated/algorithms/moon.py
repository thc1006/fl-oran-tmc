"""MOON (Li et al. 2021, CVPR) — model-contrastive federated learning.

Each client training step mixes cross-entropy with a contrastive term on
learned representations::

    z      = encode(w_local, x)
    z_g    = encode(w_global, x)       # positive — frozen
    z_prev = encode(w_prev_round, x)   # negative — frozen
    L_contrastive = -log( exp(sim(z, z_g) / tau)
                          / (exp(sim(z, z_g) / tau) + exp(sim(z, z_prev) / tau)) )
    L_total = L_CE + mu * L_contrastive

Representation extraction is delegated to a caller-supplied ``encode_fn``
to keep MOON model-agnostic. The orchestrator is responsible for providing
an ``encode_fn`` that matches the target model (ForecasterV2 in v5).

Server aggregation is standard FedAvg.
"""
from __future__ import annotations

import copy
from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F

from ..aggregation import weighted_average_state_dicts
from ..client import ClientUpdate
from ...logging_utils import get_logger
from . import register

log = get_logger(__name__)


@register
class MOON:
    """MOON — FedAvg aggregation with a local contrastive loss.

    ``encode_fn(model, cat, cont) -> Tensor`` should return a per-sample
    representation (shape ``(B, repr_dim)``). The same function is invoked
    on the current local model (grad-tracked) and on frozen snapshots of
    the global and previous-local models.
    """

    name = "moon"

    def __init__(
        self,
        *,
        max_steps: int,
        batch_size: int,
        mu: float,
        tau: float,
        encode_fn: Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor],
        grad_clip: float = 1.0,
        amp_enabled: bool = False,
        amp_dtype: torch.dtype | None = None,
    ) -> None:
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.mu = float(mu)
        self.tau = float(tau)
        self.encode_fn = encode_fn
        self.grad_clip = grad_clip
        self.amp_enabled = amp_enabled
        self.amp_dtype = amp_dtype
        # {client_id: state_dict on CPU} — previous-round local model, used
        # as the contrastive negative. Lazily set on first client_update.
        self.prev_models: dict[int, dict[str, torch.Tensor]] = {}

    def _contrastive_loss(
        self,
        z: torch.Tensor,
        z_g: torch.Tensor,
        z_prev: torch.Tensor,
    ) -> torch.Tensor:
        # Row-wise cosine similarity.
        sim_pos = F.cosine_similarity(z, z_g, dim=-1) / self.tau
        sim_neg = F.cosine_similarity(z, z_prev, dim=-1) / self.tau
        # -log( exp(pos) / (exp(pos) + exp(neg)) ) = log(1 + exp(neg - pos)) = softplus(neg - pos)
        return F.softplus(sim_neg - sim_pos).mean()

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
        y_g_tensor = y_c.to(device, non_blocking=True)

        local_model.to(device).train()

        mu = self.mu
        if mu != 0.0:
            # Frozen global model (positive) — clone at the start of training.
            global_model = copy.deepcopy(local_model).eval()
            for p in global_model.parameters():
                p.requires_grad_(False)
            # Frozen previous-local model (negative). First round: use global
            # as the negative; then the contrastive numerator and denominator
            # would collapse, but our softplus formulation handles it cleanly
            # (returns log(2)) and mu-weighted gradient is still well-defined.
            if client_id in self.prev_models:
                prev_model = copy.deepcopy(local_model)
                prev_model.load_state_dict(self.prev_models[client_id])
            else:
                prev_model = copy.deepcopy(local_model)
            prev_model.eval()
            for p in prev_model.parameters():
                p.requires_grad_(False)
        else:
            global_model = None
            prev_model = None

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
            cb = cat_g[idx]
            ob = cont_g[idx]
            yb = y_g_tensor[idx]
            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                logits = local_model(cb, ob)
                loss = loss_fn(logits, yb)
                if mu != 0.0:
                    z = self.encode_fn(local_model, cb, ob)
                    with torch.no_grad():
                        z_g = self.encode_fn(global_model, cb, ob)
                        z_prev = self.encode_fn(prev_model, cb, ob)
                    loss = loss + mu * self._contrastive_loss(z, z_g, z_prev)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(local_model.parameters(), self.grad_clip)
            optimizer.step()
            total_loss += loss.item()

        # Persist post-training local state as this client's "prev" for next round.
        state = {k: v.detach().cpu() for k, v in local_model.state_dict().items()}
        self.prev_models[client_id] = {k: v.clone() for k, v in state.items()}

        avg_loss = total_loss / max(self.max_steps, 1)
        log.debug("moon client %s: steps=%d batch=%d mu=%.4f tau=%.3f loss=%.4f",
                  client_id, self.max_steps, self.batch_size, self.mu, self.tau, avg_loss)
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
        del global_state  # MOON uses FedAvg aggregation
        return weighted_average_state_dicts(
            [u.state_dict for u in updates],
            [u.num_examples for u in updates],
        )
