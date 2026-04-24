"""MOON (Li et al. 2021, CVPR) — model-contrastive federated learning.

Each client training step mixes cross-entropy with a contrastive term on
learned representations::

    z      = encode(w_local, x)
    z_g    = encode(w_global, x)       # positive, frozen
    z_prev = encode(w_prev_round, x)   # negative, frozen
    L_contrastive = -log( exp(sim(z, z_g) / tau)
                          / (exp(sim(z, z_g) / tau) + exp(sim(z, z_prev) / tau)) )
    L_total = L_CE + mu * L_contrastive

Implementation notes:

- Representation extraction is delegated to a caller-supplied
  ``encode_fn(model, cat, cont) -> Tensor`` to keep MOON model-agnostic.
  The orchestrator is responsible for providing an encode_fn that matches
  the target architecture (ForecasterV2 in v5).
- The contrastive term is injected via ``run_local_sgd``'s
  ``loss_modifier`` hook so MOON shares the single canonical training
  loop with the other algorithms — no per-algorithm training-loop copy.
- Cold-start (first round for a client): ``prev_model`` is aliased to the
  frozen global snapshot, which makes sim_pos == sim_neg and collapses
  L_contrastive to log(2) with zero gradient. This matches the paper's
  treatment of the round-1 case; behaviour becomes nontrivial from
  round 2 onwards.

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
from ._local_loop import run_local_sgd

log = get_logger(__name__)


@register
class MOON:
    """MOON — FedAvg aggregation with a local contrastive loss.

    ``encode_fn(model, cat, cont) -> Tensor`` should return a per-sample
    representation of shape ``(B, repr_dim)``. It is called on the current
    local model (grad-tracked) and on frozen snapshots of the global and
    previous-local models.
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
        # as the contrastive negative. Lazily populated after round 1.
        self.prev_models: dict[int, dict[str, torch.Tensor]] = {}

    def _contrastive_loss(
        self,
        z: torch.Tensor,
        z_g: torch.Tensor,
        z_prev: torch.Tensor,
    ) -> torch.Tensor:
        # Row-wise cosine similarity, scaled by temperature.
        sim_pos = F.cosine_similarity(z, z_g, dim=-1) / self.tau
        sim_neg = F.cosine_similarity(z, z_prev, dim=-1) / self.tau
        # -log( exp(pos) / (exp(pos) + exp(neg)) ) = softplus(neg - pos).
        # Single softplus call is numerically stable and equivalent to the
        # paper's log-softmax form.
        return F.softplus(sim_neg - sim_pos).mean()

    @staticmethod
    def _freeze_(model: nn.Module) -> nn.Module:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model

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
        local_model.to(device)

        mu = self.mu
        loss_modifier = None
        if mu != 0.0:
            # Frozen global model (positive). One deepcopy at the start of
            # training; reused for every step's z_g computation.
            global_model = self._freeze_(copy.deepcopy(local_model))
            # Frozen previous-local model (negative). On the first round for
            # this client we alias to the global snapshot — the contrastive
            # term then collapses to log(2) with zero gradient (see module
            # docstring). From round 2 onwards prev_model carries the
            # post-training state from the client's previous round.
            if client_id in self.prev_models:
                prev_model = copy.deepcopy(local_model)
                prev_model.load_state_dict(self.prev_models[client_id])
                self._freeze_(prev_model)
            else:
                prev_model = global_model  # alias — both are frozen & identical

            encode_fn = self.encode_fn
            contrastive_loss = self._contrastive_loss

            def loss_modifier(
                model: nn.Module,
                cb: torch.Tensor,
                ob: torch.Tensor,
                base_loss: torch.Tensor,
            ) -> torch.Tensor:
                z = encode_fn(model, cb, ob)
                with torch.no_grad():
                    z_g = encode_fn(global_model, cb, ob)
                    z_prev = encode_fn(prev_model, cb, ob)
                return base_loss + mu * contrastive_loss(z, z_g, z_prev)

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
            loss_modifier=loss_modifier,
            grad_correction=None,
        )

        # Persist post-training local state as this client's "prev" for
        # next round. Keyed per-client so concurrent clients don't collide.
        self.prev_models[client_id] = {k: v.clone() for k, v in state.items()}

        log.debug("moon client %s: steps=%d batch=%d mu=%.4f tau=%.3f loss=%.4f",
                  client_id, self.max_steps, self.batch_size,
                  self.mu, self.tau, avg_loss)
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
