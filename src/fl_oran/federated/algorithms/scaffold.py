"""SCAFFOLD (Karimireddy et al. 2020, ICML) — variance-reduced FedAvg via
client-specific control variates.

Per-client ``c_i`` and server ``c`` persist across rounds. During local SGD,
every step's gradient is corrected by ``(c - c_i)``. We use the paper's
**Option-II** update for ``c_i`` — set ``c_i`` to the gradient of L_i at
the server model (on a mini-batch)::

    c_i_plus = grad_{L_i}(w_g)

Option-II is **optimizer-agnostic**: the magnitude of ``c_i`` matches the
data-gradient magnitude regardless of whether the client optimiser is SGD
or Adam. The original paper's Option-I derives ``c_i_plus`` from the
weight drift ``(w_g - w_l)/(K*eta)``, which implicitly assumes SGD
dynamics (where drift scales linearly with ``grad * eta``). Under Adam,
per-step drift is ``~eta`` regardless of gradient magnitude (Adam
normalises), so ``(w_g - w_l)/(K*eta) ~ O(1)`` — ~100x the true gradient
magnitude. Option-II side-steps this entirely.

Determinism note: the mini-batch for the Option-II gradient uses a
deterministic ``arange`` index and runs the model in eval mode (dropout
off) so it does not perturb the global torch RNG state consumed by the
training loop. This preserves the ``SCAFFOLD first-round ==
FedAvg trajectory`` bit-equivalence when ``c = c_i = 0``.
"""
from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from ..aggregation import weighted_average_state_dicts
from ..client import ClientUpdate
from ...logging_utils import get_logger
from . import register
from ._local_loop import run_local_sgd

log = get_logger(__name__)


@register
class SCAFFOLD:
    """SCAFFOLD (Option I).

    Stateful: ``self.c`` holds the global control variate (lazy-init on first
    use), ``self.c_i[client_id]`` holds per-client control variates.
    """

    name = "scaffold"

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
        self.c: dict[str, torch.Tensor] = {}
        self.c_i: dict[int, dict[str, torch.Tensor]] = {}

    def _ensure_c(self, model: nn.Module, device: torch.device) -> None:
        if not self.c:
            self.c = {
                name: torch.zeros_like(p, device=device)
                for name, p in model.named_parameters()
            }

    def _ensure_c_i(self, client_id: int, model: nn.Module,
                     device: torch.device) -> None:
        if client_id not in self.c_i:
            self.c_i[client_id] = {
                name: torch.zeros_like(p, device=device)
                for name, p in model.named_parameters()
            }

    def _compute_gradient_at(
        self,
        model: nn.Module,
        cat: torch.Tensor,
        cont: torch.Tensor,
        y: torch.Tensor,
        loss_fn: Callable,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Compute per-parameter gradient of loss_fn at model's current weights.

        Uses a deterministic mini-batch (``arange(0, batch_size)``) so the
        batch choice does not consume RNG. We MUST run in train mode because
        cuDNN's LSTM backward is only implemented for training mode — eval
        mode would raise ``RuntimeError: cudnn RNN backward can only be
        called in training mode``. Dropout would then consume torch RNG, so
        we snapshot + restore both the CPU and CUDA RNG streams around the
        call to keep the outer training-loop's RNG chain unchanged.
        """
        bs = min(self.batch_size, cat.shape[0])
        det_idx = torch.arange(bs, device=device)
        was_training = model.training
        model.train()
        cpu_rng = torch.get_rng_state()
        cuda_rng = (torch.cuda.get_rng_state(device)
                    if device.type == "cuda" else None)
        model.zero_grad(set_to_none=True)
        amp_ctx = torch.autocast(
            device_type=device.type,
            dtype=self.amp_dtype or torch.bfloat16,
            enabled=self.amp_enabled,
        )
        with amp_ctx:
            logits = model(cat[det_idx], cont[det_idx])
            loss = loss_fn(logits, y[det_idx])
        loss.backward()
        grads = {
            name: p.grad.detach().clone()
            for name, p in model.named_parameters()
            if p.grad is not None
        }
        model.zero_grad(set_to_none=True)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)
        if not was_training:
            model.eval()
        return grads

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
        # Snapshot w_g as a state_dict *before* training so we can reload it
        # later for the Option-II gradient computation.
        global_snapshot_state = {
            k: v.detach().clone()
            for k, v in local_model.state_dict().items()
        }
        self._ensure_c(local_model, device)
        self._ensure_c_i(client_id, local_model, device)
        c = {k: v.to(device) for k, v in self.c.items()}
        c_i = {k: v.to(device) for k, v in self.c_i[client_id].items()}

        def scaffold_correction(model: nn.Module) -> None:
            # grad += (c - c_i). When c = c_i = 0 (first round), no-op
            # so SCAFFOLD first-round trajectory == FedAvg bit-wise.
            for name, p in model.named_parameters():
                if p.grad is not None:
                    p.grad.add_(c[name] - c_i[name])

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
            grad_correction=scaffold_correction,
        )

        # Option-II: c_i_plus = grad_{L_i}(w_g) on a deterministic mini-batch.
        # Reload w_g (training mutated local_model) — state_dict move is
        # cheap and avoids keeping a second full model resident.
        cat_c, cont_c, y_c = client_tensors
        cat_g = cat_c.to(device, non_blocking=True)
        cont_g = cont_c.to(device, non_blocking=True)
        y_g = y_c.to(device, non_blocking=True)
        local_model.load_state_dict(global_snapshot_state, strict=True)
        grad_at_wg = self._compute_gradient_at(
            local_model, cat_g, cont_g, y_g, loss_fn, device,
        )
        # Restore post-training weights so the returned state matches ``state``.
        local_model.load_state_dict(
            {k: v.to(device) for k, v in state.items()}, strict=True,
        )

        new_c_i: dict[str, torch.Tensor] = {}
        delta_c_i: dict[str, torch.Tensor] = {}
        for name in grad_at_wg:
            c_i_plus = grad_at_wg[name]
            new_c_i[name] = c_i_plus.detach().cpu()
            delta_c_i[name] = (c_i_plus - c_i[name]).detach().cpu()
        self.c_i[client_id] = new_c_i

        log.debug("scaffold client %s: steps=%d batch=%d loss=%.4f",
                  client_id, self.max_steps, self.batch_size, avg_loss)
        return ClientUpdate(
            client_id=client_id,
            state_dict=state,
            num_examples=self.max_steps * self.batch_size,
            train_loss=avg_loss,
            aux={"delta_c_i": delta_c_i},
        )

    def server_aggregate(
        self,
        *,
        global_state: dict[str, torch.Tensor],
        updates: list[ClientUpdate],
    ) -> dict[str, torch.Tensor]:
        """Invariant: at least one ``client_update`` must run before this call,
        so that ``self.c`` is lazily initialized (shape inferred from the
        model). Calling ``server_aggregate`` with only ``delta_c_i`` payloads
        but no prior client_update would leave the global control variate
        uninitialized and the deltas would be silently dropped.
        """
        del global_state
        # Weights: standard FedAvg over client states.
        new_w = weighted_average_state_dicts(
            [u.state_dict for u in updates],
            [u.num_examples for u in updates],
        )
        deltas_present = [u.aux.get("delta_c_i") for u in updates if u.aux]
        deltas_present = [d for d in deltas_present if d]
        # Detect accidental misuse: deltas are present but self.c wasn't
        # initialized by a prior client_update. Warn loudly so the silent
        # drop is observable.
        if not self.c and deltas_present:
            log.warning(
                "scaffold.server_aggregate: self.c is empty but %d updates "
                "carry delta_c_i — dropping them (call client_update first)",
                len(deltas_present),
            )
            return new_w
        if not self.c or not deltas_present:
            return new_w
        # Control variate: c <- c + mean(delta_c_i) over the participating
        # clients. Paper form is c + (|S|/N) * (1/|S|) * sum(delta_c_i) =
        # (1/N) * sum(delta_c_i). We approximate N by |S| (exact when all
        # clients participate); partial participation accounting is a known
        # simplification that the M3 orchestrator will revisit.
        for name in self.c:
            stack = torch.stack([d[name] for d in deltas_present if name in d])
            self.c[name] = self.c[name].cpu() + stack.mean(dim=0)
        log.debug("scaffold aggregate: updated c from %d delta_c_i",
                  len(deltas_present))
        return new_w
