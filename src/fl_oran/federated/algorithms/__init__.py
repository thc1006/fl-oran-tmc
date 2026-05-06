"""FL algorithm registry (ADR-001 §3.2, M1 Day 3).

The registry lets the v5 sweep orchestrator instantiate algorithms by name
without knowing their internals:

    algo_cls = get_algorithm("fedavg")
    algo = algo_cls(max_steps=50, batch_size=64, ...)
    # per round, per selected client:
    update = algo.client_update(
        client_id=cid, local_model=fresh_model_with_global_state,
        client_tensors=(cat, cont, y),
        loss_fn=..., current_lr=..., device=..., round_idx=r,
    )
    new_global = algo.server_aggregate(
        global_state=current_global_state, updates=updates,
    )

Design notes (ultrathink 2026-04-25):
- ``client_update`` does NOT receive ``global_state`` — any algorithm that
  needs it (FedProx, MOON, FedDyn) snapshots ``local_model.state_dict()`` at
  the start of its own method. The orchestrator has already loaded the global
  state into ``local_model`` before calling, so the snapshot is GPU-resident
  with no extra host-device transfer.
- ``server_aggregate`` DOES receive ``global_state`` — SCAFFOLD and FedAdam
  (M2) need it to compute deltas and server-side optimizer steps. FedAvg
  ignores it via ``del global_state`` (self-documenting).
- ``ClientUpdate`` is reused from ``federated.client`` (not redefined here).
- Any per-client or per-server persistent state (SCAFFOLD control variates,
  FedAdam moment estimators) lives on the algorithm instance, not in the
  orchestrator.
"""
from __future__ import annotations

from typing import Callable, ClassVar, Protocol

import torch
from torch import nn

from ..client import ClientUpdate

__all__ = ["FLAlgorithm", "REGISTRY", "register", "get_algorithm", "ClientUpdate"]


class FLAlgorithm(Protocol):
    """Structural contract for a federated learning algorithm."""

    name: ClassVar[str]

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
        """Train ``local_model`` in place; return the resulting state_dict and weight.

        Implementations MUST treat ``client_tensors`` as read-only: the same
        tuple is reused across algorithms within a sweep, so in-place edits
        would corrupt subsequent runs. Index / slice freely, but do not mutate.
        """
        ...

    def server_aggregate(
        self,
        *,
        global_state: dict[str, torch.Tensor],
        updates: list[ClientUpdate],
    ) -> dict[str, torch.Tensor]:
        """Compute the new global state_dict from the previous one and client updates."""
        ...


REGISTRY: dict[str, type] = {}


def register(cls: type) -> type:
    """Class decorator: register ``cls`` in ``REGISTRY`` keyed by ``cls.name``.

    Raises ValueError on duplicate to surface accidental collisions early.
    """
    if not hasattr(cls, "name"):
        raise TypeError(f"{cls.__name__} is missing required `name` class attribute")
    key = cls.name
    if key in REGISTRY:
        raise ValueError(f"algorithm '{key}' already registered as {REGISTRY[key]!r}")
    REGISTRY[key] = cls
    return cls


def get_algorithm(name: str) -> type:
    """Look up an algorithm class by name; raise KeyError listing known names on miss."""
    try:
        return REGISTRY[name]
    except KeyError as e:
        known = sorted(REGISTRY.keys())
        raise KeyError(f"unknown algorithm '{name}'. Known: {known}") from e


# Trigger concrete-algorithm registration. Keep this at the bottom so
# REGISTRY/register are already defined when submodules import them.
from . import fedavg  # noqa: E402,F401
from . import fedprox  # noqa: E402,F401
from . import fedadam  # noqa: E402,F401
from . import scaffold  # noqa: E402,F401
from . import feddyn  # noqa: E402,F401
from . import moon  # noqa: E402,F401
from . import fedbn  # noqa: E402,F401  (P1.3 / R3.2)
from . import fedswa  # noqa: E402,F401  (R3.4)
