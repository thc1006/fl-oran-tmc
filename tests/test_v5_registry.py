"""TDD red-phase tests for FL algorithm registry (ADR-001 §3.2).

Contract (per ultrathink design, 2026-04-25):
- ``FLAlgorithm`` Protocol: structural contract with ``name``, ``client_update``,
  ``server_aggregate``.
- ``REGISTRY``: str → algorithm class. ``@register`` class decorator adds to it.
- ``get_algorithm(name)`` → class; raises KeyError with known names on miss.
- ``FedAvg`` registered under "fedavg".
- Double-registering the same name raises ValueError (avoid silent collisions).
- ``server_aggregate`` signature includes ``global_state`` (required for FedAdam /
  SCAFFOLD in M2; FedAvg ignores). ``client_update`` does NOT include
  ``global_state`` (algorithms snapshot from ``local_model`` when needed).
"""
from __future__ import annotations

import inspect

import pytest


def test_registry_has_fedavg():
    from fl_oran.federated.algorithms import REGISTRY
    assert "fedavg" in REGISTRY, f"REGISTRY missing 'fedavg'; has {sorted(REGISTRY)}"
    assert REGISTRY["fedavg"].name == "fedavg"


def test_fedavg_satisfies_protocol_shape():
    """FedAvg class exposes the Protocol's methods with the expected kwargs."""
    from fl_oran.federated.algorithms import REGISTRY
    cls = REGISTRY["fedavg"]
    assert hasattr(cls, "name")
    assert callable(getattr(cls, "client_update", None))
    assert callable(getattr(cls, "server_aggregate", None))

    cu_params = set(inspect.signature(cls.client_update).parameters)
    cu_required = {"client_id", "local_model", "client_tensors",
                   "loss_fn", "current_lr", "device", "round_idx"}
    assert cu_required <= cu_params, (
        f"FedAvg.client_update missing {cu_required - cu_params}"
    )
    # client_update must NOT take global_state — algorithms snapshot locally.
    assert "global_state" not in cu_params, (
        "client_update should not take global_state; algorithms snapshot from "
        "local_model instead (see ADR-001 design note)"
    )

    sa_params = set(inspect.signature(cls.server_aggregate).parameters)
    sa_required = {"global_state", "updates"}
    assert sa_required <= sa_params, (
        f"FedAvg.server_aggregate missing {sa_required - sa_params}"
    )


def test_get_algorithm_unknown_raises():
    from fl_oran.federated.algorithms import get_algorithm
    with pytest.raises(KeyError, match="unknown algorithm"):
        get_algorithm("nonexistent_xyz")


def test_duplicate_registration_raises():
    from fl_oran.federated.algorithms import REGISTRY, register
    cls = REGISTRY["fedavg"]
    with pytest.raises(ValueError, match="already registered"):
        register(cls)
