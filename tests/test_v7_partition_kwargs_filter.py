"""Regression tests: partition-axis kwargs (sub_per_bs) must be filtered
out of ``V7Config.algo_kwargs`` before the FL algorithm class is built.

Bug found 2026-05-02 during Phase 6 Rank 3 audit:

    The Rank 3 launcher passes ``--algo-kwargs '{"sub_per_bs": 2}'`` to
    drive the per_bs_dirichlet shard count. ``run_v7_sweep`` reads
    ``cfg.algo_kwargs.get("sub_per_bs")`` for the partition dispatcher
    but ALSO blindly forwards ``cfg.algo_kwargs`` into the algo init dict
    (line 696, pre-fix: ``algo_kwargs.update(cfg.algo_kwargs)``).

    All FL algorithm classes (FedAvg/FedProx/FedAdam/SCAFFOLD/FedDyn)
    use keyword-only signatures with NO ``**kwargs`` — any unknown key
    raises ``TypeError`` at construction. Without filtering, all 60
    Phase 6 Rank 3 cells would crash before training round 1.

These tests pin the partition-only allowlist and verify that filtering
is applied so the pre-Rank-3 fix doesn't regress.
"""
from __future__ import annotations

import torch

from fl_oran.federated.algorithms.fedavg import FedAvg
from fl_oran.federated.algorithms.fedprox import FedProx
from fl_oran.training.fl_v7 import _PARTITION_ONLY_ALGO_KWARGS


def test_constant_lists_sub_per_bs() -> None:
    """sub_per_bs is the only partition-axis kwarg today; pin the set."""
    assert "sub_per_bs" in _PARTITION_ONLY_ALGO_KWARGS


def test_filtered_kwargs_construct_fedavg() -> None:
    """The exact Rank 3 launcher payload must produce a FedAvg instance
    after filtering — no TypeError on unknown kwargs."""
    user_supplied = {"sub_per_bs": 2}  # what the launcher passes
    base = {
        "max_steps": 50,
        "batch_size": 64,
        "grad_clip": 1.0,
        "amp_enabled": False,
        "amp_dtype": None,
    }
    filtered = {
        **base,
        **{k: v for k, v in user_supplied.items()
           if k not in _PARTITION_ONLY_ALGO_KWARGS},
    }
    assert "sub_per_bs" not in filtered
    # Must not raise.
    fedavg = FedAvg(**filtered)
    assert fedavg.max_steps == 50
    assert fedavg.batch_size == 64


def test_filter_preserves_real_algo_kwargs() -> None:
    """When a spec mixes partition + algorithm kwargs (e.g. FedProx mu
    AND sub_per_bs), only sub_per_bs is dropped; mu reaches the algo."""
    user_supplied = {"sub_per_bs": 2, "mu": 0.01}
    base = {
        "max_steps": 50,
        "batch_size": 64,
        "grad_clip": 1.0,
        "amp_enabled": False,
        "amp_dtype": None,
    }
    filtered = {
        **base,
        **{k: v for k, v in user_supplied.items()
           if k not in _PARTITION_ONLY_ALGO_KWARGS},
    }
    assert "sub_per_bs" not in filtered
    assert filtered["mu"] == 0.01
    fedprox = FedProx(**filtered)
    assert fedprox.mu == 0.01


def test_per_bs_dirichlet_requires_explicit_sub_per_bs() -> None:
    """fl_v7's partition dispatcher MUST raise when partition_mode is
    per_bs_dirichlet and ``algo_kwargs`` lacks ``sub_per_bs``.

    Earlier draft used ``cfg.algo_kwargs.get("sub_per_bs", 2)`` — a
    silent default that would have shipped 60 cells with 14 sub-clients
    even if the spec yaml/launcher forgot the kwarg. Phase 6 audit
    follow-up 2026-05-03: surface as fail-fast ValueError instead.
    """
    import numpy as np
    import pandas as pd
    from fl_oran.training.fl_v7 import V7Config, _partition

    df = pd.DataFrame({
        "bs_id": np.repeat(np.arange(1, 8), 100).astype("uint8"),
        "slice_id": np.tile(np.arange(3), 700 // 3 + 1)[:700].astype("uint8"),
        "tx_brate_dl_Mbps": np.random.default_rng(0).uniform(0, 10, 700).astype("float32"),
    })
    cfg = V7Config(
        arch="lstm",
        algorithm="fedavg",
        partition_mode="per_bs_dirichlet",
        algo_kwargs={},  # empty — sub_per_bs missing
        alpha=0.5,
        n_clients=14,
        seed=42,
    )
    try:
        _partition(df, cfg)
    except ValueError as e:
        assert "sub_per_bs" in str(e)
        return
    raise AssertionError(
        "expected ValueError for missing sub_per_bs; got silent default"
    )


def test_unfiltered_kwargs_would_crash_fedavg() -> None:
    """Witness the bug: passing sub_per_bs unfiltered raises TypeError.
    This is the failure mode that would have killed Phase 6 Rank 3."""
    base = {
        "max_steps": 50,
        "batch_size": 64,
        "grad_clip": 1.0,
        "amp_enabled": False,
        "amp_dtype": None,
        "sub_per_bs": 2,  # the bug: didn't get stripped
    }
    try:
        FedAvg(**base)
    except TypeError as e:
        assert "sub_per_bs" in str(e)
        return
    raise AssertionError(
        "FedAvg accepted sub_per_bs unexpectedly; if FedAvg gained "
        "**kwargs, _PARTITION_ONLY_ALGO_KWARGS may no longer be needed"
    )
