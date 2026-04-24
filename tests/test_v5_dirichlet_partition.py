"""TDD red-phase tests for Dirichlet non-IID partition (ADR-001 §3.1).

These tests fail before the implementation in
``src/fl_oran/data_v2/partition.py`` is extended with ``mode="dirichlet"``.

Test assertions per ADR-001 §3.1:
1. alpha concentration — large alpha -> near-uniform; small alpha -> concentrated
2. total rows preserved — sum of shard sizes equals input row count
3. seed reproducible — same seed → bit-equivalent partition
4. n_clients respected — returns up to n_clients shards
5. missing alpha when mode="dirichlet" raises ValueError
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fl_oran.data_v2.partition import partition_clients


# --------------------------------------------------------------------------
# Synthetic data helper (local to this test module; not a reusable fixture
# because D-3 forbids duplicating shared helpers that don't exist yet).
# --------------------------------------------------------------------------

def _make_df(n_rows: int = 3000, n_slices: int = 3, seed: int = 0) -> pd.DataFrame:
    """Balanced synthetic data with 3 slices — mimics ColO-RAN schema.

    n_rows is rounded DOWN to the nearest multiple of n_slices to keep arrays
    aligned (avoids pandas length-mismatch in DataFrame construction).
    """
    rows_per_slice = n_rows // n_slices
    actual_rows = rows_per_slice * n_slices
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "bs_id": rng.integers(1, 8, actual_rows).astype("uint8"),
        "slice_id": np.repeat(np.arange(n_slices, dtype="uint8"), rows_per_slice),
        "tx_brate_dl_Mbps": rng.uniform(0, 10, actual_rows).astype("float32"),
    })


# --------------------------------------------------------------------------
# Tests (written before implementation — red phase)
# --------------------------------------------------------------------------

def test_dirichlet_partition_respects_alpha_concentration():
    """Large alpha -> near-uniform shard sizes. Small alpha -> concentrated."""
    df = _make_df(n_rows=3000)

    # Large alpha = 100 -> approximately equal shard sizes
    uniform = partition_clients(df, mode="dirichlet", alpha=100.0, n_clients=5, seed=42)
    target = len(df) / 5
    for shard in uniform.values():
        assert 0.6 * target < len(shard) < 1.4 * target, \
            f"at alpha=100, shard size {len(shard)} drifts too far from uniform target {target:.0f}"

    # Small alpha = 0.01 -> highly concentrated (at least one client gets << average)
    concentrated = partition_clients(df, mode="dirichlet", alpha=0.01, n_clients=5, seed=42)
    avg = len(df) / 5
    sparse_clients = sum(1 for s in concentrated.values() if len(s) < 0.1 * avg)
    assert sparse_clients >= 1, \
        f"at alpha=0.01, expected at least 1 sparse client, got {sparse_clients}"


def test_dirichlet_partition_preserves_total_rows():
    """Sum of shard sizes must equal input row count (no leakage, no duplication)."""
    df = _make_df(n_rows=3000)
    shards = partition_clients(df, mode="dirichlet", alpha=0.5, n_clients=5, seed=42)
    total = sum(len(s) for s in shards.values())
    assert total == len(df), f"total {total} != input {len(df)}"


def test_dirichlet_partition_seed_reproducible():
    """Same seed + data → bit-equivalent partition (D-11 determinism requirement)."""
    df = _make_df(n_rows=3000, seed=1)
    s1 = partition_clients(df, mode="dirichlet", alpha=0.5, n_clients=5, seed=42)
    s2 = partition_clients(df, mode="dirichlet", alpha=0.5, n_clients=5, seed=42)
    assert set(s1.keys()) == set(s2.keys())
    for cid in s1:
        # Reset indices because order after reset_index(drop=True) is what
        # downstream sequence builder sees.
        a = s1[cid].reset_index(drop=True)
        b = s2[cid].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)


def test_dirichlet_partition_n_clients():
    """n_clients=7 on moderate alpha → returns ≤7 shards, most non-empty.

    The lower bound (≥4) holds comfortably on seed=42 (observed: 6-7 non-empty
    shards). If a future seed sweep is introduced and this test flakes, tighten
    by either lowering the bound further or computing an alpha-specific
    expectation analytically.
    """
    df = _make_df(n_rows=5000)
    shards = partition_clients(df, mode="dirichlet", alpha=0.5, n_clients=7, seed=42)
    assert len(shards) <= 7, f"produced {len(shards)} shards, expected ≤7"
    # Moderate alpha: majority of clients should still receive a non-empty share.
    assert len(shards) >= 4, f"expected ≥4 non-empty shards at alpha=0.5, got {len(shards)}"


def test_dirichlet_partition_missing_alpha_raises():
    """mode='dirichlet' without alpha is a contract violation."""
    df = _make_df(n_rows=300)
    with pytest.raises(ValueError, match="alpha"):
        partition_clients(df, mode="dirichlet", n_clients=5, seed=42)


def test_dirichlet_partition_missing_n_clients_raises():
    """mode='dirichlet' without n_clients is a contract violation."""
    df = _make_df(n_rows=300)
    with pytest.raises(ValueError, match="n_clients"):
        partition_clients(df, mode="dirichlet", alpha=0.5, seed=42)


def test_partition_existing_modes_still_work():
    """Backward-compat regression: IID and noniid_slice still function after the
    Dirichlet branch is added."""
    df = _make_df(n_rows=300)
    iid_shards = partition_clients(df, mode="iid")
    assert len(iid_shards) == df["bs_id"].nunique()

    noniid_shards = partition_clients(
        df, mode="noniid_slice",
        client_slice_map={1: [0], 2: [1]},
    )
    assert all(
        set(s["slice_id"].unique()) <= {0, 1}
        for s in noniid_shards.values()
    )
