"""TDD red-phase tests for ``partition_clients(mode="random_split")``.

This partition mode is the ablation control for the inverted-α / natural-by-BS
mechanism question in PAPER_DRAFT.md §7. It splits all training rows uniformly
at random across ``n_clients`` clients, ignoring every column (including
``bs_id`` and ``slice_id``). Each client therefore sees approximately the
global marginal distribution — this is the "true IID" partition.

Compared to existing modes:

  * ``mode="iid"`` groups by ``bs_id`` (one client per BS, all slices).
    The misleading name "iid" was inherited from earlier code; this mode
    does NOT produce statistical IID across clients in heterogeneous
    real-world datasets.

  * ``mode="dirichlet"`` redistributes each slice's rows by Dirichlet
    proportions, breaking ``bs_id`` correlation but preserving slice-level
    grouping.

  * ``mode="random_split"`` (NEW): shuffle all row indices and split into
    ``n_clients`` ~equal-size chunks. Breaks both bs_id and slice_id
    grouping. This is the partition the §7 mechanism ablation needs.

Test assertions:
  1. Returns a dict with ``n_clients`` shards (no empty client unless
     ``n_clients`` > total rows).
  2. Total row count preserved: ``sum(len(shard)) == len(df)``.
  3. Per-client sample size is balanced within ±1 row of ``len(df) / n``.
  4. Same ``seed`` produces bit-equivalent partition (reproducibility per
     ADR-001 D-15).
  5. Different ``seed`` produces different partitions (no silent hardcode).
  6. Missing ``n_clients`` when ``mode="random_split"`` raises ValueError.
  7. Each shard's ``bs_id`` distribution is within 5 percentage points of
     the global ``bs_id`` distribution (uniform random sampling preserves
     marginals to first order).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fl_oran.data_v2.partition import partition_clients


def _make_df(n_rows: int = 3000, n_slices: int = 3, n_bs: int = 7,
             seed: int = 0) -> pd.DataFrame:
    """Synthetic ColO-RAN-shaped data: bs_id ∈ [1..n_bs], slice_id ∈ [0..n_slices-1]."""
    rows_per_slice = n_rows // n_slices
    actual_rows = rows_per_slice * n_slices
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "bs_id": rng.integers(1, n_bs + 1, actual_rows).astype("uint8"),
        "slice_id": np.repeat(np.arange(n_slices, dtype="uint8"), rows_per_slice),
        "tx_brate_dl_Mbps": rng.uniform(0, 10, actual_rows).astype("float32"),
    })


# ---------- 1: shard count ----------

def test_random_split_returns_n_clients_shards() -> None:
    df = _make_df(n_rows=3000, n_slices=3)
    shards = partition_clients(df, mode="random_split", n_clients=7, seed=42)
    assert isinstance(shards, dict)
    assert len(shards) == 7
    for cid in range(7):
        assert cid in shards or (cid + 1) in shards, (
            f"client id missing; got keys {sorted(shards.keys())}"
        )


# ---------- 2: total rows preserved ----------

def test_random_split_preserves_total_row_count() -> None:
    df = _make_df(n_rows=3000)
    shards = partition_clients(df, mode="random_split", n_clients=7, seed=42)
    assert sum(len(s) for s in shards.values()) == len(df)


# ---------- 3: balanced per-client size ----------

def test_random_split_per_client_size_balanced() -> None:
    df = _make_df(n_rows=3500, n_slices=5)  # 700 rows × 5 slices
    shards = partition_clients(df, mode="random_split", n_clients=7, seed=42)
    sizes = sorted(len(s) for s in shards.values())
    expected = len(df) // 7
    # Each client gets either floor(N/7) or floor(N/7)+1 rows when N is not
    # exactly divisible — np.array_split semantics. Allow 1-row tolerance.
    assert max(sizes) - min(sizes) <= 1, (
        f"per-client sizes {sizes} should differ by at most 1"
    )
    # Mean exact
    assert sum(sizes) // 7 in (expected, expected + 1)


# ---------- 4: seed reproducible ----------

def test_random_split_same_seed_bit_equivalent() -> None:
    df = _make_df(n_rows=3000)
    s1 = partition_clients(df, mode="random_split", n_clients=7, seed=42)
    s2 = partition_clients(df, mode="random_split", n_clients=7, seed=42)
    assert sorted(s1.keys()) == sorted(s2.keys())
    for cid in s1:
        # DataFrames must be element-wise identical (same rows in same order)
        pd.testing.assert_frame_equal(
            s1[cid].reset_index(drop=True),
            s2[cid].reset_index(drop=True),
            check_dtype=True,
        )


# ---------- 5: different seed gives different partition ----------

def test_random_split_different_seed_different_partition() -> None:
    df = _make_df(n_rows=3000)
    s1 = partition_clients(df, mode="random_split", n_clients=7, seed=42)
    s2 = partition_clients(df, mode="random_split", n_clients=7, seed=999)
    # Some client at least must have a different row set
    n_diff = 0
    for cid in s1:
        if cid in s2 and not s1[cid].equals(s2[cid]):
            n_diff += 1
    assert n_diff > 0, "different seeds produced identical partition (hardcoded?)"


# ---------- 6: missing n_clients raises ----------

def test_random_split_missing_n_clients_raises() -> None:
    df = _make_df(n_rows=300)
    with pytest.raises(ValueError, match=r"random_split.*n_clients"):
        partition_clients(df, mode="random_split", seed=42)


# ---------- 7: bs_id marginal preserved (within 5pp) ----------

def test_random_split_preserves_bs_id_marginals() -> None:
    """A uniform random split should approximately preserve the bs_id
    distribution per client. With 3000 rows / 7 clients ≈ 428 rows each
    and 7 bs's with ≈14% global share each, sampling noise gives σ ≈
    sqrt(0.14 × 0.86 / 428) ≈ 1.7%. We allow 5pp tolerance."""
    df = _make_df(n_rows=3500, n_bs=7)
    shards = partition_clients(df, mode="random_split", n_clients=7, seed=42)
    global_bs = df["bs_id"].value_counts(normalize=True).sort_index()
    for cid, shard in shards.items():
        per_client_bs = shard["bs_id"].value_counts(normalize=True).sort_index()
        # Reindex to global to fill missing bs's with 0
        per_client_bs = per_client_bs.reindex(global_bs.index, fill_value=0.0)
        diffs = (per_client_bs - global_bs).abs()
        assert diffs.max() < 0.05, (
            f"client {cid}: bs_id distribution deviates from global by "
            f"{diffs.max():.3f}; per-client {per_client_bs.values}, "
            f"global {global_bs.values}"
        )
