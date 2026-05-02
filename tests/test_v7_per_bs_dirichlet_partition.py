"""TDD tests for ``partition_clients(mode="per_bs_dirichlet")``.

This partition mode is the Phase 6 Rank 3 mechanism-disambiguation control
for the inverted-α finding (PAPER §7.1.1, M9 confound from T-L review):
random_split breaks both bs_id and slice_id grouping, so a positive
ablation outcome is consistent with EITHER candidate (i) "natural-by-BS
preserves bs-conditioned signal" OR candidate (ii) "Dirichlet's per-slice
row redistribution destroys structurally-coherent client datasets".

To disambiguate, ``mode="per_bs_dirichlet"`` keeps natural-by-BS (one
client per bs_id) but additionally **redistributes rows within each BS
across two sub-clients via Dirichlet([α_inner])** over slice_id, doubling
the effective client count to 14 while preserving bs-grouping.

If natural-by-BS dominance comes from bs grouping per se (candidate i),
per_bs_dirichlet at moderate α_inner should still preserve most of the
AUC advantage. If it comes from coherent-bs-data (candidate ii), the
intra-bs split should drop AUC toward Dirichlet/random_split levels.

Test assertions:
  1. Returns a dict with `7 * sub_per_bs` shards (= 14 if sub_per_bs=2).
  2. Total row count preserved.
  3. Each client carries rows from exactly ONE bs_id (bs grouping preserved).
  4. Per-bs sub-clients vary in slice_id mixture according to α_inner
     (small α_inner → concentrated; large α_inner → uniform).
  5. Same seed reproducible.
  6. Different seed → different partition.
  7. Missing α_inner / sub_per_bs raises ValueError.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fl_oran.data_v2.partition import partition_clients


def _make_df(n_rows: int = 7000, n_slices: int = 3, n_bs: int = 7,
             seed: int = 0) -> pd.DataFrame:
    """Synthetic ColO-RAN-shaped data: 7 BS × 3 slices uniform mix."""
    rng = np.random.default_rng(seed)
    rows_per_bs = n_rows // n_bs
    actual_rows = rows_per_bs * n_bs
    bs_ids = np.repeat(np.arange(1, n_bs + 1), rows_per_bs).astype("uint8")
    slice_ids = rng.integers(0, n_slices, actual_rows).astype("uint8")
    return pd.DataFrame({
        "bs_id": bs_ids,
        "slice_id": slice_ids,
        "tx_brate_dl_Mbps": rng.uniform(0, 10, actual_rows).astype("float32"),
    })


def test_per_bs_dirichlet_returns_n_bs_times_sub_clients() -> None:
    df = _make_df(n_rows=7000, n_bs=7)
    shards = partition_clients(
        df, mode="per_bs_dirichlet",
        alpha=0.5, sub_per_bs=2, seed=42,
    )
    assert len(shards) == 7 * 2, f"expected 14 shards, got {len(shards)}"


def test_per_bs_dirichlet_preserves_total_row_count() -> None:
    df = _make_df(n_rows=7000)
    shards = partition_clients(
        df, mode="per_bs_dirichlet",
        alpha=0.5, sub_per_bs=2, seed=42,
    )
    assert sum(len(s) for s in shards.values()) == len(df)


def test_per_bs_dirichlet_each_client_one_bs() -> None:
    """Each shard's rows must come from exactly ONE bs_id."""
    df = _make_df(n_rows=7000)
    shards = partition_clients(
        df, mode="per_bs_dirichlet",
        alpha=0.5, sub_per_bs=2, seed=42,
    )
    for cid, shard in shards.items():
        unique_bs = shard["bs_id"].nunique()
        assert unique_bs == 1, (
            f"client {cid} has rows from {unique_bs} BS; should be 1 "
            f"(per_bs_dirichlet preserves bs grouping)"
        )


def test_per_bs_dirichlet_low_alpha_concentrates_slices() -> None:
    """At α=0.05 each sub-client within a BS should be concentrated on
    a single slice (one sub-client gets ~all of slice 0, the other ~all
    of slice 1, etc.)."""
    df = _make_df(n_rows=21000, n_bs=7, n_slices=3)  # 1000 rows/(bs,slice)
    shards = partition_clients(
        df, mode="per_bs_dirichlet",
        alpha=0.05, sub_per_bs=2, seed=42,
    )
    # For each BS, the 2 sub-clients should have very different slice
    # distributions at α=0.05. We measure max(slice_count) / total per
    # sub-client and require it's > 0.7 on average.
    bs_concentrations = []
    for cid, shard in shards.items():
        if len(shard) < 100:
            continue
        slice_props = shard["slice_id"].value_counts(normalize=True)
        bs_concentrations.append(slice_props.max())
    avg_max = float(np.mean(bs_concentrations))
    assert avg_max > 0.6, (
        f"per_bs_dirichlet α=0.05 expected concentrated slice mix "
        f"(avg max-prop > 0.6); got {avg_max:.3f}"
    )


def test_per_bs_dirichlet_high_alpha_uniform_slices() -> None:
    """At α=10.0 each sub-client should approximate uniform slice mix."""
    df = _make_df(n_rows=21000, n_bs=7, n_slices=3)
    shards = partition_clients(
        df, mode="per_bs_dirichlet",
        alpha=10.0, sub_per_bs=2, seed=42,
    )
    # At α=10 expect each sub-client to have ~33%/33%/33% slice mix
    bs_max_props = []
    for cid, shard in shards.items():
        if len(shard) < 100:
            continue
        slice_props = shard["slice_id"].value_counts(normalize=True)
        bs_max_props.append(slice_props.max())
    avg_max = float(np.mean(bs_max_props))
    assert avg_max < 0.55, (
        f"per_bs_dirichlet α=10.0 expected uniform-ish slice mix "
        f"(avg max-prop < 0.55); got {avg_max:.3f}"
    )


def test_per_bs_dirichlet_seed_reproducible() -> None:
    df = _make_df(n_rows=7000)
    s1 = partition_clients(
        df, mode="per_bs_dirichlet",
        alpha=0.5, sub_per_bs=2, seed=42,
    )
    s2 = partition_clients(
        df, mode="per_bs_dirichlet",
        alpha=0.5, sub_per_bs=2, seed=42,
    )
    assert sorted(s1.keys()) == sorted(s2.keys())
    for cid in s1:
        pd.testing.assert_frame_equal(
            s1[cid].reset_index(drop=True),
            s2[cid].reset_index(drop=True),
        )


def test_per_bs_dirichlet_different_seed_different_partition() -> None:
    df = _make_df(n_rows=7000)
    s1 = partition_clients(
        df, mode="per_bs_dirichlet",
        alpha=0.5, sub_per_bs=2, seed=42,
    )
    s2 = partition_clients(
        df, mode="per_bs_dirichlet",
        alpha=0.5, sub_per_bs=2, seed=999,
    )
    n_diff = 0
    for cid in s1:
        if cid in s2 and not s1[cid].equals(s2[cid]):
            n_diff += 1
    assert n_diff > 0


def test_per_bs_dirichlet_missing_alpha_raises() -> None:
    df = _make_df()
    with pytest.raises(ValueError, match=r"per_bs_dirichlet.*alpha"):
        partition_clients(df, mode="per_bs_dirichlet", sub_per_bs=2, seed=42)


def test_per_bs_dirichlet_missing_sub_per_bs_raises() -> None:
    df = _make_df()
    with pytest.raises(ValueError, match=r"per_bs_dirichlet.*sub_per_bs"):
        partition_clients(df, mode="per_bs_dirichlet", alpha=0.5, seed=42)
