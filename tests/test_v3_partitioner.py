"""TDD for Non-IID client partitioner (Task 24)."""
from __future__ import annotations

import pandas as pd
import pytest


def test_iid_partition_assigns_by_bs_id_only():
    from fl_oran.data_v2.partition import partition_clients
    df = pd.DataFrame({"bs_id": [1, 2, 3, 1, 2], "slice_id": [0, 1, 2, 0, 1], "x": range(5)})
    shards = partition_clients(df, mode="iid")
    assert set(shards.keys()) == {1, 2, 3}
    # Each client gets exactly its own bs_id's rows
    assert set(shards[1]["bs_id"].unique()) == {1}
    assert len(shards[1]) == 2


def test_noniid_slice_partition_respects_mapping():
    from fl_oran.data_v2.partition import partition_clients
    df = pd.DataFrame({
        "bs_id":   [1, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        "slice_id": [0, 1, 2, 0, 0, 1, 2, 0, 1, 2],
        "x":       range(10),
    })
    # Mapping: client 1 → slice 0 only; client 2 → slice 1 only; client 3 → slice 2 only
    mapping = {1: [0], 2: [1], 3: [2]}
    shards = partition_clients(df, mode="noniid_slice", client_slice_map=mapping)
    assert set(shards.keys()) == {1, 2, 3}
    assert set(shards[1]["slice_id"].unique()) == {0}
    assert set(shards[2]["slice_id"].unique()) == {1}
    assert set(shards[3]["slice_id"].unique()) == {2}
    # Client 1 still only sees its own bs_id
    assert set(shards[1]["bs_id"].unique()) == {1}


def test_noniid_partition_drops_rows_with_no_match():
    from fl_oran.data_v2.partition import partition_clients
    df = pd.DataFrame({
        "bs_id":   [1, 2, 3],
        "slice_id": [0, 1, 2],
        "x":       range(3),
    })
    # client 1 assigned slice 2 only; its own bs has only slice 0 → empty shard
    mapping = {1: [2], 2: [1], 3: [2]}
    shards = partition_clients(df, mode="noniid_slice", client_slice_map=mapping)
    # client 1 → bs_id=1 ∩ slice_id=2 → 0 rows → omitted
    assert 1 not in shards
    assert 2 in shards
    assert 3 in shards


def test_partition_unknown_mode_raises():
    from fl_oran.data_v2.partition import partition_clients
    df = pd.DataFrame({"bs_id": [1], "slice_id": [0], "x": [1]})
    with pytest.raises(ValueError):
        partition_clients(df, mode="bogus")


def test_partition_noniid_requires_mapping():
    from fl_oran.data_v2.partition import partition_clients
    df = pd.DataFrame({"bs_id": [1], "slice_id": [0], "x": [1]})
    with pytest.raises(ValueError):
        partition_clients(df, mode="noniid_slice", client_slice_map=None)
