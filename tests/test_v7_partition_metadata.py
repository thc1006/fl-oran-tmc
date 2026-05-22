"""Partition metadata integrity (sequence-integrity evidence bundle, deliverable 2-iii).

Confirms the partition kwargs (mode, alpha, n_clients) are recorded in the serialized
artifact metadata (`asdict(V7Config)` -> summary.json `config`) and do NOT leak into
`algo_kwargs` (the strip step; see auto-memory `feedback-audit-before-launch`, where
partition kwargs riding on `algo_kwargs` caused a 60-cell crash).

The (run_id, slice_id)-not-split-across-client and window-contiguity invariants are
covered by tests/test_v7_run_random_partition.py and tests/test_v7_run_dirichlet_partition.py.
"""
from dataclasses import asdict

import pytest

from fl_oran.training.fl_v7 import V7Config


@pytest.mark.parametrize(
    "mode,alpha",
    [
        ("iid", None),
        ("random_split", None),
        ("run_random", None),
        ("dirichlet", 1.0),
        ("run_dirichlet", 0.1),
    ],
)
def test_partition_kwargs_in_metadata_and_not_in_algo_kwargs(mode, alpha):
    kw = dict(partition_mode=mode, n_clients=7)
    if alpha is not None:
        kw["alpha"] = alpha
    cfg = V7Config(**kw)
    meta = asdict(cfg)  # exactly what fl_v7 serializes into summary.json's "config"

    # (2-iii) partition kwargs are present and correct in the artifact metadata
    assert meta["partition_mode"] == mode
    assert meta["n_clients"] == 7
    if alpha is not None:
        assert meta["alpha"] == alpha

    # no leakage of partition kwargs into algo_kwargs (the strip step works)
    assert "partition_mode" not in cfg.algo_kwargs
    assert "alpha" not in cfg.algo_kwargs
    assert "n_clients" not in cfg.algo_kwargs


def test_run_level_modes_round_trip_in_cell_name():
    # the canonical cell name (built in __post_init__) encodes the partition mode,
    # so the artifact directory is itself auditable metadata
    assert "runrandom" in V7Config(partition_mode="run_random").name
    assert "rundir_a" in V7Config(partition_mode="run_dirichlet", alpha=0.1).name
