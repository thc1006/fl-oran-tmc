"""Tests for ``partition_clients(mode="run_random")`` — the sequence-integrity
control for the PREREG-A1 follow-up (natural-by-BS = BS-coherence vs artifact?).

``run_random`` assigns each WHOLE ``(run_id, slice_id)`` group to one client
(greedy least-loaded by rows over a seed-shuffled group order). Unlike
``random_split`` (row-level shuffle → scatters a run's timesteps across clients
→ ``build_run_sequences`` builds temporally-broken windows), ``run_random``
keeps each run intact (valid windows) while still breaking bs_id coherence.

Asserted invariants (the ones the gate's validity depends on):
  1. No ``(run_id, slice_id)`` group is split across clients.
  2. Each intact run keeps its CONTIGUOUS ``step_idx`` inside its client (the
     property that makes the windows valid — the whole point of the control).
  3. Total rows preserved; every row assigned exactly once.
  4. Per-client size is reasonably balanced (greedy least-loaded).
  5. bs_id coherence is broken (each client sees a mix of BS).
  6. Same seed → bit-equivalent; different seed → different partition.
  7. Missing ``n_clients`` raises.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fl_oran.data_v2.partition import partition_clients


def _make_df(n_bs: int = 7, n_slices: int = 3, runs_per_bs: int = 12,
             run_len: int = 50, seed: int = 0) -> pd.DataFrame:
    """ColO-RAN-shaped: each (bs, run) is one experiment with `run_len`
    CONTIGUOUS step_idx, carrying all `n_slices` slices. run_id is unique per
    (bs, run) and belongs to exactly one bs."""
    rng = np.random.default_rng(seed)
    rows = []
    for bs in range(1, n_bs + 1):
        for r in range(runs_per_bs):
            run_id = f"bs{bs}_run{r}"
            for sl in range(n_slices):
                steps = np.arange(run_len, dtype="int32")
                rows.append(pd.DataFrame({
                    "run_id": run_id,
                    "bs_id": np.uint8(bs),
                    "slice_id": np.uint8(sl),
                    "step_idx": steps,
                    "tx_brate_dl_Mbps": rng.uniform(0, 10, run_len).astype("float32"),
                }))
    df = pd.concat(rows, ignore_index=True)
    # Shuffle row order so the partitioner can't rely on input ordering.
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _tagged_concat(shards: dict) -> pd.DataFrame:
    out = []
    for cid, s in shards.items():
        s = s.copy()
        s["__client"] = cid
        out.append(s)
    return pd.concat(out, ignore_index=True)


# ---------- 1: no (run_id, slice_id) group split across clients ----------

def test_run_random_no_group_split_across_clients() -> None:
    df = _make_df()
    shards = partition_clients(df, mode="run_random", n_clients=7, seed=42)
    allrows = _tagged_concat(shards)
    span = allrows.groupby(["run_id", "slice_id"], observed=True)["__client"].nunique()
    bad = span[span > 1]
    assert bad.empty, f"(run_id,slice_id) groups split across clients: {bad.to_dict()}"


# ---------- 2: contiguous run preserved inside each client ----------

def test_run_random_preserves_contiguous_runs() -> None:
    """The whole reason this control exists: a client's windows must be over
    contiguous step_idx, not row-level fragments (cf. random_split)."""
    df = _make_df()
    orig = df.groupby(["run_id", "slice_id"], observed=True)["step_idx"].apply(
        lambda x: tuple(sorted(int(v) for v in x))
    )
    shards = partition_clients(df, mode="run_random", n_clients=7, seed=42)
    for cid, s in shards.items():
        for key, g in s.groupby(["run_id", "slice_id"], observed=True):
            steps = tuple(sorted(int(v) for v in g["step_idx"]))
            assert steps == orig[key], (
                f"client {cid} group {key} fragmented vs original"
            )
            assert steps == tuple(range(steps[0], steps[0] + len(steps))), (
                f"client {cid} group {key} step_idx not contiguous: {steps[:5]}..."
            )


# ---------- 3: total rows preserved, each row exactly once ----------

def test_run_random_preserves_total_rows() -> None:
    df = _make_df()
    shards = partition_clients(df, mode="run_random", n_clients=7, seed=42)
    assert sum(len(s) for s in shards.values()) == len(df)


# ---------- 4: balanced per-client size ----------

def test_run_random_balanced_sizes() -> None:
    df = _make_df()
    shards = partition_clients(df, mode="run_random", n_clients=7, seed=42)
    sizes = [len(s) for s in shards.values()]
    mean = len(df) / 7
    # Greedy least-loaded by rows keeps every client within one max-group of
    # the mean. Groups are equal-size here, so balance is near-perfect.
    assert min(sizes) >= 0.7 * mean, f"a client is starved: sizes={sorted(sizes)}"
    assert max(sizes) <= 1.3 * mean, f"a client is overloaded: sizes={sorted(sizes)}"


# ---------- 5: bs coherence broken (clients see mixed BS) ----------

def test_run_random_breaks_bs_coherence() -> None:
    df = _make_df()
    shards = partition_clients(df, mode="run_random", n_clients=7, seed=42)
    for cid, s in shards.items():
        assert s["bs_id"].nunique() > 1, (
            f"client {cid} has a single bs_id ({s['bs_id'].unique()}) — "
            "coherence not broken"
        )


# ---------- 6: reproducibility ----------

def test_run_random_same_seed_bit_equivalent() -> None:
    df = _make_df()
    s1 = partition_clients(df, mode="run_random", n_clients=7, seed=42)
    s2 = partition_clients(df, mode="run_random", n_clients=7, seed=42)
    assert sorted(s1.keys()) == sorted(s2.keys())
    for cid in s1:
        pd.testing.assert_frame_equal(
            s1[cid].reset_index(drop=True), s2[cid].reset_index(drop=True),
            check_dtype=True,
        )


def test_run_random_different_seed_differs() -> None:
    df = _make_df()
    s1 = partition_clients(df, mode="run_random", n_clients=7, seed=42)
    s2 = partition_clients(df, mode="run_random", n_clients=7, seed=999)
    n_diff = sum(
        1 for cid in s1
        if cid in s2 and not s1[cid].reset_index(drop=True).equals(
            s2[cid].reset_index(drop=True)
        )
    )
    assert n_diff > 0, "different seeds produced identical partition"


# ---------- 7: missing n_clients raises ----------

def test_run_random_missing_n_clients_raises() -> None:
    df = _make_df()
    with pytest.raises(ValueError, match=r"run_random.*n_clients"):
        partition_clients(df, mode="run_random", seed=42)
