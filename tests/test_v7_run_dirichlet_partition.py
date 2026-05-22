"""Tests for ``partition_clients(mode="run_dirichlet")`` — the run-level analog
of ``dirichlet`` (the decisive control for the sequence-integrity confound).

``run_dirichlet`` distributes, per slice, that slice's WHOLE ``(run_id)`` groups
across clients by ``Dir(alpha)`` — intact runs (valid windows), Dirichlet-skewed
allocation. Comparing ``run_dirichlet(alpha)`` vs ``dirichlet(alpha)`` at matched
alpha isolates sequence corruption from heterogeneity.

Invariants:
  1. No ``(run_id, slice_id)`` group split across clients (intact).
  2. Each run keeps CONTIGUOUS ``step_idx`` inside its client (valid windows).
  3. Dirichlet skew: smaller alpha -> more concentrated client sizes.
  4. Total rows preserved; reproducible by seed.
  5. Missing ``alpha`` / ``n_clients`` raises.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fl_oran.data_v2.partition import partition_clients


def _make_df(n_bs: int = 7, n_slices: int = 3, runs_per_bs: int = 12,
             run_len: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for bs in range(1, n_bs + 1):
        for r in range(runs_per_bs):
            run_id = f"bs{bs}_run{r}"
            for sl in range(n_slices):
                rows.append(pd.DataFrame({
                    "run_id": run_id,
                    "bs_id": np.uint8(bs),
                    "slice_id": np.uint8(sl),
                    "step_idx": np.arange(run_len, dtype="int32"),
                    "tx_brate_dl_Mbps": rng.uniform(0, 10, run_len).astype("float32"),
                }))
    df = pd.concat(rows, ignore_index=True)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _padded_sizes(shards: dict, n_clients: int) -> np.ndarray:
    sizes = [0] * n_clients
    for cid, s in shards.items():
        sizes[int(cid)] = len(s)
    return np.array(sizes, dtype=float)


def test_run_dirichlet_no_group_split() -> None:
    df = _make_df()
    shards = partition_clients(df, mode="run_dirichlet", alpha=0.5, n_clients=7, seed=42)
    tag = []
    for cid, s in shards.items():
        s = s.copy(); s["__c"] = cid; tag.append(s)
    allrows = pd.concat(tag, ignore_index=True)
    span = allrows.groupby(["run_id", "slice_id"], observed=True)["__c"].nunique()
    assert (span == 1).all(), f"groups split across clients: {span[span>1].to_dict()}"


def test_run_dirichlet_preserves_contiguous_runs() -> None:
    df = _make_df()
    orig = df.groupby(["run_id", "slice_id"], observed=True)["step_idx"].apply(
        lambda x: tuple(sorted(int(v) for v in x)))
    shards = partition_clients(df, mode="run_dirichlet", alpha=0.5, n_clients=7, seed=42)
    for cid, s in shards.items():
        for key, g in s.groupby(["run_id", "slice_id"], observed=True):
            steps = tuple(sorted(int(v) for v in g["step_idx"]))
            assert steps == orig[key], f"client {cid} group {key} fragmented"
            assert steps == tuple(range(steps[0], steps[0] + len(steps))), \
                f"client {cid} group {key} not contiguous"


def test_run_dirichlet_skew_increases_as_alpha_decreases() -> None:
    df = _make_df()
    lo = partition_clients(df, mode="run_dirichlet", alpha=0.05, n_clients=7, seed=1)
    hi = partition_clients(df, mode="run_dirichlet", alpha=100.0, n_clients=7, seed=1)
    s_lo, s_hi = _padded_sizes(lo, 7), _padded_sizes(hi, 7)
    cv_lo = s_lo.std() / (s_lo.mean() + 1e-9)
    cv_hi = s_hi.std() / (s_hi.mean() + 1e-9)
    assert cv_lo > cv_hi, (
        f"alpha=0.05 CV {cv_lo:.3f} should exceed alpha=100 CV {cv_hi:.3f}"
    )


def test_run_dirichlet_preserves_total_rows() -> None:
    df = _make_df()
    shards = partition_clients(df, mode="run_dirichlet", alpha=0.5, n_clients=7, seed=42)
    assert sum(len(s) for s in shards.values()) == len(df)


def test_run_dirichlet_same_seed_bit_equivalent() -> None:
    df = _make_df()
    a = partition_clients(df, mode="run_dirichlet", alpha=0.5, n_clients=7, seed=42)
    b = partition_clients(df, mode="run_dirichlet", alpha=0.5, n_clients=7, seed=42)
    assert sorted(a.keys()) == sorted(b.keys())
    for cid in a:
        pd.testing.assert_frame_equal(
            a[cid].reset_index(drop=True), b[cid].reset_index(drop=True), check_dtype=True)


def test_run_dirichlet_missing_alpha_raises() -> None:
    df = _make_df()
    with pytest.raises(ValueError, match=r"run_dirichlet.*alpha"):
        partition_clients(df, mode="run_dirichlet", n_clients=7, seed=42)


def test_run_dirichlet_missing_n_clients_raises() -> None:
    df = _make_df()
    with pytest.raises(ValueError, match=r"run_dirichlet.*n_clients"):
        partition_clients(df, mode="run_dirichlet", alpha=0.5, seed=42)
