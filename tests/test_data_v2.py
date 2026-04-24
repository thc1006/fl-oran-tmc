"""Tests for v2 feature engineering, split, and sequence building."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fl_oran.data_v2 import (
    CLEAN_FEATURES,
    CLASSIFICATION_TARGETS,
    REGRESSION_TARGETS,
    build_run_sequences,
    engineer_features,
    ood_split_by_tr,
)


def _make_synthetic_raw(n_runs: int = 3, steps: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for r in range(n_runs):
        for s in (0, 1, 2):
            base = rng.uniform(1, 5)
            for t in range(steps):
                rows.append(dict(
                    run_id=f"run{r}", bs_id=1, slice_id=s, sched=0, tr=r, exp=1,
                    step_idx=t, Timestamp=t * 250,
                    num_ues=int(rng.integers(1, 8)),
                    slice_prb=int(rng.integers(1, 15)),
                    sum_requested_prbs=float(rng.uniform(0, 500)),
                    sum_granted_prbs=float(rng.uniform(0, 400)),
                    tx_brate_dl_Mbps=float(base + rng.normal(0, 0.1)),
                    rx_brate_ul_Mbps=float(rng.uniform(0, 0.5)),
                    tx_pkts_dl=int(rng.integers(0, 30)),
                    rx_pkts_ul=int(rng.integers(0, 30)),
                    dl_mcs=float(rng.uniform(0, 28)),
                    ul_mcs=float(rng.uniform(0, 28)),
                    dl_cqi=float(rng.uniform(0, 15)),
                    ul_sinr=float(rng.uniform(-5, 30)),
                    ul_rssi=float(rng.uniform(-110, -60)),
                    dl_buffer_bytes=float(rng.uniform(0, 1e5)),
                    ul_buffer_bytes=float(rng.uniform(0, 1e3)),
                    dl_bler=float(rng.uniform(0, 0.2)),
                    ul_bler=float(rng.uniform(0, 0.1)),
                    tx_errors_dl_pct=float(rng.uniform(0, 2)),
                ))
    return pd.DataFrame(rows)


def test_engineer_features_drops_last_row_per_group():
    df = _make_synthetic_raw(n_runs=2, steps=10)
    before = len(df)  # 2 runs × 3 slices × 10 = 60
    out = engineer_features(df)
    # Each (run, slice) group loses exactly 1 row (the last one — NaN target).
    assert len(out) == before - 2 * 3


def test_engineer_features_has_targets():
    df = _make_synthetic_raw()
    out = engineer_features(df)
    for t in REGRESSION_TARGETS + CLASSIFICATION_TARGETS:
        assert t in out.columns
        assert not out[t].isna().any()


def test_engineer_features_target_is_future_value():
    df = _make_synthetic_raw(n_runs=1, steps=5)
    out = engineer_features(df).sort_values(["run_id", "slice_id", "step_idx"]).reset_index(drop=True)
    # For slice 0, row with step_idx=i should have y_tx_brate_dl_next == tx_brate at step_idx=i+1.
    for sid in (0, 1, 2):
        g = out[out.slice_id == sid].reset_index(drop=True)
        raw_g = df[(df.slice_id == sid)].sort_values("step_idx").reset_index(drop=True)
        for i in range(len(g)):
            t = int(g["step_idx"].iloc[i])
            expected = float(raw_g.loc[raw_g["step_idx"] == t + 1, "tx_brate_dl_Mbps"].iloc[0])
            actual = float(g["y_tx_brate_dl_next"].iloc[i])
            assert abs(actual - expected) < 1e-4, f"slice {sid} step {t}: {actual} vs {expected}"


def test_ood_split_partitions():
    df = _make_synthetic_raw(n_runs=5, steps=10)
    df = engineer_features(df)
    split = ood_split_by_tr(df, train_tr=[0, 1, 2], val_tr=[3], test_tr=[4])
    assert set(split.train["tr"].unique()) == {0, 1, 2}
    assert set(split.val["tr"].unique()) == {3}
    assert set(split.test["tr"].unique()) == {4}
    assert len(split.train) + len(split.val) + len(split.test) == len(df)


def test_ood_split_rejects_overlap():
    df = _make_synthetic_raw()
    df = engineer_features(df)
    with pytest.raises(ValueError):
        ood_split_by_tr(df, train_tr=[0, 1], val_tr=[1], test_tr=[2])


def test_build_run_sequences_no_cross_boundary():
    """Windows must never span two different (run, slice) groups."""
    df = _make_synthetic_raw(n_runs=2, steps=8)
    df = engineer_features(df)
    feats = [c for c in CLEAN_FEATURES if c in df.columns]
    X, Y = build_run_sequences(df, feats, REGRESSION_TARGETS, seq_len=3)
    # Each group has (8 - 1) = 7 rows after drop, and 7 - 3 + 1 = 5 windows per group.
    # Total: 2 runs × 3 slices × 5 = 30.
    assert X.shape == (30, 3, len(feats))
    assert Y.shape == (30, 1)


def test_build_run_sequences_shape():
    df = _make_synthetic_raw(n_runs=1, steps=20)
    df = engineer_features(df)
    feats = [c for c in CLEAN_FEATURES if c in df.columns]
    X, Y = build_run_sequences(df, feats, REGRESSION_TARGETS, seq_len=5)
    # After drop: 19 rows per group. 19 - 5 + 1 = 15 windows per group. 3 groups = 45.
    assert X.shape == (45, 5, len(feats))
