"""window_cache.build_windows_grouped must reproduce build_run_sequences EXACTLY (ADR D-3:
single-source windowing; the local cache optimization must not silently diverge)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "prea1"))
from window_cache import build_windows_grouped, client_windows_intact
from fl_oran.data_v2.sequences import build_run_sequences


def _df():
    # two (run_id, slice_id) groups of unequal length, with an extra short group that drops out
    return pd.DataFrame({
        "run_id":   [0] * 10 + [1] * 8 + [2] * 3,
        "slice_id": [0] * 10 + [0] * 8 + [0] * 3,
        "step_idx": list(range(10)) + list(range(8)) + list(range(3)),
        "f0": np.arange(21, dtype=float),
        "f1": np.arange(100, 121, dtype=float),
        "y": (np.arange(21) % 2).astype(float),
    })


def test_build_windows_grouped_matches_build_run_sequences():
    df = _df()
    feats, tgt = ["f0", "f1"], ["y"]
    X1, Y1 = build_run_sequences(df, feats, tgt, seq_len=5)
    X2, Y2, gid = build_windows_grouped(df, feats, tgt, seq_len=5)
    assert np.array_equal(X1, X2), "X diverges from build_run_sequences"
    assert np.array_equal(Y1, Y2), "Y diverges from build_run_sequences"
    # group 0 (10 rows -> 6 windows), group 1 (8 -> 4), group 2 (3 < seq_len -> dropped)
    assert gid.tolist() == [0] * 6 + [1] * 4
    assert len(X2) == 10


def test_client_windows_intact_indexes_by_group():
    df = _df()
    X, Y, gid = build_windows_grouped(df, ["f0", "f1"], ["y"], seq_len=5)
    Xc, Yc = client_windows_intact(X, Y, gid, {0})   # client holds group 0 only
    assert len(Xc) == 6 and np.array_equal(Xc, X[:6])
    Xc2, _ = client_windows_intact(X, Y, gid, {1})
    assert len(Xc2) == 4
