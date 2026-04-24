"""Build sliding-window sequences STRICTLY within each (run_id, slice_id).

Each window of length L at step t contains features from [t-L+1, ..., t] and
predicts targets at t+1 (which already live on the same row after
``engineer_features`` did the shift).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..logging_utils import get_logger

log = get_logger(__name__)


def build_run_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    seq_len: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, Y) numpy arrays.

    X shape: (N_windows, seq_len, n_features)
    Y shape: (N_windows, n_targets)

    Windows DO NOT cross (run_id, slice_id) boundaries.
    """
    df = df.sort_values(["run_id", "slice_id", "step_idx"]).reset_index(drop=True)
    # Group into contiguous runs, produce per-group windows.
    Xs, Ys = [], []
    for _, g in df.groupby(["run_id", "slice_id"], observed=True, sort=False):
        n = len(g)
        if n < seq_len:
            continue
        X = g[feature_cols].to_numpy(dtype=np.float32, copy=False)
        Y = g[target_cols].to_numpy(dtype=np.float32, copy=False)
        # Windows aligned at the END: window i uses X[i-L+1..i], label Y[i].
        # i ranges from seq_len-1 to n-1 inclusive.
        windows = np.lib.stride_tricks.sliding_window_view(X, window_shape=seq_len, axis=0)
        # sliding_window_view → (n-L+1, F, L); transpose to (n-L+1, L, F).
        Xs.append(np.ascontiguousarray(windows.transpose(0, 2, 1)))
        Ys.append(Y[seq_len - 1:])
    if not Xs:
        return np.empty((0, seq_len, len(feature_cols)), dtype=np.float32), \
               np.empty((0, len(target_cols)), dtype=np.float32)
    X_all = np.concatenate(Xs, axis=0)
    Y_all = np.concatenate(Ys, axis=0)
    log.info("built %s sequences (seq_len=%d, features=%d, targets=%d)",
             f"{len(X_all):,}", seq_len, len(feature_cols), len(target_cols))
    return X_all, Y_all
