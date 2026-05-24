"""Disk-cached, group-id-tagged windowing -- local 4060 Ti optimization for the Delta_seq pipeline.

The verified local bottleneck is CPU-side pandas windowing (build_run_sequences groupby), rebuilt
redundantly per (target x mode x seed x arch). This:
  (1) windows ONCE per (target, frac) and caches to .npz -> every rerun (new arch/seed/diagnostic)
      loads from disk instead of re-windowing;
  (2) tags each window with its (run_id, slice_id) group id -> INTACT-mode partitions index by
      group->client with NO re-windowing (whole groups go to one client by construction).

build_windows_grouped reproduces build_run_sequences' windowing EXACTLY (same sort, same END-aligned
windows, same Y alignment) and is guarded by a bit-exact regression test (ADR D-3 single-source rule:
we do not silently diverge from the validated windower).
"""
from __future__ import annotations
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

GROUP = ["run_id", "slice_id"]


def build_windows_grouped(df: pd.DataFrame, feature_cols: list[str], target_cols: list[str],
                          seq_len: int = 5):
    """(X, Y, gid). X/Y identical to build_run_sequences; gid[i] = contiguous integer id of the
    (run_id, slice_id) group that window i came from (groups numbered in groupby(sort=False) order)."""
    df = df.sort_values(GROUP + ["step_idx"]).reset_index(drop=True)
    Xs, Ys, gids = [], [], []
    for gi, (_, g) in enumerate(df.groupby(GROUP, observed=True, sort=False)):
        n = len(g)
        if n < seq_len:
            continue
        X = g[feature_cols].to_numpy(dtype=np.float32, copy=False)
        Y = g[target_cols].to_numpy(dtype=np.float32, copy=False)
        w = np.lib.stride_tricks.sliding_window_view(X, window_shape=seq_len, axis=0)
        Xs.append(np.ascontiguousarray(w.transpose(0, 2, 1)))
        Ys.append(Y[seq_len - 1:])
        gids.append(np.full(n - seq_len + 1, gi, dtype=np.int32))
    if not Xs:
        return (np.empty((0, seq_len, len(feature_cols)), np.float32),
                np.empty((0, len(target_cols)), np.float32), np.empty(0, np.int32))
    return np.concatenate(Xs), np.concatenate(Ys), np.concatenate(gids)


def cached_windows(df, feature_cols, target_cols, key: str, cache_dir="artifacts/prea1/wincache",
                   seq_len: int = 5):
    """Load (X, Y, gid) from .npz cache keyed by ``key`` + the feature/target/seq signature;
    build + save on miss. ``key`` should encode the (target, frac, split) so reruns hit the cache."""
    sig = hashlib.md5(f"{key}|{feature_cols}|{target_cols}|{seq_len}".encode()).hexdigest()[:16]
    p = Path(cache_dir) / f"win_{sig}.npz"
    if p.exists():
        d = np.load(p)
        return d["X"], d["Y"], d["gid"]
    X, Y, gid = build_windows_grouped(df, feature_cols, target_cols, seq_len)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, X=X, Y=Y, gid=gid)
    return X, Y, gid


def client_windows_intact(X, Y, gid, client_groups):
    """Index cached windows for an INTACT-mode client = the windows whose group id is assigned to
    that client. NO re-windowing. ``client_groups`` is a set/array of group ids for this client."""
    mask = np.isin(gid, np.asarray(list(client_groups)))
    return X[mask], Y[mask]
