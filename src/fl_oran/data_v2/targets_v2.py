"""Target builders that shift values 1 step into the future within groups."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..logging_utils import get_logger

log = get_logger(__name__)


def add_classification_target(
    df: pd.DataFrame,
    *,
    column: str,
    threshold: float,
    target_name: str,
    group_keys: tuple[str, ...] = ("run_id", "slice_id"),
    sort_keys: tuple[str, ...] = ("run_id", "slice_id", "step_idx"),
) -> pd.DataFrame:
    """Binary y_{t+1} = (column_{t+1} > threshold), respecting group boundaries.

    The last row of every group has NaN (no next step) and is dropped.
    """
    df = df.sort_values(list(sort_keys)).reset_index(drop=True)
    future = df.groupby(list(group_keys), observed=True)[column].shift(-1)
    df[target_name] = (future > threshold).astype("float32")
    df.loc[future.isna(), target_name] = np.nan
    before = len(df)
    df = df.dropna(subset=[target_name]).reset_index(drop=True)
    log.info("classification target '%s' (> %.3f of '%s'): %s → %s rows",
             target_name, threshold, column, f"{before:,}", f"{len(df):,}")
    return df


def add_regression_target(
    df: pd.DataFrame,
    *,
    column: str,
    horizon: int,
    target_name: str,
    group_keys: tuple[str, ...] = ("run_id", "slice_id"),
    sort_keys: tuple[str, ...] = ("run_id", "slice_id", "step_idx"),
) -> pd.DataFrame:
    """y_{t+horizon} = column at t+horizon, within each group."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    df = df.sort_values(list(sort_keys)).reset_index(drop=True)
    future = df.groupby(list(group_keys), observed=True)[column].shift(-horizon)
    df[target_name] = future.astype("float32")
    before = len(df)
    df = df.dropna(subset=[target_name]).reset_index(drop=True)
    log.info("regression target '%s' (h=%d of '%s'): %s → %s rows",
             target_name, horizon, column, f"{before:,}", f"{len(df):,}")
    return df
