"""Feature engineering on the unified raw parquet — zero target leakage.

Task definition (FL-friendly, scientifically honest):
    Given past L steps of cell/slice state, forecast:
      - tx_brate_dl_Mbps_{t+1}   (regression)   downlink throughput 1 step ahead
      - sla_violation_{t+1}      (classification) dl_bler > 0.10 OR tx_errors > 1%

The forecast target is a FUTURE observation, not a linear function of present
inputs. Past KPIs as inputs is a legitimate autoregressive setup.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..logging_utils import get_logger

log = get_logger(__name__)

# Inputs at time t. Static per-run: bs_id, slice_id, sched, tr.
# These are model inputs; no derived target leakage.
CLEAN_FEATURES = [
    # identifiers (categorical, can be embedded or one-hot)
    "bs_id", "slice_id", "sched", "tr",
    # dynamic traffic
    "num_ues", "slice_prb",
    "sum_requested_prbs", "sum_granted_prbs",
    "tx_brate_dl_Mbps", "rx_brate_ul_Mbps",
    "tx_pkts_dl", "rx_pkts_ul",
    "dl_buffer_bytes", "ul_buffer_bytes",
    "dl_bler", "ul_bler",
    # physical quality
    "dl_mcs", "ul_mcs", "dl_cqi", "ul_sinr", "ul_rssi",
    # trend features (added by engineer_features)
    "tx_brate_dl_roll3", "tx_brate_dl_volatility",
]

REGRESSION_TARGETS = ["y_tx_brate_dl_next"]
CLASSIFICATION_TARGETS = ["y_sla_violation_next"]

# SLA threshold: uplink BLER > 10% counts as violation.
# (NB: downlink tx_errors are always 0 in this ColO-RAN dump, so we use
# uplink block-error rate — which does vary meaningfully (mean 14.6%).)
SLA_BLER_THRESHOLD = 0.10


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 1-step-ahead targets, within each (run_id, slice_id).

    Drops the final row of each (run, slice) where target is NaN.
    Also removes rows with obviously invalid inputs (NaN in key columns).
    """
    assert "run_id" in df.columns and "step_idx" in df.columns
    assert "slice_id" in df.columns
    df = df.sort_values(["run_id", "slice_id", "step_idx"]).reset_index(drop=True)

    # Replace inf with NaN first (can come from divisions in the raw CSV).
    traffic_cols = [
        "tx_brate_dl_Mbps", "rx_brate_ul_Mbps", "tx_pkts_dl", "rx_pkts_ul",
        "dl_buffer_bytes", "ul_buffer_bytes", "dl_bler", "ul_bler",
        "sum_requested_prbs", "sum_granted_prbs", "num_ues",
    ]
    quality_cols = ["dl_mcs", "ul_mcs", "dl_cqi", "ul_sinr", "ul_rssi"]
    for c in traffic_cols + quality_cols:
        if c in df.columns:
            df[c] = df[c].replace([np.inf, -np.inf], np.nan)

    # Traffic metrics: NaN → 0 (genuinely no traffic in that slot).
    for c in traffic_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype("float32")
    # Quality metrics: NaN → -1 sentinel (so the model can learn "not measured").
    for c in quality_cols:
        if c in df.columns:
            df[c] = df[c].fillna(-1).astype("float32")

    # Add simple trend features on tx_brate (rolling over 5 steps within (run, slice)).
    key = ["run_id", "slice_id"]
    if "tx_brate_dl_Mbps" in df.columns:
        g = df.groupby(key, observed=True)["tx_brate_dl_Mbps"]
        df["tx_brate_dl_roll3"] = g.transform(lambda x: x.rolling(3, min_periods=1).mean()).astype("float32")
        df["tx_brate_dl_volatility"] = g.transform(
            lambda x: x.rolling(5, min_periods=1).std().fillna(0)
        ).astype("float32")

    # Forecast target: shift values up by 1 within each (run, slice) group.
    df["y_tx_brate_dl_next"] = df.groupby(key, observed=True)["tx_brate_dl_Mbps"].shift(-1)
    # SLA violation label at t+1 — uplink BLER > threshold.
    next_ul_bler = df.groupby(key, observed=True)["ul_bler"].shift(-1) \
        if "ul_bler" in df.columns else pd.Series(0.0, index=df.index)
    df["y_sla_violation_next"] = (next_ul_bler > SLA_BLER_THRESHOLD).astype("float32")
    # Mark last-of-group rows as NaN (matching the tx_brate shift NaN pattern).
    df.loc[df["y_tx_brate_dl_next"].isna(), "y_sla_violation_next"] = np.nan

    before = len(df)
    df = df.dropna(subset=REGRESSION_TARGETS + CLASSIFICATION_TARGETS)
    # Also drop rows with inf/nan in core inputs.
    feature_cols_present = [c for c in CLEAN_FEATURES if c in df.columns]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols_present).reset_index(drop=True)
    log.info("feature engineering: %s → %s rows (dropped %s for NaN target/inputs)",
             f"{before:,}", f"{len(df):,}", f"{before - len(df):,}")

    return df
