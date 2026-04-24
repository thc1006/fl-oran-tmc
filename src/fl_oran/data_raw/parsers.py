"""Robust parsers for the three CSV variants in the raw ColO-RAN dataset.

The slice-metrics CSVs have irregular headers with blank columns (e.g.
``Timestamp,num_ues,IMSI,RNTI,,slicing_enabled,...``) — we normalise these on
read so the rest of the pipeline sees a clean schema.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..logging_utils import get_logger

log = get_logger(__name__)


# Normalised column names we keep from each file type.
SLICE_METRICS_KEEP = [
    "Timestamp", "num_ues", "IMSI", "RNTI",
    "slicing_enabled", "slice_id", "slice_prb", "power_multiplier", "scheduling_policy",
    "dl_mcs", "dl_n_samples", "dl_buffer_bytes",
    "tx_brate_dl_Mbps", "tx_pkts_dl", "tx_errors_dl_pct", "dl_cqi",
    "ul_mcs", "ul_n_samples", "ul_buffer_bytes",
    "rx_brate_ul_Mbps", "rx_pkts_ul", "rx_errors_ul_pct", "ul_rssi", "ul_sinr", "phr",
    "sum_requested_prbs", "sum_granted_prbs",
    "dl_pmi", "dl_ri", "ul_n", "ul_turbo_iters",
]

_SLICE_RENAME = {
    "dl_buffer [bytes]": "dl_buffer_bytes",
    "tx_brate downlink [Mbps]": "tx_brate_dl_Mbps",
    "tx_pkts downlink": "tx_pkts_dl",
    "tx_errors downlink (%)": "tx_errors_dl_pct",
    "ul_buffer [bytes]": "ul_buffer_bytes",
    "rx_brate uplink [Mbps]": "rx_brate_ul_Mbps",
    "rx_pkts uplink": "rx_pkts_ul",
    "rx_errors uplink (%)": "rx_errors_ul_pct",
}


def parse_slice_metrics(path: str | Path) -> pd.DataFrame:
    """Parse one ``{IMSI}_metrics.csv`` into a tidy DataFrame.

    Handles:
    - blank header columns (``,,``)
    - space-containing column names
    - stray NaN rows
    """
    path = Path(path)
    df = pd.read_csv(path, low_memory=False)
    # Drop unnamed columns (blank headers become ``Unnamed: N``).
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed:")]
    df = df.rename(columns=_SLICE_RENAME)
    # Some files have extra whitespace in header names.
    df.columns = [c.strip() for c in df.columns]
    # Keep only known columns (drops anything unexpected).
    keep = [c for c in SLICE_METRICS_KEEP if c in df.columns]
    df = df[keep]
    return df


def parse_bs_csv(path: str | Path) -> pd.DataFrame:
    """Parse ``bs{N}.csv`` — BS-level aggregate: time, nof_ue, dl_brate, ul_brate."""
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df


def parse_ue_csv(path: str | Path) -> pd.DataFrame:
    """Parse ``ue{N}.csv`` — per-UE physical-layer measurements."""
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df
