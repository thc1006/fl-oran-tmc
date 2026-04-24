"""Tests for raw ColO-RAN CSV parsers."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fl_oran.data_raw import parse_slice_metrics, parse_bs_csv, parse_ue_csv


def _write_slice_csv(path: Path) -> Path:
    """Write a minimal slice metrics CSV matching the real file format."""
    content = (
        "Timestamp,num_ues,IMSI,RNTI,,slicing_enabled,slice_id,slice_prb,power_multiplier,scheduling_policy,,"
        "dl_mcs,dl_n_samples,dl_buffer [bytes],tx_brate downlink [Mbps],tx_pkts downlink,tx_errors downlink (%),dl_cqi,,"
        "ul_mcs,ul_n_samples,ul_buffer [bytes],rx_brate uplink [Mbps],rx_pkts uplink,rx_errors uplink (%),"
        "ul_rssi,ul_sinr,phr,,sum_requested_prbs,sum_granted_prbs,,dl_pmi,dl_ri,ul_n,ul_turbo_iters\n"
        "1617070531726,6,1010123456002,82,,1,2,5,1,0,,4.12,33,0,0.0092,9,0,8,,11.28,42,0,0.113,12,0,0,34.03,31,,17,30,,0,0,0,1\n"
        "1617070531976,6,1010123456002,82,,1,2,5,1,0,,4.50,35,0,0.0100,10,1,9,,11.30,44,0,0.115,13,0,1,34.10,32,,18,31,,0,0,0,1\n"
    )
    path.write_text(content)
    return path


def test_parse_slice_metrics_renames_columns(tmp_path: Path):
    p = _write_slice_csv(tmp_path / "metrics.csv")
    df = parse_slice_metrics(p)
    # Renamed columns must be present, originals removed.
    for c in ["tx_brate_dl_Mbps", "tx_errors_dl_pct", "dl_buffer_bytes",
              "rx_brate_ul_Mbps", "rx_errors_ul_pct"]:
        assert c in df.columns, f"missing {c}"
    for old in ["tx_brate downlink [Mbps]", "tx_errors downlink (%)"]:
        assert old not in df.columns


def test_parse_slice_metrics_drops_unnamed_columns(tmp_path: Path):
    p = _write_slice_csv(tmp_path / "metrics.csv")
    df = parse_slice_metrics(p)
    assert all(not c.startswith("Unnamed") for c in df.columns)


def test_parse_slice_metrics_row_count(tmp_path: Path):
    p = _write_slice_csv(tmp_path / "metrics.csv")
    df = parse_slice_metrics(p)
    assert len(df) == 2


def test_parse_bs_csv(tmp_path: Path):
    p = tmp_path / "bs.csv"
    p.write_text("time,nof_ue,dl_brate,ul_brate\n1000,0,0.0,0.0\n1250,1,2.0,1.0\n")
    df = parse_bs_csv(p)
    assert list(df.columns) == ["time", "nof_ue", "dl_brate", "ul_brate"]
    assert len(df) == 2


def test_parse_ue_csv(tmp_path: Path):
    p = tmp_path / "ue1.csv"
    p.write_text("time,cc,pci,rsrp,dl_snr,dl_brate\n0,0,0,0,0,0\n249,0,1,-57,14,583\n")
    df = parse_ue_csv(p)
    assert "dl_snr" in df.columns
    assert df["dl_snr"].iloc[1] == 14
