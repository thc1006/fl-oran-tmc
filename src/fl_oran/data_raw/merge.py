"""Convert the raw CSV tree to one tidy parquet.

Output schema (one row per Timestamp × bs × slice):

    # run identity
    sched, tr, tr_name, exp, bs_id, slice_id, run_id
    Timestamp, step_idx

    # direct physical KPIs (aggregated across UEs in the slice)
    num_ues, sum_requested_prbs, sum_granted_prbs
    tx_brate_dl_Mbps, rx_brate_ul_Mbps
    tx_pkts_dl, rx_pkts_ul
    tx_errors_dl_pct, rx_errors_ul_pct
    dl_mcs, ul_mcs, dl_cqi, ul_sinr, ul_rssi, phr
    dl_buffer_bytes, ul_buffer_bytes
    dl_n_samples, ul_n_samples
    dl_pmi, dl_ri, ul_n, ul_turbo_iters

    # engineering-friendly derivations
    dl_bler (= tx_errors_dl_pct / 100, clipped [0, 1])
    scheduling_policy (raw numeric), slice_prb (allocated PRB count for slice)

Streaming: writes each per-run aggregate to a pyarrow ParquetWriter incrementally
so we don't have to hold the whole dataset in memory.
"""
from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from ..logging_utils import get_logger
from .inventory import InventoryEntry, scan_inventory
from .parsers import parse_slice_metrics

log = get_logger(__name__)


_AGG_MEAN = [
    "tx_brate_dl_Mbps", "rx_brate_ul_Mbps",
    "tx_errors_dl_pct", "rx_errors_ul_pct",
    "dl_mcs", "ul_mcs", "dl_cqi", "ul_sinr", "ul_rssi", "phr",
    "dl_buffer_bytes", "ul_buffer_bytes",
    "dl_pmi", "dl_ri", "ul_n", "ul_turbo_iters",
]
_AGG_SUM = [
    "tx_pkts_dl", "rx_pkts_ul",
    "dl_n_samples", "ul_n_samples",
]
# sum_requested/granted_prbs are reported by each UE in the slice but represent
# the slice-level totals. All UEs in the same (slice, timestamp) should report
# the SAME value; we use "max" to be robust against the occasional stale 0.
_AGG_MAX = ["sum_requested_prbs", "sum_granted_prbs"]
_AGG_FIRST = ["num_ues", "slice_id", "slice_prb", "scheduling_policy"]


def _load_and_aggregate_run(entry: InventoryEntry) -> pd.DataFrame | None:
    """Load all per-UE slice metric CSVs in ``entry.slices_dir`` and aggregate
    to one row per (Timestamp, slice_id) — summing across UEs as appropriate."""
    if not entry.slices_dir.exists():
        return None
    frames: list[pd.DataFrame] = []
    for csv in entry.slices_dir.glob("*_metrics.csv"):
        try:
            df = parse_slice_metrics(csv)
        except Exception as e:
            log.warning("parse failed for %s: %s", csv, e)
            continue
        if len(df) == 0:
            continue
        frames.append(df)
    if not frames:
        return None
    big = pd.concat(frames, ignore_index=True)

    # Convert to numeric where possible.
    for col in big.columns:
        if col in ("IMSI", "RNTI"):
            continue
        big[col] = pd.to_numeric(big[col], errors="coerce")

    # Aggregate across UEs at the same (Timestamp, slice_id).
    agg_dict: dict = {}
    for c in _AGG_MEAN:
        if c in big.columns:
            agg_dict[c] = "mean"
    for c in _AGG_SUM:
        if c in big.columns:
            agg_dict[c] = "sum"
    for c in _AGG_MAX:
        if c in big.columns:
            agg_dict[c] = "max"
    for c in _AGG_FIRST:
        if c in big.columns:
            agg_dict[c] = "first"

    out = big.groupby(["Timestamp", "slice_id"], as_index=False, observed=True).agg(agg_dict)
    out["sched"] = entry.sched
    out["tr"] = entry.tr
    out["tr_name"] = f"tr{entry.tr}"
    out["exp"] = entry.exp
    out["bs_id"] = entry.bs
    out["run_id"] = entry.run_id

    # Order by time within the run; assign step_idx per (run, slice).
    out = out.sort_values(["slice_id", "Timestamp"]).reset_index(drop=True)
    out["step_idx"] = out.groupby("slice_id").cumcount()

    # Derived: dl_bler / ul_bler from % columns.
    if "tx_errors_dl_pct" in out.columns:
        out["dl_bler"] = (out["tx_errors_dl_pct"].fillna(0) / 100).clip(0, 1).astype("float32")
    if "rx_errors_ul_pct" in out.columns:
        out["ul_bler"] = (out["rx_errors_ul_pct"].fillna(0) / 100).clip(0, 1).astype("float32")

    # Cast for compact parquet.
    for col, dtype in [
        ("num_ues", "uint8"), ("slice_id", "uint8"),
        ("bs_id", "uint8"), ("sched", "uint8"), ("tr", "uint8"), ("exp", "uint8"),
        ("slice_prb", "uint8"), ("scheduling_policy", "uint8"),
        ("step_idx", "uint32"),
    ]:
        if col in out.columns:
            out[col] = out[col].astype(dtype, errors="ignore")
    return out


def build_unified_parquet(
    raw_root: str | Path,
    out_path: str | Path,
    *,
    limit_runs: int | None = None,
) -> Path:
    """Scan ``raw_root``, aggregate every run, **stream** to a single parquet.

    Memory-bounded: we never hold more than ~2 runs in RAM. ``limit_runs`` is
    for smoke tests — set to e.g. 10 to only process 10 runs.
    """
    raw_root = Path(raw_root)
    out_path = Path(out_path)
    entries = scan_inventory(raw_root)
    if limit_runs:
        entries = entries[:limit_runs]
        log.warning("limit_runs=%d — only processing %d entries", limit_runs, len(entries))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    total_rows = 0
    try:
        for entry in tqdm(entries, desc="merging runs"):
            part = _load_and_aggregate_run(entry)
            if part is None or len(part) == 0:
                continue
            table = pa.Table.from_pandas(part, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(str(out_path), table.schema, compression="snappy")
            writer.write_table(table)
            total_rows += len(part)
            del part, table
            gc.collect()
    finally:
        if writer is not None:
            writer.close()

    if total_rows == 0:
        raise RuntimeError("no data produced — check paths.")
    log.info("done. rows=%s → %s", f"{total_rows:,}", out_path)
    return out_path
