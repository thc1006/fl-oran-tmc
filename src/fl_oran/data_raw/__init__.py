"""Raw ColO-RAN CSV parsing and aggregation.

Dataset layout (as downloaded from
https://github.com/wineslab/colosseum-oran-coloran-dataset):

    rome_static_medium/
        sched{0,1,2}/
            tr{0..27}/
                exp{N}/
                    bs{1..7}/
                        bs{N}.csv                        # BS-level aggregate
                        ue{M}.csv                        # Per-UE measurements
                        slices_bs{N}/
                            {IMSI}_metrics.csv           # Per-UE per-slice KPIs

The slice-metrics file is the richest signal source and is the primary input
for our feature pipeline.
"""
from .parsers import parse_slice_metrics, parse_bs_csv, parse_ue_csv
from .inventory import scan_inventory, InventoryEntry
from .merge import build_unified_parquet

__all__ = [
    "parse_slice_metrics", "parse_bs_csv", "parse_ue_csv",
    "scan_inventory", "InventoryEntry", "build_unified_parquet",
]
