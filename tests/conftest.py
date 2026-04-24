"""Shared test fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure the source package is importable without install.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def synthetic_parquet(tmp_path_factory) -> Path:
    """Small synthetic parquet that mirrors the real schema for fast tests."""
    rng = np.random.default_rng(0)
    n = 5_000
    clients = 4
    rows = []
    for cid in range(1, clients + 1):
        for i in range(n // clients):
            row = {
                "num_ues": rng.integers(1, 10),
                "slice_id": rng.integers(0, 3),
                "sched_policy_num": rng.integers(0, 3),
                "allocated_rbgs": rng.integers(1, 14),
                "bs_id": cid,
                "exp_id": 1,
                "sum_requested_prbs": float(rng.uniform(0, 2000)),
                "sum_granted_prbs": float(rng.uniform(0, 2500)),
                "prb_utilization": float(rng.uniform(0, 1)),
                "throughput_efficiency": float(rng.uniform(0, 0.5)),
                "qos_score": float(rng.uniform(0.4, 1.0)),
                "network_load": float(rng.uniform(0.02, 0.5)),
                "hour": int(rng.integers(0, 24)),
                "minute": int(rng.integers(0, 60)),
                "day_of_week": int(rng.integers(0, 7)),
                "allocation_efficiency": float(rng.uniform(0.12, 0.66)),
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    path = tmp_path_factory.mktemp("data") / "synthetic.parquet"
    df.to_parquet(path, index=False)
    return path


@pytest.fixture
def small_dataframe(synthetic_parquet) -> pd.DataFrame:
    return pd.read_parquet(synthetic_parquet)


@pytest.fixture(autouse=True)
def _set_log_level(monkeypatch):
    monkeypatch.setenv("FL_ORAN_LOG_LEVEL", "WARNING")
