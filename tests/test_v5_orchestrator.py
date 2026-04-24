"""Integration tests for the v5 sweep orchestrator.

Tests exercise the full end-to-end pipeline on a tiny synthetic parquet:
data load -> OOD split -> Dirichlet partition -> scaler fit -> round loop
-> test eval -> artifact emission. Two algorithms are smoke-tested
(FedAvg and MOON) to verify the registry dispatch and the MOON-specific
encode_fn auto-injection both work end-to-end.

Runs on CPU with tiny dims (2 rounds, 2 clients, 3 max_steps, batch=4) so
each test completes in well under a second.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from fl_oran.data_v2.encoders import FeatureSchema
from fl_oran.models.forecaster_v2 import ForecasterV2
from fl_oran.training.centralized_v3 import (
    V3_CATEGORICAL,
    V3_CAT_SIZES,
    V3_CONTINUOUS,
)
from fl_oran.training.fl_v5 import V5Config, forecaster_encode_fn, run_v5_sweep


# --------------------------------------------------------------------------
# Synthetic data helper.
# --------------------------------------------------------------------------


def _make_v5_df(seed: int = 0) -> pd.DataFrame:
    """Build a v3-schema DataFrame large enough for the full pipeline.

    Dimensions:
      5 ``tr`` values (0..4) — tr 0..2 train, tr 3 val, tr 4 test.
      For each ``tr``: 3 run_ids, 3 slice_ids, 10 step_idx per group.
      => 5 * 3 * 3 * 10 = 450 rows.
      After classification target drops the last step per (run, slice):
      450 - 45 = 405 rows; seq_len=2 windows yield ~9 windows per group.
    """
    rng = np.random.default_rng(seed)
    cont_bounds = {
        "num_ues": (1, 10), "slice_prb": (0.0, 100.0),
        "sum_requested_prbs": (0.0, 2000.0), "sum_granted_prbs": (0.0, 2500.0),
        "tx_brate_dl_Mbps": (0.0, 50.0), "rx_brate_ul_Mbps": (0.0, 30.0),
        "tx_pkts_dl": (0, 5000), "rx_pkts_ul": (0, 5000),
        "dl_buffer_bytes": (0, 1e6), "ul_buffer_bytes": (0, 1e6),
        "dl_bler": (0.0, 0.3), "ul_bler": (0.0, 0.3),
        "dl_mcs": (0, 28), "ul_mcs": (0, 28),
        "dl_cqi": (0, 15), "ul_sinr": (-5.0, 30.0), "ul_rssi": (-100.0, -40.0),
    }
    rows: list[dict] = []
    for tr in range(5):
        for run_idx in range(3):
            run_id = tr * 100 + run_idx  # globally unique
            for slice_id in range(3):
                for step in range(10):
                    row: dict = {
                        "tr": tr,
                        "run_id": run_id,
                        "step_idx": step,
                        "slice_id": slice_id,
                        "bs_id": 1 + (run_idx % 7),
                        "sched": run_idx % 3,
                    }
                    for col, (lo, hi) in cont_bounds.items():
                        row[col] = float(rng.uniform(lo, hi))
                    rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def v5_parquet(tmp_path: Path) -> Path:
    df = _make_v5_df(seed=0)
    path = tmp_path / "v5_synthetic.parquet"
    df.to_parquet(path, index=False)
    return path


@pytest.fixture
def _tiny_v5_config(v5_parquet, tmp_path) -> V5Config:
    """Small, fast config used by both smoke tests."""
    return V5Config(
        algorithm="fedavg",
        alpha=0.5,
        n_clients=2,
        num_rounds=2,
        clients_per_round=2,
        max_steps_per_round=3,
        batch_size=4,
        lr=1e-3,
        lr_warmup_rounds=1,
        unified_parquet=v5_parquet,
        sample_ratio=1.0,
        seq_len=2,
        train_tr=[0, 1, 2],
        val_tr=[3],
        test_tr=[4],
        seed=7,
        device="cpu",
        mixed_precision="off",
        output_dir=tmp_path / "v5_out",
    )


# --------------------------------------------------------------------------
# Unit test: forecaster_encode_fn.
# --------------------------------------------------------------------------


def test_forecaster_encode_fn_shape_and_gradient():
    """encode_fn returns (B, fc_hidden) and gradient flows back to fc.weight."""
    schema = FeatureSchema(
        categorical=V3_CATEGORICAL,
        categorical_sizes=V3_CAT_SIZES,
        continuous=V3_CONTINUOUS,
    )
    model = ForecasterV2(schema=schema, task="classification", seq_len=2)
    B = 4
    x_cat = torch.stack([
        torch.randint(0, V3_CAT_SIZES[c], (B, 2)) for c in schema.categorical
    ], dim=-1)  # (B, L, n_cat)
    x_cont = torch.randn(B, 2, len(V3_CONTINUOUS))

    z = forecaster_encode_fn(model, x_cat, x_cont)
    assert z.shape == (B, 64), f"expected (B, 64), got {tuple(z.shape)}"

    # Gradient sanity: a scalar on z should produce a nonzero grad on fc.weight.
    model.zero_grad()
    z.sum().backward()
    assert model.fc.weight.grad is not None
    assert float(model.fc.weight.grad.abs().sum()) > 0, \
        "fc.weight should receive gradient via encode_fn"


# --------------------------------------------------------------------------
# Integration: end-to-end sweep for FedAvg and MOON.
# --------------------------------------------------------------------------


def test_run_v5_sweep_fedavg_smoke(_tiny_v5_config):
    """FedAvg path runs end-to-end and emits the expected artifacts."""
    result = run_v5_sweep(_tiny_v5_config)

    # Result shape sanity.
    assert "config" in result
    assert "history" in result
    assert "test" in result
    assert len(result["history"]) == _tiny_v5_config.num_rounds
    assert {"accuracy", "f1"} <= set(result["test"].keys())
    # Per-round records have the fields the outer sweep driver expects.
    for row in result["history"]:
        assert {"round", "train_loss", "val_auc", "val_acc", "lr"} <= set(row.keys())

    # Artifacts landed in the output dir.
    run_dir = _tiny_v5_config.output_dir / _tiny_v5_config.name
    assert (run_dir / "logs" / "summary.json").exists()
    assert (run_dir / "logs" / "history.csv").exists()
    saved = json.loads((run_dir / "logs" / "summary.json").read_text())
    assert saved["config"]["algorithm"] == "fedavg"


def test_run_v5_sweep_moon_smoke(_tiny_v5_config):
    """MOON path runs end-to-end with encode_fn auto-injected by the orchestrator."""
    cfg = _tiny_v5_config
    cfg.algorithm = "moon"
    cfg.algo_kwargs = {"mu": 0.1, "tau": 0.5}
    # Regenerate the auto-name so artifacts don't collide with the FedAvg run.
    cfg.name = ""
    cfg.__post_init__()

    result = run_v5_sweep(cfg)
    assert result["config"]["algorithm"] == "moon"
    assert len(result["history"]) == cfg.num_rounds
    # MOON's algo_kwargs got through the registry dispatch.
    assert result["config"]["algo_kwargs"]["mu"] == 0.1
    assert result["config"]["algo_kwargs"]["tau"] == 0.5
