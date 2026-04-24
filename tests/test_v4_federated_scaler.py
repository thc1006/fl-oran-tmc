"""TDD for v4: federated_fit_scaler — uses sufficient stats aggregation.

The server never sees raw data. Each client computes local (sum_x, sum_x², n),
server aggregates into global mean/std. Mathematically equivalent to pooled
z-score fit, but preserves FL privacy semantics.
"""
from __future__ import annotations

import numpy as np
import pytest


def test_federated_scaler_equals_pooled_scaler():
    """Federated aggregation must produce IDENTICAL stats to pooling."""
    from fl_oran.data_v2.encoders import FeatureSchema, federated_fit_scaler, fit_continuous_scaler

    rng = np.random.default_rng(0)
    sch = FeatureSchema(
        categorical=["bs_id"], categorical_sizes={"bs_id": 8},
        continuous=["a", "b", "c"],
    )
    # 3 clients with different data
    client_data = {
        1: rng.normal(0, 1, (500, 1 + 3)).astype(np.float32),
        2: rng.normal(5, 2, (700, 1 + 3)).astype(np.float32),
        3: rng.normal(-2, 3, (300, 1 + 3)).astype(np.float32),
    }
    for cid in client_data:
        client_data[cid][:, 0] = cid  # bs_id

    pooled = fit_continuous_scaler(client_data, sch)
    fed = federated_fit_scaler(client_data, sch)
    np.testing.assert_allclose(fed.mean, pooled.mean, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(fed.std, pooled.std, rtol=1e-5, atol=1e-6)


def test_federated_scaler_server_sees_only_sufficient_stats():
    """Verify API: the server-side aggregator takes only (sum_x, sum_x², n) tuples,
    never the raw data.
    """
    from fl_oran.data_v2.encoders import (
        FeatureSchema, aggregate_client_stats, compute_client_stats,
    )
    sch = FeatureSchema(
        categorical=["bs_id"], categorical_sizes={"bs_id": 8},
        continuous=["a", "b"],
    )
    X = np.arange(20, dtype=np.float32).reshape(10, 2)  # only continuous (bs_id handled elsewhere)
    # Stack a fake categorical column for realism
    X_full = np.column_stack([np.ones(10, dtype=np.float32), X])  # (10, 3)
    stats = compute_client_stats(X_full, sch)
    assert stats["n"] == 10
    assert stats["sum_x"].shape == (2,)
    assert stats["sum_x2"].shape == (2,)
    # Known values: column 0 = 0,2,4,...,18  sum=90  sum_sq=5×(0²+2²+...+18²)=2280
    # Actually 0+2+4+6+8+10+12+14+16+18 = 90 ✓
    np.testing.assert_allclose(stats["sum_x"][0], 90, atol=1e-4)

    # Aggregate two identical clients → mean same, n doubles
    agg = aggregate_client_stats([stats, stats])
    assert agg.n_total == 20
    expected_mean = 90 / 10  # same as single client
    np.testing.assert_allclose(agg.mean[0], expected_mean, atol=1e-4)


def test_federated_scaler_single_client_degenerate():
    """A single client's federated fit must equal its own pooled fit."""
    from fl_oran.data_v2.encoders import FeatureSchema, federated_fit_scaler, fit_continuous_scaler
    rng = np.random.default_rng(0)
    sch = FeatureSchema(
        categorical=[], categorical_sizes={}, continuous=["x", "y"],
    )
    data = {1: rng.normal(0, 1, (1000, 2)).astype(np.float32)}
    p = fit_continuous_scaler(data, sch)
    f = federated_fit_scaler(data, sch)
    np.testing.assert_allclose(p.mean, f.mean, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(p.std, f.std, rtol=1e-5, atol=1e-6)
