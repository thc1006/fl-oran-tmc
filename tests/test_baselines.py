"""Tests for baseline predictors."""
from __future__ import annotations

import numpy as np
import pytest

from fl_oran.baselines import PersistenceBaseline, flatten_sequences, gbm_baseline


def test_persistence_copies_current_value():
    y = np.array([1.0, 2.0, 3.0])
    pred = PersistenceBaseline.predict(y)
    assert np.array_equal(pred, y)
    # must be a copy, not view
    pred[0] = 999
    assert y[0] == 1.0


def test_persistence_metrics_perfect_prediction():
    y_current = np.array([1.0, 2.0, 3.0])
    y_next = y_current.copy()  # zero drift
    m = PersistenceBaseline.evaluate(y_current, y_next)
    assert m["rmse"] == 0.0
    assert m["mae"] == 0.0
    assert m["r2"] > 0.99  # perfect


def test_persistence_metrics_worst_case():
    y_current = np.array([0.0, 0.0, 0.0])
    y_next = np.array([1.0, 1.0, 1.0])
    m = PersistenceBaseline.evaluate(y_current, y_next)
    assert m["rmse"] == 1.0
    assert m["mae"] == 1.0


def test_flatten_sequences_last_step():
    X = np.arange(24).reshape(2, 3, 4).astype(float)  # 2 samples, 3 steps, 4 feats
    flat = flatten_sequences(X, mode="last_step")
    assert flat.shape == (2, 4)
    np.testing.assert_array_equal(flat[0], X[0, -1, :])


def test_flatten_sequences_flatten_mode():
    X = np.arange(24).reshape(2, 3, 4).astype(float)
    flat = flatten_sequences(X, mode="flatten")
    assert flat.shape == (2, 12)


def test_flatten_sequences_2d_passthrough():
    X = np.arange(6).reshape(2, 3).astype(float)
    flat = flatten_sequences(X)
    assert flat.shape == X.shape


def test_gbm_regression_with_sequence_input():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 5, 3)).astype(np.float32)
    # Dependent target: sum of last-step features + noise
    y = X[:, -1, :].sum(axis=1) + rng.normal(scale=0.1, size=100)
    Xtr, Xte = X[:80], X[80:]
    ytr, yte = y[:80], y[80:]
    m = gbm_baseline(Xtr, ytr, Xte, yte, task="regression", n_estimators=20, max_depth=3)
    assert "rmse" in m and "r2" in m
    assert m["rmse"] > 0


def test_gbm_classification():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    m = gbm_baseline(X[:80], y[:80], X[80:], y[80:], task="classification",
                     n_estimators=20, max_depth=3)
    assert "accuracy" in m
    assert 0 <= m["accuracy"] <= 1


def test_gbm_rejects_bad_shape():
    X = np.zeros((4, 2, 3, 5))  # 4D
    y = np.zeros(4)
    with pytest.raises(ValueError):
        gbm_baseline(X, y, X, y, task="regression")
