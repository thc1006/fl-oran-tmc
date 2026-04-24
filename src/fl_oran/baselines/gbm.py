"""Gradient-boosting baseline on tabular features (no time sequences).

GBM has no knowledge of history beyond hand-engineered trend features, but is
often a hard baseline to beat for structured-data forecasting.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score


def flatten_sequences(X: np.ndarray, mode: Literal["last_step", "flatten"] = "last_step") -> np.ndarray:
    """Convert (N, L, F) sequence input to 2D input suitable for GBM.

    - ``last_step``: keep only the final timestep → (N, F). Fairest comparison
      for tasks where trend features already encode short-history signals.
    - ``flatten``: flatten full window → (N, L*F). Gives GBM access to full
      history at the cost of more features.
    """
    if X.ndim == 2:
        return X
    if X.ndim != 3:
        raise ValueError(f"expected 2D or 3D input, got shape {X.shape}")
    if mode == "last_step":
        return X[:, -1, :]
    if mode == "flatten":
        return X.reshape(X.shape[0], -1)
    raise ValueError(f"unknown mode: {mode}")


def gbm_baseline(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    *,
    task: Literal["regression", "classification"] = "regression",
    seq_mode: Literal["last_step", "flatten"] = "last_step",
    n_estimators: int = 100,
    max_depth: int = 5,
    random_state: int = 42,
) -> dict:
    """Train a sklearn GBM and return test-set metrics.

    Accepts either 2D (N, F) or 3D (N, L, F) X arrays; 3D inputs are collapsed
    via ``flatten_sequences(X, seq_mode)``.
    """
    X_train = flatten_sequences(X_train, seq_mode)
    X_test = flatten_sequences(X_test, seq_mode)

    if task == "regression":
        model = GradientBoostingRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )
        model.fit(X_train, y_train.ravel())
        pred = model.predict(X_test)
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "mae": float(mean_absolute_error(y_test, pred)),
            "r2": float(r2_score(y_test, pred)),
        }
    if task == "classification":
        model = GradientBoostingClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )
        model.fit(X_train, y_train.ravel().astype(int))
        pred = model.predict(X_test)
        return {"accuracy": float(accuracy_score(y_test, pred))}
    raise ValueError(f"unknown task: {task}")
