"""Persistence baseline: ŷ_{t+1} = y_t.

For a forecasting task, this is the null model. If the FL / LSTM does not
meaningfully beat persistence, there's no learning happening — only copying.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PersistenceBaseline:
    """The trivial 'copy-the-present' predictor."""

    @staticmethod
    def predict(y_current: np.ndarray) -> np.ndarray:
        """Return y_current unchanged — the "copy" predictor."""
        return y_current.copy()

    @classmethod
    def evaluate(cls, y_current: np.ndarray, y_next: np.ndarray) -> dict:
        """Compute regression metrics comparing persistence prediction to truth."""
        pred = cls.predict(y_current)
        err = y_next - pred
        mse = float(np.mean(err * err))
        mae = float(np.mean(np.abs(err)))
        ss_tot = float(np.sum((y_next - y_next.mean()) ** 2))
        ss_res = float(np.sum(err * err))
        r2 = 1 - ss_res / max(ss_tot, 1e-12)
        return {"rmse": float(np.sqrt(mse)), "mae": mae, "r2": r2}


# Backward-compatible functional API.
def persistence_forecast(y_current: np.ndarray) -> np.ndarray:
    return PersistenceBaseline.predict(y_current)


def persistence_metrics(y_current: np.ndarray, y_next: np.ndarray) -> dict:
    return PersistenceBaseline.evaluate(y_current, y_next)
