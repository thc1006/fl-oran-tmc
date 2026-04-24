"""Feature schema + scaler that respects categorical vs continuous distinction.

Replaces the broken per-client scaler from trainer_v2 (which was blowing up
bs_id values to ±10^6 on val/test).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ..logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class FeatureSchema:
    """Which columns are categorical (→ nn.Embedding) vs continuous (→ StandardScaler)."""
    categorical: list[str]
    categorical_sizes: dict[str, int]
    continuous: list[str]

    @property
    def n_categorical(self) -> int:
        return len(self.categorical)

    @property
    def n_continuous(self) -> int:
        return len(self.continuous)

    def split_array(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Split a (..., n_cat+n_cont) array into (cat, cont) halves.
        Categorical half cast to int64; continuous half stays float32.
        """
        n_cat = self.n_categorical
        cat = X[..., :n_cat].astype(np.int64)
        cont = X[..., n_cat:].astype(np.float32)
        return cat, cont


@dataclass
class ContinuousScaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.std).astype(np.float32)


# ----------------------------------------------------------------------------
# Federated scaler fitting: server never sees raw data, only sufficient stats.
# ----------------------------------------------------------------------------

@dataclass
class AggregatedStats:
    mean: np.ndarray
    std: np.ndarray
    n_total: int


def compute_client_stats(X: np.ndarray, schema: FeatureSchema) -> dict:
    """A client computes (n, sum_x, sum_x²) of its continuous columns.

    These are the minimum sufficient statistics for a global mean/std. The
    server can aggregate them without ever seeing any raw row.
    """
    n_cat = schema.n_categorical
    cont = X[..., n_cat:]
    # Reshape to 2D if sequences (..., L, F_cont) → (N*L, F_cont).
    if cont.ndim > 2:
        cont = cont.reshape(-1, cont.shape[-1])
    n = int(cont.shape[0])
    sum_x = cont.sum(axis=0, dtype=np.float64)
    sum_x2 = (cont.astype(np.float64) ** 2).sum(axis=0)
    return {"n": n, "sum_x": sum_x, "sum_x2": sum_x2}


def aggregate_client_stats(stats_list: list[dict]) -> AggregatedStats:
    """Server-side aggregation: fold many ``compute_client_stats`` dicts into
    a single global mean/std. Uses Welford-style pooled variance."""
    n_total = sum(s["n"] for s in stats_list)
    if n_total == 0:
        raise ValueError("no data across clients")
    sum_x = sum(s["sum_x"] for s in stats_list)
    sum_x2 = sum(s["sum_x2"] for s in stats_list)
    mean = (sum_x / n_total).astype(np.float32)
    # population variance (std uses 1/n, matching numpy's default)
    var = (sum_x2 / n_total - mean.astype(np.float64) ** 2)
    std = (np.sqrt(np.maximum(var, 0)) + 1e-6).astype(np.float32)
    return AggregatedStats(mean=mean, std=std, n_total=n_total)


def federated_fit_scaler(
    client_data: Mapping[int, np.ndarray],
    schema: FeatureSchema,
    n_jobs: int = 1,
) -> ContinuousScaler:
    """FL-compatible scaler fit via sufficient-stats aggregation.

    Each client emits (n, sum_x, sum_x²); server aggregates. Mathematically
    identical to pooling but the server never sees raw rows.

    ``n_jobs > 1`` parallelises ``compute_client_stats`` across clients via
    joblib threading. NumPy reductions release the GIL, so threading gives
    near-linear speedup with no copy overhead.
    """
    if n_jobs > 1 and len(client_data) > 1:
        from joblib import Parallel, delayed
        client_stats = Parallel(
            n_jobs=min(n_jobs, len(client_data)), backend="threading",
        )(delayed(compute_client_stats)(x, schema) for x in client_data.values())
    else:
        client_stats = [compute_client_stats(x, schema) for x in client_data.values()]
    agg = aggregate_client_stats(client_stats)
    log.info("federated_fit_scaler over %s rows, %d continuous features (n_jobs=%d)",
             f"{agg.n_total:,}", schema.n_continuous, n_jobs)
    return ContinuousScaler(mean=agg.mean, std=agg.std)


def fit_continuous_scaler(
    client_data: Mapping[int, np.ndarray],
    schema: FeatureSchema,
) -> ContinuousScaler:
    """Pool all clients' continuous columns and compute global z-score stats.

    Categorical columns are IGNORED here — they go through embeddings, not scaling.
    """
    n_cat = schema.n_categorical
    pooled = np.concatenate([x[..., n_cat:] for x in client_data.values()], axis=0)
    mean = pooled.mean(axis=0).astype(np.float32)
    std = (pooled.std(axis=0) + 1e-6).astype(np.float32)
    log.info("fit ContinuousScaler over %s rows, %d continuous features",
             f"{len(pooled):,}", schema.n_continuous)
    return ContinuousScaler(mean=mean, std=std)


def apply_continuous_scaler(
    X: np.ndarray,
    schema: FeatureSchema,
    scaler: ContinuousScaler,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (cat_int64, cont_scaled_float32). Categorical passes through untouched."""
    cat, cont = schema.split_array(X)
    cont = scaler.transform(cont)
    return cat, cont
