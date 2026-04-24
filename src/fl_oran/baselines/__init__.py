"""Non-FL baselines to contextualise the federated-LSTM results."""
from .persistence import PersistenceBaseline, persistence_forecast, persistence_metrics
from .gbm import gbm_baseline, flatten_sequences

__all__ = [
    "PersistenceBaseline",
    "persistence_forecast",
    "persistence_metrics",
    "gbm_baseline",
    "flatten_sequences",
]
