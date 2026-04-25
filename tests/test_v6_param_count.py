"""Parameter-count parity test across the three Stage 1 architectures.

Per ADR-001 D-20: each architecture's hidden dimensions are tuned so the
total trainable parameter count matches ``ForecasterV2`` within ±10%.
This prevents capacity from confounding the energy comparison — a Spiking
model that is 2× smaller than the LSTM baseline could "win" on energy
purely by being a smaller network.

The schema used here mirrors the ColO-RAN v5 production schema (4
categorical features + 12 continuous), so any drift in ForecasterV2's
param count due to its own evolution will be tracked as well.
"""
from __future__ import annotations

import torch

from fl_oran.data_v2.encoders import FeatureSchema
from fl_oran.models.forecaster_v2 import ForecasterV2
from fl_oran.models.mamba_forecaster import MambaForecaster
from fl_oran.models.spiking_forecaster import SpikingForecaster


COLORAN_SCHEMA = FeatureSchema(
    categorical=["bs_id", "slice_id", "sched", "tr"],
    categorical_sizes={"bs_id": 7, "slice_id": 3, "sched": 5, "tr": 28},
    continuous=[f"c{i}" for i in range(12)],
)


def _count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_three_archs_within_10pct_param_count_parity():
    f = ForecasterV2(schema=COLORAN_SCHEMA, task="classification", seq_len=5)
    m = MambaForecaster(schema=COLORAN_SCHEMA, task="classification", seq_len=5)
    s = SpikingForecaster(schema=COLORAN_SCHEMA, task="classification", seq_len=5)
    n_f, n_m, n_s = _count(f), _count(m), _count(s)

    assert abs(n_m / n_f - 1.0) <= 0.10, (
        f"MambaForecaster has {n_m} params vs ForecasterV2 {n_f} "
        f"(drift {(n_m / n_f - 1) * 100:+.1f}%). Tune backbone_expand or n_blocks."
    )
    assert abs(n_s / n_f - 1.0) <= 0.10, (
        f"SpikingForecaster has {n_s} params vs ForecasterV2 {n_f} "
        f"(drift {(n_s / n_f - 1) * 100:+.1f}%). Tune backbone_d_model or n_blocks."
    )


def test_param_counts_are_in_v6_expected_range():
    """ForecasterV2 baseline param count should remain in [40K, 50K] for
    the production schema; if it drifts outside this range, all three
    archs need to be rebudgeted."""
    f = ForecasterV2(schema=COLORAN_SCHEMA, task="classification", seq_len=5)
    n_f = _count(f)
    assert 40_000 <= n_f <= 50_000, (
        f"ForecasterV2 param count {n_f} is outside the expected [40K, 50K] "
        f"range; if this is intentional, update the test bounds."
    )
