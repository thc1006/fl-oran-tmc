"""End-to-end smoke test that the three Stage 1 architectures train.

Per ADR-001 D-20 / TDD plan: 100 gradient steps on a 256-row toy dataset,
final loss strictly below initial, no NaN, runs under 60 s on RTX 4080
(CPU is also acceptable for this smoke). Mirrors the v5 baseline test
``tests/test_v5_end_to_end_smoke.py``.

The smoke does **not** hit the real ColO-RAN parquet — it constructs
synthetic tensors with the same shape contract as the production
pipeline. A larger smoke against the real loader is part of S1-W2.
"""
from __future__ import annotations

import time

import pytest
import torch
import torch.nn as nn

from fl_oran.data_v2.encoders import FeatureSchema
from fl_oran.models.forecaster_v2 import ForecasterV2
from fl_oran.models.mamba_forecaster import MambaForecaster
from fl_oran.models.spiking_forecaster import SpikingForecaster


_SCHEMA = FeatureSchema(
    categorical=["bs_id", "slice_id"],
    categorical_sizes={"bs_id": 7, "slice_id": 3},
    continuous=["dl_throughput_mbps", "ul_throughput_mbps", "prb_util"],
)


def _make_synthetic_dataset(n_rows: int = 256, seq_len: int = 5):
    torch.manual_seed(0)
    sizes = list(_SCHEMA.categorical_sizes.values())
    x_cat = torch.stack(
        [torch.randint(0, sz + 1, (n_rows, seq_len)) for sz in sizes],
        dim=-1,
    ).long()
    x_cont = torch.randn(n_rows, seq_len, _SCHEMA.n_continuous)
    # Make labels weakly predictable from continuous features (so loss can drop).
    signal = x_cont[:, -1, 0] - x_cont[:, -1, 1]
    y = (signal > 0).float()
    return x_cat, x_cont, y


def _train_n_steps(model, x_cat, x_cont, y, n_steps: int = 100, lr: float = 5e-4):
    """Plain Adam loop on the full toy batch."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    losses = []
    for step in range(n_steps):
        opt.zero_grad()
        logits = model(x_cat, x_cont).squeeze(-1)
        loss = bce(logits, y)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    return losses


@pytest.mark.parametrize(
    "arch_name, ctor, lr",
    [
        ("ForecasterV2", lambda: ForecasterV2(schema=_SCHEMA, task="classification", seq_len=5), 5e-4),
        ("MambaForecaster", lambda: MambaForecaster(schema=_SCHEMA, task="classification", seq_len=5), 5e-4),
        ("SpikingForecaster", lambda: SpikingForecaster(schema=_SCHEMA, task="classification", seq_len=5), 1e-4),
    ],
)
def test_arch_smoke_loss_decreases_no_nan(arch_name, ctor, lr):
    torch.manual_seed(7)
    x_cat, x_cont, y = _make_synthetic_dataset()
    model = ctor()

    t0 = time.perf_counter()
    losses = _train_n_steps(model, x_cat, x_cont, y, n_steps=100, lr=lr)
    elapsed = time.perf_counter() - t0

    initial = sum(losses[:5]) / 5  # average of first 5 steps
    final = sum(losses[-5:]) / 5  # average of last 5 steps
    assert all(torch.isfinite(torch.tensor(loss)) for loss in losses), (
        f"{arch_name} produced non-finite loss"
    )
    assert final < initial, (
        f"{arch_name} did not learn: initial mean loss {initial:.4f} → "
        f"final mean loss {final:.4f}"
    )
    # 60 s budget is generous; on RTX 4080 even SpikingForecaster should be < 30 s.
    assert elapsed < 60.0, f"{arch_name} took {elapsed:.1f} s (> 60 s budget)"
