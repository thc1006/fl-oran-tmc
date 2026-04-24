"""TDD test suite for v3 pipeline (fixes bs_id scaler bug + persistence residual + classification).

These tests are written BEFORE the implementation exists and will fail initially.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch


# ============================================================================
# Task 19: categorical feature encoder
# ============================================================================

def test_feature_schema_separates_cat_and_cont():
    from fl_oran.data_v2.encoders import FeatureSchema
    sch = FeatureSchema(
        categorical=["bs_id", "slice_id", "sched", "tr"],
        categorical_sizes={"bs_id": 8, "slice_id": 3, "sched": 3, "tr": 28},
        continuous=["num_ues", "tx_brate_dl_Mbps", "dl_cqi"],
    )
    assert sch.n_categorical == 4
    assert sch.n_continuous == 3
    assert sch.categorical_sizes["bs_id"] == 8


def test_scaler_bug_is_fixed_no_blowup_on_val():
    """The bug we found: per-client std of constant bs_id is 1e-6 → val values ±10^6.

    With the new schema-based approach, categorical features are NOT scaled —
    they pass through as integer indices for embedding layers.
    """
    from fl_oran.data_v2.encoders import FeatureSchema, apply_continuous_scaler, fit_continuous_scaler
    sch = FeatureSchema(
        categorical=["bs_id"],
        categorical_sizes={"bs_id": 8},
        continuous=["num_ues"],
    )

    # Simulate 7 clients each with constant bs_id
    rng = np.random.default_rng(0)
    client_data = {
        cid: np.column_stack([
            np.full(1000, cid, dtype=np.float32),           # bs_id (constant within client)
            rng.uniform(0, 20, 1000).astype(np.float32),    # num_ues (varies)
        ])
        for cid in range(1, 8)
    }
    scaler = fit_continuous_scaler(client_data, sch)
    # Val data with all bs_ids
    val_X = np.column_stack([
        np.arange(1, 8, dtype=np.float32),
        rng.uniform(0, 20, 7).astype(np.float32),
    ])
    val_cat, val_cont = apply_continuous_scaler(val_X, sch, scaler)
    # Categorical column comes through untouched
    assert val_cat.dtype in (np.int64, np.int32)
    np.testing.assert_array_equal(val_cat[:, 0], np.arange(1, 8))
    # Continuous column scaled to reasonable range
    assert np.abs(val_cont).max() < 10, f"continuous scaler value explosion: {val_cont.max()}"


# ============================================================================
# Task 20: ForecasterV2 with embeddings + (optional) persistence residual
# ============================================================================

def test_forecaster_classification_shape():
    from fl_oran.data_v2.encoders import FeatureSchema
    from fl_oran.models.forecaster_v2 import ForecasterV2
    sch = FeatureSchema(
        categorical=["bs_id", "slice_id"],
        categorical_sizes={"bs_id": 8, "slice_id": 3},
        continuous=["a", "b", "c"],
    )
    model = ForecasterV2(schema=sch, task="classification", seq_len=5)
    B = 16
    x_cat = torch.zeros(B, 5, 2, dtype=torch.long)   # (B, L, n_cat)
    x_cont = torch.randn(B, 5, 3)                    # (B, L, n_cont)
    logits = model(x_cat, x_cont)
    assert logits.shape == (B, 1)


def test_forecaster_regression_with_persistence_residual():
    """When LSTM + head weights are zero, regression output must equal persistence."""
    from fl_oran.data_v2.encoders import FeatureSchema
    from fl_oran.models.forecaster_v2 import ForecasterV2
    sch = FeatureSchema(
        categorical=["bs_id"],
        categorical_sizes={"bs_id": 8},
        continuous=["tx_brate_dl_Mbps"],  # this is the persistence reference
    )
    model = ForecasterV2(
        schema=sch, task="regression", seq_len=5,
        persistence_feature="tx_brate_dl_Mbps",
    )
    # Zero out LSTM + head weights so the delta contribution = 0
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.zero_()
    B = 8
    x_cat = torch.zeros(B, 5, 1, dtype=torch.long)
    x_cont = torch.randn(B, 5, 1)
    pred = model(x_cat, x_cont)
    # Persistence = last-step value of the reference feature
    expected = x_cont[:, -1, 0:1]
    torch.testing.assert_close(pred, expected, rtol=1e-5, atol=1e-5)


def test_forecaster_embeddings_actually_differentiate():
    """Different categorical values should produce different embeddings (after init)."""
    from fl_oran.data_v2.encoders import FeatureSchema
    from fl_oran.models.forecaster_v2 import ForecasterV2
    torch.manual_seed(0)
    sch = FeatureSchema(
        categorical=["bs_id"], categorical_sizes={"bs_id": 8}, continuous=["a"],
    )
    model = ForecasterV2(schema=sch, task="classification", seq_len=3)
    model.eval()
    x_cont = torch.zeros(2, 3, 1)
    x_cat_1 = torch.full((2, 3, 1), 1, dtype=torch.long)  # bs_id=1
    x_cat_7 = torch.full((2, 3, 1), 7, dtype=torch.long)  # bs_id=7
    with torch.no_grad():
        logit_1 = model(x_cat_1, x_cont)
        logit_7 = model(x_cat_7, x_cont)
    assert not torch.allclose(logit_1, logit_7, atol=1e-3), "embeddings produced identical output"


# ============================================================================
# Task 21: classification target builder
# ============================================================================

def test_classification_target_is_future_shifted():
    from fl_oran.data_v2.targets_v2 import add_classification_target

    df = pd.DataFrame({
        "run_id": ["r"] * 6,
        "slice_id": [0] * 6,
        "step_idx": list(range(6)),
        "ul_bler": [0.05, 0.12, 0.03, 0.20, 0.08, 0.15],
    })
    out = add_classification_target(df, column="ul_bler", threshold=0.10,
                                    target_name="y_sla_next")
    # Row i's target = (ul_bler at i+1) > 0.10
    # Row 0: ul_bler[1]=0.12 > 0.10 → 1
    # Row 1: ul_bler[2]=0.03 → 0
    # Row 2: ul_bler[3]=0.20 → 1
    # Row 3: ul_bler[4]=0.08 → 0
    # Row 4: ul_bler[5]=0.15 → 1
    # Row 5: NaN → dropped
    expected = [1, 0, 1, 0, 1]
    assert list(out["y_sla_next"].astype(int)) == expected
    assert len(out) == 5


def test_classification_target_respects_group_boundaries():
    from fl_oran.data_v2.targets_v2 import add_classification_target
    df = pd.DataFrame({
        "run_id": ["r1"] * 3 + ["r2"] * 3,
        "slice_id": [0, 0, 0, 0, 0, 0],
        "step_idx": [0, 1, 2, 0, 1, 2],
        "ul_bler": [0.5, 0.5, 0.5, 0.05, 0.05, 0.05],
    })
    out = add_classification_target(df, column="ul_bler", threshold=0.10,
                                    target_name="y_sla_next")
    # r1 rows 0,1 → label from r1 row 1,2 → [1,1]. r1 row 2 → NaN → dropped.
    # r2 rows 0,1 → label from r2 row 1,2 → [0,0]. r2 row 2 → NaN → dropped.
    # Result: 4 rows total, labels [1,1,0,0]
    assert len(out) == 4
    assert list(out["y_sla_next"].astype(int)) == [1, 1, 0, 0]


# ============================================================================
# Task 22: step-capped client trainer
# ============================================================================

def test_step_capped_trainer_respects_step_cap():
    from fl_oran.federated.client_v2 import train_one_client_capped
    torch.manual_seed(0)
    # Tiny synthetic data
    n, d = 1000, 4
    X = torch.randn(n, d)
    y = torch.randn(n, 1)
    model = torch.nn.Linear(d, 1)

    # With max_steps=50 and batch_size=10, we should do exactly 50 grad steps.
    counter = {"n": 0}
    orig_forward = model.forward
    def counting_forward(*a, **k):
        counter["n"] += 1
        return orig_forward(*a, **k)
    model.forward = counting_forward

    update = train_one_client_capped(
        client_id=0, model=model, X=X, y=y,
        loss_fn=torch.nn.MSELoss(), device=torch.device("cpu"),
        lr=1e-3, max_steps=50, batch_size=10,
        amp_enabled=False, amp_dtype=None,
    )
    assert counter["n"] == 50, f"expected 50 forward calls, got {counter['n']}"
    assert update.num_examples == 50 * 10


def test_step_capped_trainer_reduces_loss():
    from fl_oran.federated.client_v2 import train_one_client_capped
    torch.manual_seed(0)
    n, d = 500, 4
    X = torch.randn(n, d)
    w = torch.randn(d, 1) * 0.3
    y = X @ w + torch.randn(n, 1) * 0.01
    model = torch.nn.Linear(d, 1)
    before = torch.nn.functional.mse_loss(model(X), y).item()
    upd = train_one_client_capped(
        0, model, X, y, torch.nn.MSELoss(), torch.device("cpu"),
        lr=1e-2, max_steps=200, batch_size=32,
        amp_enabled=False, amp_dtype=None,
    )
    assert upd.train_loss < before, (upd.train_loss, before)
