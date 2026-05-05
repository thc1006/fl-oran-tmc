"""Audit invariants — regression tests guarding the Phase-0 audit findings.

These tests encode the bug status documented in `artifacts/audit/`:
- ul_bler is in 17 continuous features (V3_CONTINUOUS) — needed for the
  next-step SLA-violation forecasting target derivation.
- tr embedding rows for tr ∈ {22..29} stay at random init because
  PyTorch nn.Embedding only updates indexed rows, and train uses
  tr ∈ {0..21} only.

If any of these change (someone adds tr remapping, expands V3_CONTINUOUS,
or shrinks V3_CAT_SIZES['tr']), the test fails and the audit must be
re-run to update conclusions.
"""
from __future__ import annotations

import numpy as np
import pytest


def test_v3_continuous_has_17_features() -> None:
    """Paper §3 claim: 17 continuous features. Regression-guard the count."""
    from fl_oran.training.centralized_v3 import V3_CONTINUOUS
    assert len(V3_CONTINUOUS) == 17, (
        f"V3_CONTINUOUS has {len(V3_CONTINUOUS)} features, paper §3 claims 17. "
        f"Either update paper or restore the original feature set."
    )


def test_ul_bler_is_in_v3_continuous() -> None:
    """ul_bler must be in input features (it's the target's autoregressive
    source). If removed, the next-step SLA-violation prediction loses its
    canonical naive-baseline (last-BLER persistence) reference."""
    from fl_oran.training.centralized_v3 import V3_CONTINUOUS
    assert "ul_bler" in V3_CONTINUOUS, (
        "ul_bler must be in V3_CONTINUOUS — it is the autoregressive source "
        "for the y_sla_violation_next target (features.py L89-91)."
    )


def test_tr_categorical_size_matches_embedding_layout() -> None:
    """V3_CAT_SIZES['tr'] determines nn.Embedding(size+1, k). If this number
    changes, the tr-embedding-rows-22-29-stay-at-init bug audit must be re-run."""
    from fl_oran.training.centralized_v3 import V3_CAT_SIZES
    assert V3_CAT_SIZES["tr"] == 29, (
        f"V3_CAT_SIZES['tr']={V3_CAT_SIZES['tr']}, expected 29 "
        f"(per audit on 2026-05-05). If you've changed the tr vocab, "
        f"re-run artifacts/audit/tr_embedding_audit.md decisive test."
    )


def test_train_tr_range_is_0_to_21() -> None:
    """Default train_tr split must be [0..21] = 22 configs. If this changes,
    the embedding-bug analysis (rows 22-29 untrained) must be re-derived."""
    from fl_oran.data_v2.split import ood_split_by_tr
    import inspect
    sig = inspect.signature(ood_split_by_tr)
    train_tr_default = sig.parameters["train_tr"].default
    assert tuple(train_tr_default) == tuple(range(22)), (
        f"train_tr default = {train_tr_default}, expected tuple(range(22)). "
        f"Bug analysis assumes train uses tr ∈ {{0..21}}; if this changes, "
        f"re-run artifacts/audit/tr_embedding_audit.md."
    )


@pytest.mark.skipif(
    not __import__("pathlib").Path(
        "artifacts/v7_stage2_full/v7_lstm_fedavg_iid_n7_s0/best.pt"
    ).exists(),
    reason="Phase 5 LSTM seed-0 IID checkpoint not present; skip empirical check.",
)
def test_tr_embedding_rows_22_to_29_byte_identical_to_fresh_init() -> None:
    """Decisive empirical test: rows 22-29 of trained tr embedding must be
    bit-identical (modulo float32 round-off) to a fresh seed-0 init.

    If this fails, EITHER (a) the bug has been fixed (great — update audit
    docs and remove this regression test), OR (b) some new code path is
    inadvertently updating the unused rows (investigate, may have introduced
    a new bug)."""
    import torch
    from fl_oran.utils.seed import seed_everything
    from fl_oran.data_v2.encoders import FeatureSchema
    from fl_oran.training.centralized_v3 import (
        V3_CATEGORICAL,
        V3_CAT_SIZES,
        V3_CONTINUOUS,
    )
    from fl_oran.models.forecaster_v2 import ForecasterV2

    seed_everything(0, deterministic=True)
    schema = FeatureSchema(
        categorical=V3_CATEGORICAL,
        categorical_sizes=V3_CAT_SIZES,
        continuous=V3_CONTINUOUS,
    )
    fresh = ForecasterV2(schema=schema, task="classification", seq_len=5)
    fresh_tr = fresh.embeddings["tr"].weight.detach().numpy()

    trained = torch.load(
        "artifacts/v7_stage2_full/v7_lstm_fedavg_iid_n7_s0/best.pt",
        map_location="cpu",
        weights_only=False,
    )
    trained_tr = trained["_orig_mod.embeddings.tr.weight"].numpy()

    deltas = np.linalg.norm(trained_tr - fresh_tr, axis=1)

    # TRAIN rows 0-21 should have substantial drift
    assert deltas[0:22].mean() > 0.1, (
        f"TRAIN rows mean Δ = {deltas[0:22].mean():.4e}, expected > 0.1; "
        f"checkpoint may not be the trained model."
    )
    # VAL/TEST/UNUSED rows 22-29 should be bit-identical (delta < 1e-5
    # accounts for float32 round-off across save/load roundtrip)
    assert deltas[22:30].max() < 1e-5, (
        f"Rows 22-29 max Δ = {deltas[22:30].max():.4e} > 1e-5; "
        f"the audit hypothesis (rows 22-29 bit-identical to fresh init) "
        f"is falsified. Either bug is fixed or new update path was added; "
        f"re-run artifacts/audit/tr_embedding_audit.md decisive test."
    )
