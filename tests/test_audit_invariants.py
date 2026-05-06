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


def test_test_tr_range_is_25_to_27() -> None:
    """Default test_tr split must be [25, 26, 27] = 3 configs. The audit
    claim that 'at test time the model receives random-init vectors for
    tr ∈ {25..27}' depends on this exact range."""
    from fl_oran.data_v2.split import ood_split_by_tr
    import inspect
    sig = inspect.signature(ood_split_by_tr)
    test_tr_default = sig.parameters["test_tr"].default
    assert tuple(test_tr_default) == (25, 26, 27), (
        f"test_tr default = {test_tr_default}, expected (25, 26, 27). "
        f"If test split changed, re-derive which embedding rows are "
        f"random-init at test time."
    )


def test_plot_algo_ranking_uses_test_auc_not_best_val() -> None:
    """Per P1.4b-GREEN (reviewer Minor#2), Figure 1 must use test_auc_mean
    not auc_mean (which is best_val_auc per the aggregator). Regression
    guard: if someone reverts plot_algo_ranking() to best_val, this test
    fails and the GREEN flip in test_paper_language_invariants becomes
    inconsistent with the figure."""
    import inspect
    import re
    from pathlib import Path
    script = Path(__file__).resolve().parent.parent / "scripts" / "phase5_paper_figures.py"
    src = script.read_text()
    # Find plot_algo_ranking function body
    m = re.search(r"def plot_algo_ranking\b.*?(?=\ndef |\Z)", src, re.DOTALL)
    assert m, "plot_algo_ranking function not found in phase5_paper_figures.py"
    fn_body = m.group(0)
    assert 'r["test_auc_mean"]' in fn_body, (
        "plot_algo_ranking must aggregate test_auc_mean (not auc_mean / "
        "best_val) per P1.4b-GREEN. Reviewer Minor#2: headline figure "
        "must use the test metric to avoid selection-bias appearance."
    )
    assert 'r["auc_mean"]' not in fn_body, (
        "plot_algo_ranking must NOT use auc_mean (best_val) — switched to "
        "test_auc_mean per P1.4b-GREEN. Found stale auc_mean reference."
    )


def test_sla_bler_threshold_is_0_10() -> None:
    """The 10% BLER SLA threshold is paper §3.2 + Polese 2022 §V canonical
    value. The naive last-BLER persistence baseline depends on this exact
    threshold for prediction equivalence."""
    from fl_oran.data_v2.features import SLA_BLER_THRESHOLD
    assert SLA_BLER_THRESHOLD == 0.10, (
        f"SLA_BLER_THRESHOLD = {SLA_BLER_THRESHOLD}, expected 0.10 "
        f"(canonical ColO-RAN gate, Polese 2022 §V; paper §3.2 claim "
        f"30.9% positive rate is computed at this threshold)."
    )


_DECISIVE_TEST_CHECKPOINTS = {
    # arch_label → (relative path, model class import path)
    # The state-dict key for tr embedding is auto-detected: LSTM/Mamba use
    # `_orig_mod.embeddings.tr.weight` (torch.compile wrapper) while Spiking
    # uses `embeddings.tr.weight` (uncompiled or different compile path).
    "lstm": (
        "artifacts/v7_stage2_full/v7_lstm_fedavg_iid_n7_s0/best.pt",
        "fl_oran.models.forecaster_v2.ForecasterV2",
    ),
    "mamba": (
        "artifacts/v7_stage2_full/v7_mamba_fedavg_iid_n7_s0/best.pt",
        "fl_oran.models.mamba_forecaster.MambaForecaster",
    ),
    "spiking_expand2": (
        "artifacts/v7_stage2_full/v7_spiking_expand2_fedavg_iid_n7_s0/best.pt",
        "fl_oran.models.spiking_forecaster.SpikingForecaster",
    ),
}


def _resolve_tr_embedding_key(state_dict: dict) -> str | None:
    """Return whichever key holds the tr embedding weight, or None if absent.

    Handles both compile-wrapped (`_orig_mod.embeddings.tr.weight`) and
    bare (`embeddings.tr.weight`) state-dict layouts."""
    for candidate in ("_orig_mod.embeddings.tr.weight", "embeddings.tr.weight"):
        if candidate in state_dict:
            return candidate
    return None


def _build_fresh_model_and_extract_tr_embedding(
    model_class_path: str,
) -> "np.ndarray":
    """Build a seed-0 fresh model of the given class and return its
    initial tr-embedding weights as numpy. Reproduces what fl_v7's
    `_build_model` did at training time."""
    import importlib
    from fl_oran.utils.seed import seed_everything
    from fl_oran.data_v2.encoders import FeatureSchema
    from fl_oran.training.centralized_v3 import (
        V3_CATEGORICAL,
        V3_CAT_SIZES,
        V3_CONTINUOUS,
    )

    module_path, cls_name = model_class_path.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), cls_name)
    seed_everything(0, deterministic=True)
    schema = FeatureSchema(
        categorical=V3_CATEGORICAL,
        categorical_sizes=V3_CAT_SIZES,
        continuous=V3_CONTINUOUS,
    )
    model = cls(schema=schema, task="classification", seq_len=5)
    return model.embeddings["tr"].weight.detach().numpy()


@pytest.mark.parametrize("arch", list(_DECISIVE_TEST_CHECKPOINTS.keys()))
def test_tr_embedding_rows_22_to_29_byte_identical_to_fresh_init(arch: str) -> None:
    """Decisive empirical test (parameterized over 3 archs): rows 22-29 of
    trained tr embedding must be bit-identical (modulo float32 round-off)
    to fresh seed-0 init. The bug is mechanical (PyTorch nn.Embedding only
    updates indexed rows) so the same conclusion should hold across all
    3 architectures.

    If any arch passes (rows 22-29 byte-identical to init), the bug holds
    for that arch. Per-arch parameterize means a partial fix (e.g.,
    Mamba's tr embedding gets accidentally updated via a different code
    path) would surface as one arch failing while others pass.

    Test skips per-arch if the seed-0 IID checkpoint is missing (allows
    partial dev environments to still run the rest of the suite)."""
    import pathlib
    import torch

    ckpt_path, model_class_path = _DECISIVE_TEST_CHECKPOINTS[arch]
    if not pathlib.Path(ckpt_path).exists():
        pytest.skip(f"checkpoint missing: {ckpt_path}")

    fresh_tr = _build_fresh_model_and_extract_tr_embedding(model_class_path)
    trained = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd_key = _resolve_tr_embedding_key(trained)
    assert sd_key is not None, (
        f"[{arch}] No tr embedding key found in {ckpt_path}; checkpoint "
        f"layout has changed in an unexpected way. Inspect with "
        f"`torch.load(...).keys()` and update _resolve_tr_embedding_key."
    )
    trained_tr = trained[sd_key].numpy()

    deltas = np.linalg.norm(trained_tr - fresh_tr, axis=1)

    assert deltas[0:22].mean() > 0.1, (
        f"[{arch}] TRAIN rows 0-21 mean Δ = {deltas[0:22].mean():.4e}, "
        f"expected > 0.1; checkpoint may not be the trained model."
    )
    assert deltas[22:30].max() < 1e-5, (
        f"[{arch}] Rows 22-29 max Δ = {deltas[22:30].max():.4e} > 1e-5; "
        f"the audit hypothesis (rows 22-29 bit-identical to fresh init) "
        f"is falsified for {arch}. Either bug is fixed for this arch or "
        f"a new update path was added; re-derive "
        f"artifacts/audit/tr_embedding_audit.md."
    )
