"""R2 C4 — no-tr ablation methodology + spec invariants.

TDD-RED before C4 GREEN: assert the spec yaml exists with correct
parameters AND the ForecasterV2 model supports the drop_categorical
arg the spec depends on. Tests fail until both land.

Reviewer R2 #7 / A3: §7.1.6 quantifies the tr-embedding-bug confound
(LSTM 9.2 % / Mamba 10.2 % / Spiking 2.3 %) but doesn't prove the
finding survives when `tr` is removed entirely. C4 is the upper-bound
check: re-train LSTM × FedAvg × natural-by-BS × 10 seeds with
`drop_categorical=["tr"]`, compare to Phase 5 baseline.

If gap shrinks <10 %: C1 mechanism is robust to tr removal — strengthens §7.1.6.
If gap shrinks >50 %: tr is doing real work; rewrite §7.1.6.

Hardware: V100 4-way parallel (~10 min wall) or RTX 4080 sequential
(~60 min). See artifacts/audit/r2_gpu_design.md C4 section.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = REPO_ROOT / "experiments" / "specs" / "r2_no_tr_ablation.yaml"


@pytest.fixture(scope="module")
def spec() -> dict:
    if not SPEC_PATH.exists():
        pytest.fail(f"R2 C4 spec missing: {SPEC_PATH}")
    return yaml.safe_load(SPEC_PATH.read_text())


# ---------------------------------------------------------------------
# Spec invariants
# ---------------------------------------------------------------------


def test_spec_arch_lstm_only(spec: dict) -> None:
    """C4 needs LSTM only — Phase 5 baseline for paired comparison is
    LSTM × FedAvg × natural-by-BS × 10 seeds. Mamba/Spiking ablations
    can be added in a future C4b sweep if results require."""
    assert spec["archs"] == ["lstm"], (
        f"C4 spec must use lstm only (Phase 5 paired baseline), got "
        f"{spec['archs']}"
    )


def test_spec_algo_fedavg_only(spec: dict) -> None:
    """C4 paired comparison is against Phase 5 LSTM × FedAvg ×
    natural-by-BS, so the algorithm must match exactly."""
    algos = [a["name"] for a in spec["algorithms"]]
    assert algos == ["fedavg"], (
        f"C4 spec must use fedavg only for Phase 5 pairing, got {algos}"
    )


def test_spec_natural_partition(spec: dict) -> None:
    """natural-by-BS = mode=iid + n_clients=7 (the v7 spec convention)."""
    parts = spec["partitions"]
    assert len(parts) == 1
    assert parts[0]["mode"] == "iid", (
        f"C4 partition must be 'iid' (natural-by-BS), got {parts[0]['mode']}"
    )
    assert parts[0]["n_clients"] == 7, (
        f"C4 must use 7 clients (natural ColO-RAN BS count), got "
        f"{parts[0]['n_clients']}"
    )


def test_spec_10_seeds(spec: dict) -> None:
    """10 seeds for paired-bootstrap CI95 vs Phase 5 baseline."""
    assert len(spec["seeds"]) == 10, (
        f"C4 must use 10 seeds for paired comparison (matching Phase 5 "
        f"LSTM × FedAvg × natural-by-BS), got {spec['seeds']}"
    )


def test_spec_drop_categorical_includes_tr(spec: dict) -> None:
    """The whole point of C4: the LSTM forecaster must drop the `tr`
    embedding entirely (no embedding table, no input dim contribution)."""
    drop = spec["arch_overrides"]["lstm"].get("drop_categorical", [])
    assert "tr" in drop, (
        f"C4 spec must set arch_overrides.lstm.drop_categorical=['tr']; "
        f"got {drop}"
    )


def test_spec_matches_phase5_hyperparameters(spec: dict) -> None:
    """Hyperparameters MUST match Phase 5 LSTM × FedAvg × natural-by-BS
    exactly so the only difference between C4 and Phase 5 is the
    presence of the tr embedding."""
    s = spec["shared"]
    assert s["num_rounds"] == 100
    assert s["clients_per_round"] == 5
    assert s["max_steps_per_round"] == 50
    assert s["batch_size"] == 64
    assert s["seq_len"] == 5
    assert s["threshold"] == 0.10
    assert s["pos_weight_split"] == "train"
    assert spec["arch_overrides"]["lstm"]["lr"] == 5.0e-4


# ---------------------------------------------------------------------
# Model construction with drop_categorical=["tr"]
# ---------------------------------------------------------------------


def _build_model(drop_categorical):
    """Helper to build ForecasterV2 with R2 C4 schema."""
    import sys
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from fl_oran.data_v2.encoders import FeatureSchema
    from fl_oran.models.forecaster_v2 import ForecasterV2
    from fl_oran.training.centralized_v3 import (
        V3_CATEGORICAL,
        V3_CAT_SIZES,
        V3_CONTINUOUS,
    )
    schema = FeatureSchema(
        categorical=V3_CATEGORICAL,
        categorical_sizes=V3_CAT_SIZES,
        continuous=V3_CONTINUOUS,
    )
    return ForecasterV2(
        schema=schema, task="classification", seq_len=5,
        drop_categorical=drop_categorical,
    )


def test_forecaster_v2_supports_drop_categorical_kwarg() -> None:
    """RED until forecaster_v2 gains the drop_categorical arg."""
    try:
        m = _build_model(drop_categorical=["tr"])
    except TypeError as e:
        pytest.fail(
            f"ForecasterV2.__init__ must accept drop_categorical kwarg "
            f"per R2 C4 spec; got TypeError: {e}"
        )
    # With "tr" dropped, the embeddings ModuleDict should not contain "tr"
    assert "tr" not in m.embeddings, (
        "drop_categorical=['tr'] must remove tr from ForecasterV2.embeddings"
    )


def test_forecaster_v2_drop_categorical_changes_input_dim() -> None:
    """Dropping a categorical reduces input_dim by cat_embed_dim (=8 default).
    Verify by comparing LSTM input shapes."""
    m_full = _build_model(drop_categorical=None)
    m_no_tr = _build_model(drop_categorical=["tr"])
    full_in = m_full.lstm1.input_size
    no_tr_in = m_no_tr.lstm1.input_size
    assert no_tr_in == full_in - 8, (
        f"drop_categorical=['tr'] should reduce LSTM input_dim by "
        f"cat_embed_dim=8; got full={full_in}, no_tr={no_tr_in}"
    )


def test_forecaster_v2_drop_categorical_forward_works() -> None:
    """The model must accept the SAME input shape (B, L, n_cat) but
    silently ignore the dropped column at forward time. This is so the
    spec doesn't have to also patch the data loader."""
    m = _build_model(drop_categorical=["tr"])
    B, L, n_cat = 4, 5, 4   # n_cat=4 (bs_id, slice_id, sched, tr)
    x_cat = torch.zeros(B, L, n_cat, dtype=torch.long)
    x_cont = torch.zeros(B, L, 17)   # V3_CONTINUOUS has 17 features
    m.eval()
    with torch.no_grad():
        out = m(x_cat, x_cont)
    assert out.shape == (B, 1), (
        f"forward output shape must be (B, 1); got {out.shape}"
    )


def test_forecaster_v2_drop_categorical_invalid_raises() -> None:
    """Defensive: dropping a non-categorical column should raise at
    construction (fail-fast, not silent no-op)."""
    with pytest.raises(ValueError, match="drop_categorical"):
        _build_model(drop_categorical=["nonexistent_column"])
