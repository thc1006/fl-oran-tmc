"""TDD red-phase tests for ``src/fl_oran/training/fl_v7.py``.

Phase 1.5a (per ADR-001 Stage 2 plan §S2-W1..3): pin the expected
behavior of an arch-agnostic FL trainer that supports the v6 lineup
{lstm, mamba, mamba_expand2, spiking, spiking_expand2} via the same
ARCH_REGISTRY pattern as ``experiments/run_v6_arch_sweep.py``.

The existing ``fl_v5.py`` is hard-bound to ``ForecasterV2`` (LSTM); per
ADR D-9 we do not refactor it. Phase 1.5b will introduce a parallel
``fl_v7.py`` that:

  * accepts ``arch`` in :class:`V7Config`
  * dispatches model construction via ``_build_model``
  * runs FL rounds via the existing 6-algorithm registry
  * reuses ``federated_fit_scaler``, ``partition_clients``,
    ``weighted_average_state_dicts``, ``train_one_client_capped`` so we
    do not duplicate SoT functions (D-3)
  * raises ``NotImplementedError`` on MOON × non-LSTM (D-16 open
    question — encode_fn for spiking/mamba is paper-level design)

Tests are ordered by increasing setup cost:

  1. unit: V7Config exposes ``arch`` field
  2-4. unit: ``_build_model`` dispatches to the right ctor with
       registry-pinned kwargs
  5. unit: Spiking non-persistent buffers are excluded from
       ``state_dict`` (essential — guarantees they are not corrupted by
       FedAvg's ``weighted_average_state_dicts`` cross-client
       aggregation)
  6-7. integration: ``run_v7_sweep`` end-to-end on synthetic parquet
       with LSTM × IID and Spiking_expand2 × Dirichlet for 3 rounds
       each; loss decreases, no crash. Run on CPU with toy data; per
       test budget < 30 s.

Some tests will be GREEN immediately (test 5: existing behavior of
SpikingForecaster's ``register_buffer(persistent=False)``); others are
RED until ``fl_v7.py`` exists.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# Path to the (yet to be created) fl_v7 module. Tests import lazily so
# import errors during the red phase produce a clear test failure rather
# than a collection error that aborts the whole test run.
def _import_fl_v7():
    """Lazily import ``fl_oran.training.fl_v7``; tests fail clearly if
    the module is missing (RED phase) rather than aborting collection."""
    return importlib.import_module("fl_oran.training.fl_v7")


# ---------------------------------------------------------------------------
# Schema / synthetic data fixture (mirrors V3_CATEGORICAL / V3_CONTINUOUS).
# ---------------------------------------------------------------------------

_CATEGORICAL = ["bs_id", "slice_id", "sched", "tr"]
_CAT_SIZES = {"bs_id": 8, "slice_id": 4, "sched": 4, "tr": 29}
_CONTINUOUS = [
    "num_ues", "slice_prb",
    "sum_requested_prbs", "sum_granted_prbs",
    "tx_brate_dl_Mbps", "rx_brate_ul_Mbps",
    "tx_pkts_dl", "rx_pkts_ul",
    "dl_buffer_bytes", "ul_buffer_bytes",
    "dl_bler", "ul_bler",
    "dl_mcs", "ul_mcs", "dl_cqi", "ul_sinr", "ul_rssi",
]


def _make_synthetic_parquet(tmp_path: Path, n_rows: int = 4000) -> Path:
    """Generate a tiny ColO-RAN-shaped parquet for FL smoke tests.

    Each tr id gets ~``n_rows / 28`` rows with deterministic-but-noisy
    continuous values. ``ul_bler`` is set so the binary classification
    target ``ul_bler > 0.10`` has roughly 35% positive rate (matches
    real ColO-RAN). The values are weakly predictable from a few
    continuous features so a model with capacity > 0 can drive loss
    down on this toy data.
    """
    rng = np.random.default_rng(20260426)
    rows = []
    # 7 bs_ids (1..7 — match real ColO-RAN gNB count); 4 slice_ids; 4 sched modes.
    for tr in range(28):
        for _ in range(n_rows // 28):
            bs_id = int(rng.integers(1, 8))
            slice_id = int(rng.integers(0, 4))
            sched = int(rng.integers(0, 4))
            cont_vec = rng.normal(0, 1, len(_CONTINUOUS))
            # Make ul_bler weakly positive-correlated to certain columns
            # so the toy target is learnable.
            ul_bler_idx = _CONTINUOUS.index("ul_bler")
            cont_vec[ul_bler_idx] = (
                0.07 + 0.07 * rng.normal()
                + 0.03 * cont_vec[_CONTINUOUS.index("dl_bler")]
            )
            row = {
                "bs_id": bs_id, "slice_id": slice_id, "sched": sched, "tr": tr,
                **{c: float(v) for c, v in zip(_CONTINUOUS, cont_vec)},
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    out = tmp_path / "synthetic_coloran.parquet"
    df.to_parquet(out)
    return out


# ---------------------------------------------------------------------------
# Tests 1-4: V7Config + _build_model API
# ---------------------------------------------------------------------------

def test_v7_config_has_arch_field():
    """V7Config must accept an ``arch`` field whose default is "lstm"
    (same as Stage 1 preregistered baseline)."""
    fl_v7 = _import_fl_v7()
    cfg = fl_v7.V7Config()
    assert hasattr(cfg, "arch")
    assert cfg.arch == "lstm"


def test_v7_build_model_dispatches_lstm():
    """``_build_model(cfg, schema)`` with cfg.arch="lstm" returns a
    ForecasterV2 instance with the same kwargs the Stage 1 runner
    used (registry default = empty kwargs)."""
    fl_v7 = _import_fl_v7()
    from fl_oran.data_v2.encoders import FeatureSchema
    from fl_oran.models.forecaster_v2 import ForecasterV2

    schema = FeatureSchema(
        categorical=_CATEGORICAL,
        categorical_sizes=_CAT_SIZES,
        continuous=_CONTINUOUS,
    )
    cfg = fl_v7.V7Config(arch="lstm")
    model = fl_v7._build_model(cfg, schema)
    assert isinstance(model, ForecasterV2)


def test_v7_build_model_dispatches_mamba():
    """cfg.arch="mamba" returns MambaForecaster with registry default
    kwargs (d_model=64, expand=1, n_blocks=2)."""
    fl_v7 = _import_fl_v7()
    from fl_oran.data_v2.encoders import FeatureSchema
    from fl_oran.models.mamba_forecaster import MambaForecaster

    schema = FeatureSchema(
        categorical=_CATEGORICAL,
        categorical_sizes=_CAT_SIZES,
        continuous=_CONTINUOUS,
    )
    cfg = fl_v7.V7Config(arch="mamba")
    model = fl_v7._build_model(cfg, schema)
    assert isinstance(model, MambaForecaster)


def test_v7_build_model_dispatches_spiking_expand2_with_correct_kwargs():
    """cfg.arch="spiking_expand2" must hydrate the registry's spec:
    backbone_d_model=56, backbone_expand=2 (so d_inner=112 inside each
    SSM block). Pinning these confirms fl_v7 is reusing the same
    ARCH_REGISTRY as run_v6_arch_sweep.py — single source of truth.
    """
    fl_v7 = _import_fl_v7()
    from fl_oran.data_v2.encoders import FeatureSchema
    from fl_oran.models.spiking_forecaster import SpikingForecaster

    schema = FeatureSchema(
        categorical=_CATEGORICAL,
        categorical_sizes=_CAT_SIZES,
        continuous=_CONTINUOUS,
    )
    cfg = fl_v7.V7Config(arch="spiking_expand2")
    model = fl_v7._build_model(cfg, schema)
    assert isinstance(model, SpikingForecaster)
    # First block carries the expand-2 dimensionality.
    block = model.blocks[0]
    assert block.d_model == 56
    assert block.expand == 2
    assert block.d_inner == 112


# ---------------------------------------------------------------------------
# Test 5: Spiking non-persistent buffers excluded from state_dict
# ---------------------------------------------------------------------------

def test_spiking_state_dict_excludes_non_persistent_buffers():
    """``spike_count`` and ``forward_inferences`` are
    ``register_buffer(..., persistent=False)`` so they are NOT in
    ``state_dict()``. This is what allows FedAvg's
    ``weighted_average_state_dicts`` to cross-aggregate spiking
    clients without contaminating per-client spike counts.

    Already-true regression: passes against the existing model, but
    pinning it here protects against future "let's make these
    persistent for checkpointing" refactors that would silently break
    FL aggregation.
    """
    from fl_oran.data_v2.encoders import FeatureSchema
    from fl_oran.models.spiking_forecaster import SpikingForecaster

    schema = FeatureSchema(
        categorical=_CATEGORICAL,
        categorical_sizes=_CAT_SIZES,
        continuous=_CONTINUOUS,
    )
    model = SpikingForecaster(
        schema=schema, task="classification", seq_len=5,
        backbone_d_model=56, backbone_expand=2,
    )
    keys = list(model.state_dict().keys())
    for k in keys:
        assert "spike_count" not in k, (
            f"spike_count leaked into state_dict at key {k!r} — "
            f"register_buffer must remain persistent=False."
        )
        assert "forward_inferences" not in k, (
            f"forward_inferences leaked into state_dict at key {k!r}."
        )


# ---------------------------------------------------------------------------
# Tests 6-7: end-to-end FL smoke
# ---------------------------------------------------------------------------

@pytest.mark.timeout(60)
def test_v7_smoke_lstm_iid_3_rounds(tmp_path):
    """Full FL run on synthetic data: LSTM, IID partition (by bs_id),
    3 rounds, FedAvg. Must complete without crash, mean train loss in
    last round must be strictly below mean train loss in first round
    (non-trivial learning signal).

    Why IID + bs_id: per partition.py, IID mode partitions by bs_id and
    ignores --n-clients (each gNB = one client). This is the canonical
    "FL ≈ centralized" smoke that confirms the aggregation pipeline
    works at all before testing harder partition modes.
    """
    fl_v7 = _import_fl_v7()
    parquet = _make_synthetic_parquet(tmp_path)
    cfg = fl_v7.V7Config(
        name="smoke_lstm_iid",
        arch="lstm",
        algorithm="fedavg",
        partition_mode="iid",
        n_clients=7,                  # ignored by iid mode but documented
        num_rounds=3,
        clients_per_round=4,
        max_steps_per_round=20,
        batch_size=32,
        lr=5e-4,
        lr_warmup_rounds=1,
        unified_parquet=parquet,
        sample_ratio=1.0,
        threshold=0.10,
        seq_len=5,
        seed=42,
        device="cpu",                 # CPU smoke; integration will go GPU
        mixed_precision="off",
        output_dir=str(tmp_path / "out"),
    )
    result = fl_v7.run_v7_sweep(cfg)
    assert "history" in result
    history = result["history"]
    assert len(history) == 3
    # Loss must decrease meaningfully across 3 rounds.
    first = history[0]["train_loss"]
    last = history[-1]["train_loss"]
    assert np.isfinite(first) and np.isfinite(last)
    assert last < first, (
        f"FL run did not learn: loss went {first:.4f} -> {last:.4f}"
    )


@pytest.mark.timeout(120)
def test_v7_smoke_spiking_expand2_dirichlet_3_rounds(tmp_path):
    """End-to-end FL on the architecture that's hardest to integrate:
    SpikingForecaster with backbone_expand=2, Dirichlet partition,
    FedAvg, 3 rounds. The combination

      surrogate-gradient + Adam + FL aggregation + spike buffers

    is flagged by ADR D-19 as ``unprecedented combination``. This test
    is the de-risking step before Phase 2 commits to a real GPU sweep.

    The smoke verifies:
      - run_v7_sweep does not crash on spiking_expand2
      - server-side weighted_average_state_dicts handles the larger
        expanded-channel state dict correctly
      - non-persistent spike buffers do not pollute aggregation
        (test 5 pins the precondition; this test exercises the full path)

    Loss-decrease check is **soft** (training noise on toy data with
    surrogate gradients can plateau in 3 rounds) — the hard assertion
    is "no NaN / no crash / finite final loss".
    """
    fl_v7 = _import_fl_v7()
    parquet = _make_synthetic_parquet(tmp_path, n_rows=2800)
    cfg = fl_v7.V7Config(
        name="smoke_spiking_expand2_dirichlet",
        arch="spiking_expand2",
        algorithm="fedavg",
        partition_mode="dirichlet",
        alpha=0.5,
        n_clients=4,
        num_rounds=3,
        clients_per_round=4,
        max_steps_per_round=20,
        batch_size=32,
        lr=5e-4,
        lr_warmup_rounds=1,
        unified_parquet=parquet,
        sample_ratio=1.0,
        threshold=0.10,
        seq_len=5,
        seed=42,
        device="cpu",
        mixed_precision="off",
        output_dir=str(tmp_path / "out"),
    )
    result = fl_v7.run_v7_sweep(cfg)
    assert "history" in result
    history = result["history"]
    assert len(history) == 3
    # Hard checks: no NaN / inf, finite final loss.
    for h in history:
        assert np.isfinite(h["train_loss"]), h
        assert np.isfinite(h.get("val_auc", 0.0)), h
    # Final loss exists and is finite (no requirement that it must drop
    # — surrogate gradients on toy data with 3 rounds may not converge).
    assert np.isfinite(history[-1]["train_loss"])
