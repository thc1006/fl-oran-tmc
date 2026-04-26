"""TDD red-phase tests for ``src/fl_oran/training/fl_v7.py`` — Phase 1.5a-v2.

Per ADR-001 D-22, this is the Round 4+5 hostile-audit-hardened
14-test version. Round history:

* Round 1: 7-test initial design.
* Round 2: review found 11-test improved version.
* Round 3: review found 13-test version with 8 fixes for issues
  ranging from silent-pass bugs (``h.get("val_auc", 0.0)``) to
  weak isinstance-only assertions on _build_model.
* Round 4: review identified 4 new issues (params-count schema
  drift, shape pin missing, pytest.raises match too loose, best.pt
  over-strict assertion) and added 2 critical NEW tests:
  identity-aggregation invariant (catches normalization bugs)
  and gradient-flow-after-aggregation (D-19 surrogate-gradient
  risk core).
* Round 5: 17-angle deep audit on test #8 (gradient flow) +
  12-angle on test #13 (idempotency) — found D1 dropout=0
  contract assertion needed, D4 firing-rate threshold needed,
  D17 single-tier max-grad threshold, D18 paranoia post-forward
  state_dict re-check, E13 negative-case (different seed →
  different result). Settled at final 14-test plan.

The 14 tests fall into four groups:

  Tests 1-5: V7Config + _build_model + state_dict (unit, fast)
  Tests 6-7: aggregation invariants (unit, fast)
  Test  8 : D-19 critical gradient flow (unit + paranoid checks)
  Tests 9-12: end-to-end FL smoke + output IO (integration, ~10-30s each)
  Tests 13-14: idempotency + D-12 contract pin

In the RED phase, tests 1-4, 6-14 fail with ``ModuleNotFoundError``
(fl_v7 doesn't exist yet). Test 5 already passes — it pins existing
``SpikingForecaster`` behavior (non-persistent buffers + scalar
shape) as a regression guard before fl_v7 starts mutating the
spiking model usage pattern.
"""
from __future__ import annotations

import importlib
import json
from dataclasses import replace  # idempotency test 13 needs it
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

# Single source of truth for production schema (Round 4 B1 fix).
# If V3_CAT_SIZES changes, params count tests (44553/40489/43593)
# will fail loudly — surfaces schema drift instead of hiding it.
from fl_oran.data_v2.encoders import FeatureSchema
from fl_oran.training.centralized_v3 import (
    V3_CATEGORICAL,
    V3_CAT_SIZES,
    V3_CONTINUOUS,
)


def _import_fl_v7():
    """Lazy import. Tests fail clearly during RED phase rather than
    aborting test collection."""
    return importlib.import_module("fl_oran.training.fl_v7")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def schema():
    """Production-schema FeatureSchema. Pinned to V3_CAT_SIZES so
    params-count tests (44553 / 40489 / 43593) reproduce."""
    return FeatureSchema(
        categorical=V3_CATEGORICAL,
        categorical_sizes=V3_CAT_SIZES,
        continuous=V3_CONTINUOUS,
    )


def _make_synthetic_parquet(out_path: Path, n_rows: int = 4000) -> Path:
    """Generate ColO-RAN-shaped parquet with weakly-learnable signal.

    ``ul_bler`` is set to be slightly correlated with ``dl_bler`` so
    a model with capacity > 0 can drive loss down. Roughly 35% positive
    rate at threshold=0.10, matching real ColO-RAN distribution.
    7 bs_ids (1..7), 4 slice_ids (0..3), 4 sched modes (0..3), 28 tr ids.
    """
    rng = np.random.default_rng(20260426)
    rows = []
    for tr in range(28):
        for _ in range(n_rows // 28):
            cont_vec = rng.normal(0, 1, len(V3_CONTINUOUS))
            ul_bler_idx = V3_CONTINUOUS.index("ul_bler")
            cont_vec[ul_bler_idx] = (
                0.07 + 0.07 * rng.normal()
                + 0.03 * cont_vec[V3_CONTINUOUS.index("dl_bler")]
            )
            row = {
                "bs_id": int(rng.integers(1, 8)),
                "slice_id": int(rng.integers(0, 4)),
                "sched": int(rng.integers(0, 4)),
                "tr": tr,
                **{c: float(v) for c, v in zip(V3_CONTINUOUS, cont_vec)},
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_parquet(out_path)
    return out_path


@pytest.fixture(scope="module")
def synthetic_parquet(tmp_path_factory):
    """Module-scoped: generate parquet once, reuse across tests in the
    module. ~30-50 ms write per call — module scope amortizes."""
    base = tmp_path_factory.mktemp("v7_data")
    return _make_synthetic_parquet(base / "synthetic_coloran.parquet")


@pytest.fixture
def deterministic_torch():
    """Force deterministic mode for the test, restore after.

    Round 5 E1: on CPU this is mostly defensive (PyTorch CPU ops
    deterministic by default for our op set: Embedding, LSTM, Linear)
    but protects against future op additions that introduce
    non-determinism.
    """
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    yield
    torch.use_deterministic_algorithms(prev)


# ---------------------------------------------------------------------------
# Tests 1-5: V7Config + _build_model + state_dict
# ---------------------------------------------------------------------------

def test_v7_config_field_defaults_AND_unknown_arch_rejected_at_build(schema):
    """V7Config defaults arch="lstm" + algorithm="fedavg"; unknown arch
    constructs OK (matches dataclass conventions) but ``_build_model``
    rejects with ValueError carrying the offending arch name in the
    message (Round 4 B3 fix: precise match, not loose KeyError)."""
    fl_v7 = _import_fl_v7()
    cfg = fl_v7.V7Config()
    assert cfg.arch == "lstm"
    assert cfg.algorithm == "fedavg"
    cfg2 = fl_v7.V7Config(arch="not_a_real_arch")
    assert cfg2.arch == "not_a_real_arch"
    with pytest.raises(ValueError, match=r"unknown arch.*not_a_real_arch"):
        fl_v7._build_model(cfg2, schema)


def test_v7_build_model_lstm_pins_params_44553(schema):
    """Hand-calc validated params: 392 (emb) + 29440 (lstm1)
    + 12544 (lstm2) + 2112 (fc) + 65 (head) = 44553. Pinning catches
    BOTH registry kwargs drift AND schema drift in V3_CAT_SIZES."""
    fl_v7 = _import_fl_v7()
    from fl_oran.models.forecaster_v2 import ForecasterV2
    cfg = fl_v7.V7Config(arch="lstm")
    model = fl_v7._build_model(cfg, schema)
    assert isinstance(model, ForecasterV2)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params == 44553, f"lstm params drifted: {n_params}"


def test_v7_build_model_mamba_pins_params_40489(schema):
    """Per docs/RESULTS_V6_STAGE1_ANALYSIS.md §3.1 baseline."""
    fl_v7 = _import_fl_v7()
    from fl_oran.models.mamba_forecaster import MambaForecaster
    cfg = fl_v7.V7Config(arch="mamba")
    model = fl_v7._build_model(cfg, schema)
    assert isinstance(model, MambaForecaster)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params == 40489, f"mamba params drifted: {n_params}"


def test_v7_build_model_spiking_expand2_pins_kwargs_AND_params_43593(schema):
    """spiking_expand2 specifically: backbone_d_model=56, expand=2
    (so d_inner=112). This catches Tier B.2 'wrong baselines' bug
    class — registry must hydrate these exact kwargs into block."""
    fl_v7 = _import_fl_v7()
    from fl_oran.models.spiking_forecaster import SpikingForecaster
    cfg = fl_v7.V7Config(arch="spiking_expand2")
    model = fl_v7._build_model(cfg, schema)
    assert isinstance(model, SpikingForecaster)
    block = model.blocks[0]
    assert block.d_model == 56, f"d_model drifted: {block.d_model}"
    assert block.expand == 2, f"expand drifted: {block.expand}"
    assert block.d_inner == 112, f"d_inner drifted: {block.d_inner}"
    assert len(model.blocks) == 2
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params == 43593, f"spiking_expand2 params drifted: {n_params}"


def test_spiking_state_dict_excludes_buffers_AND_attributes_exist_AND_are_scalar(schema):
    """Round 4 B2 + Round 5 D18: spiking blocks expose ``spike_count``
    and ``forward_inferences`` as non-persistent **scalar** tensor
    buffers. They MUST exist as attributes (else energy_metrics
    ``float(spike_count)`` breaks) AND MUST NOT appear in state_dict
    (else FedAvg's ``weighted_average_state_dicts`` would average
    spike counts across clients — silent corruption of energy data).

    Already-true regression guard: passes against current
    SpikingForecaster, but pins the contract before fl_v7 starts
    cross-aggregating state dicts.
    """
    from fl_oran.models.spiking_forecaster import SpikingForecaster

    model = SpikingForecaster(
        schema=schema, task="classification", seq_len=5,
        backbone_d_model=56, backbone_expand=2,
    )
    for blk in model.blocks:
        assert hasattr(blk, "spike_count")
        assert isinstance(blk.spike_count, torch.Tensor)
        assert blk.spike_count.shape == (), (
            f"spike_count must be scalar; got shape {blk.spike_count.shape}"
        )
        assert hasattr(blk, "forward_inferences")
        assert isinstance(blk.forward_inferences, torch.Tensor)
        assert blk.forward_inferences.shape == ()
    keys = list(model.state_dict().keys())
    for k in keys:
        assert "spike_count" not in k, f"leaked: {k}"
        assert "forward_inferences" not in k, f"leaked: {k}"


# ---------------------------------------------------------------------------
# Tests 6-7: FL aggregation invariants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arch_name", ["lstm", "mamba", "spiking_expand2"])
def test_v7_identity_aggregation_returns_self_per_arch(schema, arch_name):
    """Round 4 A2 critical: ``weighted_average_state_dicts`` of two
    identical state dicts (same seed, same init) with weights
    [0.5, 0.5] must return bit-equal result. Catches aggregation
    normalization bugs.

    weights=[0.5, 0.5] is IEEE 754 exact: 0.5*v + 0.5*v == v for any
    float32 v (multiplication by power-of-2 just shifts exponent).
    """
    fl_v7 = _import_fl_v7()
    from fl_oran.federated import weighted_average_state_dicts

    cfg = fl_v7.V7Config(arch=arch_name)

    torch.manual_seed(42)
    m1 = fl_v7._build_model(cfg, schema)
    torch.manual_seed(42)
    m2 = fl_v7._build_model(cfg, schema)

    sd_avg = weighted_average_state_dicts(
        [m1.state_dict(), m2.state_dict()], [0.5, 0.5]
    )
    for k, v in m1.state_dict().items():
        assert torch.allclose(sd_avg[k], v, atol=1e-7), (
            f"{arch_name} key {k!r} drifted from identical input"
        )


def test_v7_random_aggregation_load_AND_forward_no_nan_for_spiking_expand2(schema):
    """Two DIFFERENT-init spiking_expand2 models, average state
    dicts, load into target, forward — must produce finite output.
    Catches: load_state_dict raising on key mismatch, forward NaN
    under aggregated weights."""
    fl_v7 = _import_fl_v7()
    from fl_oran.federated import weighted_average_state_dicts
    from fl_oran.models.spiking_forecaster import SpikingForecaster

    cfg = fl_v7.V7Config(arch="spiking_expand2")
    torch.manual_seed(42)
    m1 = fl_v7._build_model(cfg, schema)
    torch.manual_seed(99)
    m2 = fl_v7._build_model(cfg, schema)

    sd_avg = weighted_average_state_dicts(
        [m1.state_dict(), m2.state_dict()], [0.5, 0.5]
    )
    m_target = SpikingForecaster(
        schema=schema, task="classification", seq_len=5,
        backbone_d_model=56, backbone_expand=2,
    )
    m_target.load_state_dict(sd_avg)  # no key mismatch raise
    m_target.eval()

    torch.manual_seed(7)
    x_cat = torch.randint(0, 4, (4, 5, 4)).long()
    x_cont = torch.randn(4, 5, 17)
    with torch.no_grad():
        out = m_target(x_cat, x_cont)
    assert torch.isfinite(out).all()


def test_v7_gradient_flow_through_spiking_expand2_after_aggregation(schema):
    """ADR D-19 unprecedented combination risk: surrogate × Adam ×
    FL aggregation must produce finite, non-trivial gradients with
    the spike-driven AC path actively exercised.

    Round 5 hardening:
      - D1: dropout=0 contract assertion (test logic depends on it
        because eval() makes dropout a no-op; if registry default
        changed to dropout>0 the spike path would still work but
        test rationale would need updating).
      - D4: firing_rate > 0.1% — ensures spike-driven AC path is
        actually exercised by this batch (otherwise we'd only test
        dense MAC gradient flow, missing half of D-19's risk).
      - D17: max_abs_grad > 1e-7 — catches dead surrogate (where
        gradients are all zero/inf despite isfinite passing).
      - D18: post-forward state_dict re-check — paranoia that
        ``self.spike_count = self.spike_count + ...`` reassignment
        in forward path doesn't accidentally promote the buffer to
        persistent.
    """
    fl_v7 = _import_fl_v7()
    from fl_oran.federated import weighted_average_state_dicts
    from fl_oran.models.spiking_forecaster import SpikingForecaster

    cfg = fl_v7.V7Config(arch="spiking_expand2")
    torch.manual_seed(42)
    m1 = fl_v7._build_model(cfg, schema)
    torch.manual_seed(99)
    m2 = fl_v7._build_model(cfg, schema)

    sd_avg = weighted_average_state_dicts(
        [m1.state_dict(), m2.state_dict()], [0.5, 0.5]
    )
    m_target = SpikingForecaster(
        schema=schema, task="classification", seq_len=5,
        backbone_d_model=56, backbone_expand=2,
    )
    m_target.load_state_dict(sd_avg)
    m_target.eval()

    # D1: dropout=0 contract (registry default for spiking_expand2)
    assert m_target.dropout.p == 0.0, (
        "test logic relies on registry default dropout=0; if registry "
        "changes, force cfg.arch_kwargs={'dropout': 0} explicitly"
    )

    # Deterministic + scaled input → push past LIF threshold=1.0
    torch.manual_seed(7)
    x_cat = torch.randint(0, 4, (16, 5, 4)).long()
    x_cont = torch.randn(16, 5, 17) * 2.0

    out = m_target(x_cat, x_cont)
    out.sum().backward()

    # D4: spike-driven AC path actually exercised
    total_spikes = sum(float(b.spike_count) for b in m_target.blocks)
    total_slots = 16 * 5 * 112 * 2  # batch × seq × d_inner × n_blocks
    firing_rate = total_spikes / total_slots
    assert firing_rate > 0.001, (
        f"firing_rate={firing_rate:.4%} below 0.1% threshold — "
        f"spike-driven AC gradient path not exercised; test invalid"
    )

    # All trainable params have finite grad (catches NaN/Inf surrogate)
    for name, p in m_target.named_parameters():
        assert p.grad is not None, f"{name}: no grad"
        assert torch.isfinite(p.grad).all(), f"{name}: non-finite grad"

    # D17: at least one param with non-trivial grad (catches dead surrogate)
    max_abs_grad = max(p.grad.abs().max().item() for p in m_target.parameters())
    assert max_abs_grad > 1e-7, (
        f"all gradients ≈ 0 (max={max_abs_grad:.2e}); "
        f"surrogate likely dead under aggregated weights"
    )

    # D18: spike_count remains non-persistent post-forward
    sd_after = m_target.state_dict()
    for k in sd_after:
        assert "spike_count" not in k, f"spike_count leaked post-forward: {k}"
        assert "forward_inferences" not in k


# ---------------------------------------------------------------------------
# Tests 9-12: end-to-end FL smoke + output IO
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("algo,algo_kwargs", [
    ("fedavg", {}),
    ("fedprox", {"mu": 0.01}),
])
def test_v7_smoke_lstm_iid_per_algo(synthetic_parquet, tmp_path, algo, algo_kwargs):
    """Round 4 H: parametric over fedavg + fedprox to exercise
    algo_kwargs threading through fl_v7's algorithm dispatch.

    Round 4 #5 (loss-decrease relaxation): instead of strict
    last < first, check that AT LEAST ONE later round improves
    over round 0 (catches "training loop runs but does nothing"
    bugs without flaking on toy-data noise)."""
    fl_v7 = _import_fl_v7()

    cfg = fl_v7.V7Config(
        name=f"smoke_lstm_iid_{algo}",
        arch="lstm",
        algorithm=algo,
        algo_kwargs=algo_kwargs,
        partition_mode="iid",
        n_clients=7,
        num_rounds=3,
        clients_per_round=4,
        max_steps_per_round=20,
        batch_size=32,
        lr=5e-4,
        lr_warmup_rounds=1,
        unified_parquet=synthetic_parquet,
        sample_ratio=1.0,
        threshold=0.10,
        seq_len=5,
        seed=42,
        device="cpu",
        mixed_precision="off",
        output_dir=str(tmp_path / f"out_{algo}"),
    )
    result = fl_v7.run_v7_sweep(cfg)

    assert "history" in result
    history = result["history"]
    assert len(history) == 3

    losses = [h["train_loss"] for h in history]
    for l in losses:
        assert np.isfinite(l), f"NaN/Inf in losses: {losses}"

    # At least one later round improves on round 0 (with epsilon)
    assert min(losses[1:]) < losses[0] + 0.05, (
        f"loss frozen across rounds: {losses}; training loop may not be updating"
    )


def test_v7_smoke_spiking_expand2_dirichlet_3_rounds_history_finite_AND_val_auc_present(
    synthetic_parquet, tmp_path,
):
    """End-to-end FL on the architecture combination flagged by
    ADR D-19 as ``unprecedented`` (surrogate gradient + Adam + FL).

    Round 4 #1 critical fix: explicit ``assert "val_auc" in h``,
    NOT silent ``h.get("val_auc", 0.0)`` which always returns finite
    0.0 if the key is missing — false-pass bug class.

    Soft loss-decrease (only finite check) because surrogate
    gradients on toy data with 3 rounds may legitimately not show
    monotone improvement; the hard claim is "no NaN, no crash,
    val_auc reported as required".
    """
    fl_v7 = _import_fl_v7()

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
        unified_parquet=synthetic_parquet,
        sample_ratio=1.0,
        threshold=0.10,
        seq_len=5,
        seed=42,
        device="cpu",
        mixed_precision="off",
        output_dir=str(tmp_path / "out"),
    )
    result = fl_v7.run_v7_sweep(cfg)

    history = result["history"]
    assert len(history) == 3
    for h in history:
        assert np.isfinite(h["train_loss"]), h
        # Round 4 #1 critical: val_auc key MUST exist (not silent default)
        assert "val_auc" in h, f"history entry missing val_auc: {h}"
        assert np.isfinite(h["val_auc"]), h


def test_v7_moon_non_lstm_arch_raises_at_select_fast(schema):
    """Round 4 E + ADR D-22 contract: ``_select_algorithm`` raises
    ``NotImplementedError`` on MOON (any arch in Phase 1.5 minimum-
    viable; D-16's per-arch encode_fn is paper-level open question
    deferred to Phase 2 polish).

    Fail-fast at config-time helper (no parquet load, no training)
    means the test runs in milliseconds.
    """
    fl_v7 = _import_fl_v7()
    cfg = fl_v7.V7Config(algorithm="moon", arch="spiking_expand2")
    with pytest.raises(NotImplementedError, match=r"MOON"):
        fl_v7._select_algorithm(cfg)


def test_v7_sweep_writes_summary_history_AND_summary_has_test_auc_key(
    synthetic_parquet, tmp_path,
):
    """Round 4 G: ADR D-7 mandates ``summary.json`` + ``history.csv``
    written to ``cfg.output_dir / cfg.name``. Round 5 B4: ``best.pt``
    is optional — toy data on 2 rounds may not improve val_auc, in
    which case best_state never gets written. Don't require it.

    Validates summary.json content: must have ``test_auc`` key (the
    primary metric the aggregator reads)."""
    fl_v7 = _import_fl_v7()
    cell_name = "out_io_test"
    cfg = fl_v7.V7Config(
        name=cell_name,
        arch="lstm",
        algorithm="fedavg",
        partition_mode="iid",
        num_rounds=2,
        clients_per_round=2,
        max_steps_per_round=10,
        batch_size=16,
        lr=5e-4,
        lr_warmup_rounds=0,
        unified_parquet=synthetic_parquet,
        sample_ratio=1.0,
        threshold=0.10,
        seq_len=5,
        seed=42,
        device="cpu",
        mixed_precision="off",
        output_dir=str(tmp_path),
    )
    fl_v7.run_v7_sweep(cfg)

    cell_dir = tmp_path / cell_name
    assert (cell_dir / "summary.json").exists(), \
        f"summary.json missing in {cell_dir}"
    assert (cell_dir / "history.csv").exists(), \
        f"history.csv missing in {cell_dir}"
    summary = json.loads((cell_dir / "summary.json").read_text())
    assert "test_auc" in summary, \
        f"summary.json keys: {list(summary.keys())}"


# ---------------------------------------------------------------------------
# Tests 13-14: idempotency + D-12 contract
# ---------------------------------------------------------------------------

def test_v7_sweep_idempotent_AND_seed_actually_matters(
    deterministic_torch, synthetic_parquet, tmp_path,
):
    """Round 4 A4 + Round 5 E13 critical: 3-sweep test catches
    TWO classes of seed bug:

      1. r1 == r2 (with global RNG corrupted between runs):
         fl_v7 must seed all RNG sources (torch, numpy, random)
         from cfg.seed. If any source is left to global state,
         corrupting global RNG between runs would cause r1 != r2.

      2. r1 != r3 (different seed): if fl_v7 hardcodes a seed
         instead of reading from cfg.seed, r1 == r3 would pass
         the first check but fail this one.

    Tiny budget (1 round × 5 steps × 2 clients) keeps total
    runtime ~15s for 3 sweeps."""
    fl_v7 = _import_fl_v7()

    base_cfg = fl_v7.V7Config(
        name="idem_base",
        arch="lstm",
        algorithm="fedavg",
        partition_mode="iid",
        num_rounds=1,
        clients_per_round=2,
        max_steps_per_round=5,
        batch_size=16,
        lr=5e-4,
        lr_warmup_rounds=0,
        unified_parquet=synthetic_parquet,
        sample_ratio=1.0,
        threshold=0.10,
        seq_len=5,
        seed=42,
        device="cpu",
        mixed_precision="off",
    )
    cfg1 = replace(base_cfg, name="r1", output_dir=str(tmp_path / "r1"))
    r1 = fl_v7.run_v7_sweep(cfg1)

    # E13 part 1: corrupt global RNG → seed=42 must still give same result
    torch.manual_seed(99999)
    np.random.seed(99999)

    cfg2 = replace(base_cfg, name="r2", output_dir=str(tmp_path / "r2"))
    r2 = fl_v7.run_v7_sweep(cfg2)
    assert r1["history"][-1]["train_loss"] == r2["history"][-1]["train_loss"], (
        f"r1={r1['history'][-1]['train_loss']} != "
        f"r2={r2['history'][-1]['train_loss']}; "
        f"fl_v7 likely depends on global RNG (must seed all sources from cfg.seed)"
    )

    # E13 part 2: seed=43 MUST give different result
    cfg3 = replace(base_cfg, seed=43, name="r3", output_dir=str(tmp_path / "r3"))
    r3 = fl_v7.run_v7_sweep(cfg3)
    assert r1["history"][-1]["train_loss"] != r3["history"][-1]["train_loss"], (
        f"seed=42 and seed=43 give SAME train_loss; "
        f"fl_v7 likely ignores cfg.seed (hardcoded seed bug)"
    )


def test_v7_config_pos_weight_split_default_train_per_D12():
    """ADR D-12 audit fix: pos_weight derives from train split
    (not test) to avoid label-distribution leakage. fl_v7 must
    inherit this default. M5 had a bug where pos_weight came from
    test split — caught in adversarial review and fixed.
    """
    fl_v7 = _import_fl_v7()
    cfg = fl_v7.V7Config()
    assert cfg.pos_weight_split == "train", (
        f"D-12 contract violated: pos_weight_split={cfg.pos_weight_split!r}; "
        f"must default to 'train' to avoid leakage from test split"
    )
