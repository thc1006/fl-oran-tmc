"""R2 C1 — same-step centralized LSTM methodology + spec invariants.

TDD-RED before C1 GREEN: assert the spec yaml exists with correct
parameters AND the centralized LSTM runner supports the `--max-steps`
truncation mode the spec depends on. Tests should FAIL until both
land. After GREEN, prevents regression on the C1 reproducibility
contract (same FL-equivalent gradient-step budget = 25,000).

Reviewer R2 #14 / A1: §6.7 "federation cost at equivalent training
budget" was factually wrong because FL is ~0.1 epoch and centralized
was 1 epoch (10× difference). C1 measures centralized at the matched
25k gradient steps so we can replace that section with a true
same-budget comparison.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = REPO_ROOT / "experiments" / "specs" / "r2_same_step_centralized.yaml"
RUNNER_PATH = REPO_ROOT / "experiments" / "run_p1_centralized_lstm.py"


@pytest.fixture(scope="module")
def spec() -> dict:
    if not SPEC_PATH.exists():
        pytest.fail(
            f"R2 C1 spec missing: {SPEC_PATH}. Write it per "
            "artifacts/audit/r2_gpu_design.md."
        )
    return yaml.safe_load(SPEC_PATH.read_text())


# ---------------------------------------------------------------------
# Spec invariants
# ---------------------------------------------------------------------


def test_spec_arch_is_lstm(spec: dict) -> None:
    assert spec["arch"] == "lstm", (
        "C1 must use LSTM (matching Phase 5 LSTM × FedAvg × natural-by-BS "
        "for the §6.7 same-budget comparison)"
    )


def test_spec_max_steps_matches_fl_budget(spec: dict) -> None:
    """The whole point of C1: gradient-step budget MUST equal FL's.
    FL = 100 rounds × 50 max_steps × 5 clients/round = 25_000 steps."""
    assert spec["max_steps"] == 25_000, (
        f"C1 max_steps must equal FL budget = 100*50*5 = 25_000, "
        f"got {spec['max_steps']}. Anything else defeats the purpose: "
        f"§6.7 needs a same-budget centralized to drop the 'federation "
        f"cost at equivalent training budget' factual error."
    )


def test_spec_5_seeds(spec: dict) -> None:
    """Need 5 seeds for paired-bootstrap CI95 vs Phase 5 LSTM × FedAvg."""
    assert len(spec["seeds"]) == 5, (
        f"C1 must run 5 seeds for paired comparison, got {spec['seeds']}"
    )


def test_spec_hyperparameters_match_phase5_fl(spec: dict) -> None:
    """Hyperparameters must match Phase 5 FL LSTM exactly so the only
    difference between centralized and FL is the centralization itself
    (no mixed batch_size / lr / seq_len confound)."""
    assert spec["batch_size"] == 64, "must match Phase 5 batch_size"
    assert spec["lr"] == 5.0e-4, "must match Phase 5 LSTM lr"
    assert spec["seq_len"] == 5, "must match Phase 5 seq_len"
    assert spec["threshold"] == 0.10, "must match Phase 5 SLA threshold"
    assert spec["pos_weight_split"] == "train", (
        "must match Phase 5 pos_weight_split"
    )


def test_spec_uses_bf16_for_rtx4080(spec: dict) -> None:
    """Per artifacts/audit/r2_gpu_design.md: RTX 4080 is BF16-native;
    use bf16 to match Phase 5 numerics. (Don't use fp16 — that'd be the
    V100 fallback path; mixing precision modes between Phase 5 and C1
    would invalidate the budget-only-difference claim.)"""
    assert spec["mixed_precision"] == "bf16", (
        "C1 spec must use bf16 (RTX 4080 native, matches Phase 5 LSTM)"
    )


def test_spec_cudnn_deterministic(spec: dict) -> None:
    """Same reproducibility contract as Phase 5."""
    assert spec["cudnn_deterministic"] is True


# ---------------------------------------------------------------------
# Runner supports --max-steps truncation mode
# ---------------------------------------------------------------------


def test_runner_supports_max_steps_arg() -> None:
    """The existing run_p1_centralized_lstm.py runs by --epochs. C1
    requires a --max-steps mode that truncates training at exactly
    25,000 gradient steps regardless of how many epochs that spans
    (here ~0.1 epoch). RED until the script gains the new arg."""
    if not RUNNER_PATH.exists():
        pytest.fail(f"runner missing: {RUNNER_PATH}")
    src = RUNNER_PATH.read_text()
    assert "--max-steps" in src, (
        "run_p1_centralized_lstm.py must accept --max-steps for C1; "
        "currently it only takes --epochs and runs full epochs."
    )


def test_runner_max_steps_takes_precedence_over_epochs() -> None:
    """When --max-steps is set, training must stop at exactly that many
    gradient steps and not run additional epochs. Pure source-level
    check — actual behavioural test deferred until Phase 2 GPU run."""
    if not RUNNER_PATH.exists():
        pytest.fail(f"runner missing: {RUNNER_PATH}")
    src = RUNNER_PATH.read_text()
    # We expect the script to break out of the epoch loop when step >= max_steps
    assert "max_steps" in src and "break" in src, (
        "runner must break out of training when --max-steps is reached; "
        "look for 'if step >= max_steps: break' or equivalent."
    )
