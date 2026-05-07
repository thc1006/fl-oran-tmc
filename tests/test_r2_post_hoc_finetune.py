"""R2 C3 — post-hoc per-BS fine-tune methodology + script invariants.

TDD-RED before C3 GREEN: assert the per-BS fine-tune script exists with
the expected CLI signature and produces a per-BS-personalised-vs-global
AUC JSON. Tests fail until script lands.

Reviewer R2 #5 / A2 (FedBN spirit): our paper's mechanism story
attributes the natural-by-BS dominance to per-BS feature shift (CQI /
MCS distribution differences). FedBN's design literally targets feature
shift; we argued FedBN reduces to FedAvg on no-BN backbones (true) but
that argument satisfies the *letter* not the *spirit* of MC3.

C3 fills the spirit: take a global Phase-5 FedAvg checkpoint, fine-tune
it on each BS's local train data, and compare per-BS personalised AUC
against the global model. If personalisation gives < +0.005 AUC: our
"feature-shift personalisation gives little" claim is empirically
strengthened. If > +0.01: the paper needs a new caveat acknowledging
local personalisation does help on this dataset.

Hardware: V100 cluster (4 cards × 4 concurrent cells/card oversubscribed)
per artifacts/audit/r2_gpu_design.md. ~30-40 min wall on V100.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "experiments" / "run_r2_post_hoc_per_bs_finetune.py"
LAUNCHER_PATH = REPO_ROOT / "scripts" / "v100_r2_c3_launcher.sh"


# ---------------------------------------------------------------------
# Script existence + CLI signature
# ---------------------------------------------------------------------


def test_script_exists() -> None:
    """RED until C3 script lands."""
    assert SCRIPT_PATH.exists(), (
        f"R2 C3 script missing: {SCRIPT_PATH}. Implement per "
        "artifacts/audit/r2_gpu_design.md C3 section."
    )


def test_script_help_lists_required_args() -> None:
    """The script must accept --cells / --device / --finetune-steps /
    --batch-size / --mixed-precision / --out per the launcher contract.
    Source-level check — actual runtime test deferred until Phase 2 GPU."""
    if not SCRIPT_PATH.exists():
        pytest.skip("script not yet implemented (R2 C3-GREEN)")
    src = SCRIPT_PATH.read_text()
    required_args = (
        "--cells",
        "--device",
        "--finetune-steps",
        "--batch-size",
        "--mixed-precision",
        "--out",
    )
    missing = [a for a in required_args if a not in src]
    assert not missing, (
        f"R2 C3 script must declare CLI args: {required_args}. "
        f"Missing: {missing}"
    )


def test_script_loads_phase5_checkpoint_path() -> None:
    """The script must load checkpoints from Phase 5 stage-2 sweep
    using the canonical naming convention (artifacts/v7_stage2_full/
    v7_<arch>_<algo>_iid_n7_s<seed>/best.pt). RED until script lands."""
    if not SCRIPT_PATH.exists():
        pytest.skip("script not yet implemented (R2 C3-GREEN)")
    src = SCRIPT_PATH.read_text()
    assert "v7_stage2_full" in src, (
        "R2 C3 script must reference v7_stage2_full checkpoint dir"
    )
    assert "best.pt" in src, "must load best.pt checkpoints"


def test_script_outputs_per_bs_personalised_vs_global() -> None:
    """Output JSON must contain per-BS personalised + global AUC so the
    paper §8 L2 can quantify the personalisation lift. RED until script
    documents this in its module docstring or output schema comment."""
    if not SCRIPT_PATH.exists():
        pytest.skip("script not yet implemented (R2 C3-GREEN)")
    src = SCRIPT_PATH.read_text()
    # Must reference both keys in output construction
    assert "personalised" in src or "personalized" in src, (
        "output JSON must distinguish personalised vs global per-BS AUC"
    )
    assert "global" in src.lower(), (
        "output JSON must include global per-BS AUC (the FedAvg baseline)"
    )


# ---------------------------------------------------------------------
# Launcher invariants
# ---------------------------------------------------------------------


def test_launcher_exists() -> None:
    """V100 4-way parallel launcher per artifacts/audit/r2_gpu_design.md."""
    assert LAUNCHER_PATH.exists(), (
        f"R2 C3 V100 launcher missing: {LAUNCHER_PATH}. "
        f"See artifacts/audit/r2_gpu_design.md C3 section for spec."
    )


def test_launcher_uses_4_v100_cards() -> None:
    """The launcher must distribute the 105 cells across the 4 V100 cards
    (CUDA_VISIBLE_DEVICES + GPU_NUM env). RED until launcher lands."""
    if not LAUNCHER_PATH.exists():
        pytest.skip("launcher not yet implemented (R2 C3-GREEN)")
    src = LAUNCHER_PATH.read_text()
    assert "CUDA_VISIBLE_DEVICES" in src, (
        "launcher must set per-card CUDA_VISIBLE_DEVICES"
    )
    # Reference the 4-card structure
    has_4_cards = any(t in src for t in ("for gpu in 0 1 2 3", "GPUS=(0 1 2 3)",
                                          "0 1 2 3", "card 0", "card 1"))
    assert has_4_cards, (
        "launcher must reference all 4 V100 cards (look for GPU 0..3 loop)"
    )


def test_launcher_uses_fp16_for_v100() -> None:
    """Per artifacts/audit/r2_gpu_design.md: V100 BF16 is emulated; use
    FP16 native. Launcher must pass --mixed-precision fp16."""
    if not LAUNCHER_PATH.exists():
        pytest.skip("launcher not yet implemented (R2 C3-GREEN)")
    src = LAUNCHER_PATH.read_text()
    assert "fp16" in src, (
        "V100 launcher must use --mixed-precision fp16 (not bf16; "
        "bf16 is software-emulated on V100 sm_70 → 2× slower)"
    )
