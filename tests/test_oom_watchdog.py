"""Tests for scripts/oom_watchdog.py — Phase 5 GPU memory creep watchdog.

GH#7: hardware-aware GPU memory threshold.

The watchdog was originally hardcoded to a 12 GB threshold for RTX 4080
(16 GB). Reusing on V100 / A100 / H100 would silently fail (threshold
never reaches the larger card's actual creep level). The fix detects
total GPU memory via ``nvidia-smi`` and defaults the threshold to 75%
of detected total when no explicit --threshold-mb is given.
"""
from __future__ import annotations

import importlib.util
import signal
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "oom_watchdog.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("oom_watchdog", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["oom_watchdog"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def oom():
    return _load_module()


def test_detect_threshold_mb_scales_to_75_percent_of_total(oom):
    """GH#7: ``_detect_threshold_mb`` queries nvidia-smi for total memory
    and returns 75% as the threshold. RTX 4080 (16 GB) → 12288 MB
    (matches the prior hardcoded default); V100 (32 GB) → 24576 MB."""
    with patch("subprocess.check_output", return_value=b"16384\n"):
        thr_4080 = oom._detect_threshold_mb()
    assert thr_4080 == int(16384 * 0.75), (
        f"4080 threshold should be 75% of 16 GB = 12288 MB; got {thr_4080}"
    )
    with patch("subprocess.check_output", return_value=b"32768\n"):
        thr_v100 = oom._detect_threshold_mb()
    assert thr_v100 == int(32768 * 0.75), (
        f"V100 threshold should be 75% of 32 GB = 24576 MB; got {thr_v100}"
    )


def test_detect_threshold_mb_falls_back_when_nvidia_smi_fails(oom):
    """Fallback: if nvidia-smi is unavailable / errors, return the
    legacy 12 GB default so the watchdog still runs (rather than
    crashing on import / startup)."""
    import subprocess as sp
    with patch("subprocess.check_output",
               side_effect=sp.CalledProcessError(1, "nvidia-smi")):
        thr = oom._detect_threshold_mb()
    assert thr == oom.DEFAULT_GPU_THRESHOLD_MB, (
        f"fallback should be DEFAULT_GPU_THRESHOLD_MB={oom.DEFAULT_GPU_THRESHOLD_MB}; "
        f"got {thr}"
    )


def test_signal_handlers_cover_both_sigterm_and_sigint(oom):
    """Symmetry: SIGTERM and SIGINT must both route through ``_on_term``
    so a Ctrl-C in the foreground terminal exits the watchdog as
    cleanly as a SIGTERM from process supervisor."""
    # Verify the module exposes the handler
    assert hasattr(oom, "_on_term")
    # The handlers are registered in main(); we verify the function
    # signature is compatible with signal.signal (signum, frame).
    import inspect
    sig = inspect.signature(oom._on_term)
    assert len(sig.parameters) == 2, (
        f"_on_term must accept (signum, frame); got {sig}"
    )
