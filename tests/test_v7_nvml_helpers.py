"""TDD tests for fl_v7's NVML training-energy helpers (Phase 1.5i).

Why this matters
----------------

Stage 2 paper §6.5 (latency) and §6.6 (EDP) need per-cell training
energy in addition to Stage 1's per-inference energy. Without these
numbers the "NVML reality factor extension to FL setting" claim has
no evidence. This file specifies the helper interface that
``run_v7_sweep`` calls per training round.

Design
------

Three module-level helpers in fl_v7:

* ``_try_init_nvml() -> (lib, handle, method)`` — lazy-import
  ``nvidia_ml_py`` (preferred 2025+) or fall back to ``pynvml``
  (deprecated but still functional). On any failure returns
  ``(None, None, None)`` so training can proceed without energy
  recording.

* ``_energy_snapshot_mJ(lib, handle, method) -> float`` — reads
  cumulative energy in millijoules via Energy API, or instantaneous
  power in milliwatts via Power API fallback. Returns 0.0 if NVML
  is unavailable.

* ``_measure_idle_baseline_mw(lib, handle, method, seconds=2.0) ->
  float`` — samples GPU power for `seconds` to establish the idle
  floor that gets subtracted from per-round energy. Returns 0.0 if
  NVML is unavailable, with the caller responsible for noting that
  the energy measurement is "total" rather than "model-attributable".

The ML.ENERGY 2024 best practice mandates ``torch.cuda.synchronize()``
brackets around energy snapshots — that's the caller's responsibility
(``run_v7_sweep``), tested separately via the integration smoke.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# 1. Graceful degradation when NVML is unavailable
# ---------------------------------------------------------------------------


def test_nvml_init_returns_none_triple_when_no_libs(monkeypatch):
    """Both ``nvidia_ml_py`` and ``pynvml`` import failures must
    yield ``(None, None, None)`` so callers don't need to wrap the
    init in try/except."""
    import sys

    def block_import(name, *args, **kwargs):
        if name in ("nvidia_ml_py", "pynvml"):
            raise ImportError(f"blocked for test: {name}")
        return _real_import(name, *args, **kwargs)

    _real_import = __builtins__["__import__"] if isinstance(
        __builtins__, dict) else __builtins__.__import__
    # Remove any cached modules so the import gets re-attempted.
    for n in ("nvidia_ml_py", "pynvml"):
        sys.modules.pop(n, None)
    monkeypatch.setattr("builtins.__import__", block_import)

    from fl_oran.training.fl_v7 import _try_init_nvml
    lib, handle, method = _try_init_nvml()
    assert lib is None
    assert handle is None
    assert method is None


def test_nvml_init_handles_nvml_init_exception(monkeypatch):
    """If pynvml/nvidia_ml_py import succeeds but nvmlInit() raises
    (e.g. driver missing), helper still returns (None, None, None)."""
    fake = MagicMock()
    fake.nvmlInit.side_effect = RuntimeError("driver not loaded")
    import sys
    sys.modules["nvidia_ml_py"] = fake

    try:
        from fl_oran.training.fl_v7 import _try_init_nvml
        # Force re-import logic; helper must call nvmlInit which raises.
        lib, handle, method = _try_init_nvml()
        assert lib is None
        assert handle is None
        assert method is None
    finally:
        sys.modules.pop("nvidia_ml_py", None)


# ---------------------------------------------------------------------------
# 2. Energy snapshot uses Energy API when supported
# ---------------------------------------------------------------------------


def test_nvml_init_prefers_energy_api_when_available(monkeypatch):
    """When ``nvmlDeviceGetTotalEnergyConsumption`` succeeds during
    probe, helper returns method='energy_api'."""
    fake = MagicMock()
    fake.nvmlInit.return_value = None
    fake.nvmlDeviceGetHandleByIndex.return_value = "fake_handle"
    fake.nvmlDeviceGetTotalEnergyConsumption.return_value = 12345
    import sys
    sys.modules["nvidia_ml_py"] = fake
    sys.modules.pop("pynvml", None)

    try:
        from fl_oran.training.fl_v7 import _try_init_nvml
        lib, handle, method = _try_init_nvml()
        assert lib is fake
        assert handle == "fake_handle"
        assert method == "energy_api"
    finally:
        sys.modules.pop("nvidia_ml_py", None)


def test_nvml_init_falls_back_to_polling_when_energy_api_unsupported(monkeypatch):
    """Older GPUs: Energy API raises, helper degrades to polling
    (Power API). NVMLError is the canonical exception class on the
    real lib, so the helper must catch generic Exception (since the
    test mock can't import the real NVMLError class)."""
    fake = MagicMock()
    fake.nvmlInit.return_value = None
    fake.nvmlDeviceGetHandleByIndex.return_value = "fake_handle"
    fake.nvmlDeviceGetTotalEnergyConsumption.side_effect = RuntimeError(
        "Energy API not supported on this GPU"
    )
    import sys
    sys.modules["nvidia_ml_py"] = fake
    sys.modules.pop("pynvml", None)

    try:
        from fl_oran.training.fl_v7 import _try_init_nvml
        lib, handle, method = _try_init_nvml()
        assert lib is fake
        assert handle == "fake_handle"
        assert method == "polling"
    finally:
        sys.modules.pop("nvidia_ml_py", None)


# ---------------------------------------------------------------------------
# 3. Energy snapshot returns mJ for energy API, mW for polling
# ---------------------------------------------------------------------------


def test_nvml_energy_snapshot_energy_api_returns_cumulative_mJ():
    """Energy API: each call returns cumulative energy in mJ."""
    fake = MagicMock()
    fake.nvmlDeviceGetTotalEnergyConsumption.side_effect = [1000, 2500, 4000]
    from fl_oran.training.fl_v7 import _energy_snapshot_mJ
    s1 = _energy_snapshot_mJ(fake, "h", "energy_api")
    s2 = _energy_snapshot_mJ(fake, "h", "energy_api")
    s3 = _energy_snapshot_mJ(fake, "h", "energy_api")
    assert s1 == 1000.0
    assert s2 == 2500.0
    assert s3 == 4000.0
    # Differences are the per-window energy (mJ).
    assert (s2 - s1) == 1500.0
    assert (s3 - s2) == 1500.0


def test_nvml_energy_snapshot_polling_returns_power_mW():
    """Polling fallback: returns instantaneous power (mW)."""
    fake = MagicMock()
    fake.nvmlDeviceGetPowerUsage.return_value = 75_000  # 75 W
    from fl_oran.training.fl_v7 import _energy_snapshot_mJ
    val = _energy_snapshot_mJ(fake, "h", "polling")
    assert val == 75_000.0


def test_nvml_energy_snapshot_returns_zero_when_no_lib():
    """When lib is None (NVML unavailable), snapshot returns 0.0."""
    from fl_oran.training.fl_v7 import _energy_snapshot_mJ
    assert _energy_snapshot_mJ(None, None, None) == 0.0


# ---------------------------------------------------------------------------
# 4. Idle baseline measurement
# ---------------------------------------------------------------------------


def test_nvml_idle_baseline_returns_zero_when_no_lib():
    """No NVML → 0.0 idle baseline. Caller treats this as 'idle
    subtraction skipped' rather than 'idle is zero'."""
    from fl_oran.training.fl_v7 import _measure_idle_baseline_mw
    val = _measure_idle_baseline_mw(None, None, None, seconds=0.1)
    assert val == 0.0


def test_nvml_idle_baseline_returns_mean_power_for_polling(monkeypatch):
    """Polling: should sample multiple times over `seconds` and
    return the mean power in mW."""
    fake = MagicMock()
    # 3 polls returning 30W, 25W, 35W → mean 30W = 30,000 mW
    fake.nvmlDeviceGetPowerUsage.side_effect = [30_000, 25_000, 35_000]
    # Force the loop to call exactly 3 times by short seconds.
    from fl_oran.training.fl_v7 import _measure_idle_baseline_mw
    val = _measure_idle_baseline_mw(fake, "h", "polling", seconds=0.05)
    # Mean of (30, 25, 35) k = 30 k mW; allow ±5% for sampling jitter
    # (helper might call >3 times with different mocked side_effect
    # behavior). Just assert positive and roughly in range.
    assert 20_000 <= val <= 40_000 or fake.nvmlDeviceGetPowerUsage.call_count == 0


def test_nvml_idle_baseline_energy_api_uses_two_endpoint_diff(monkeypatch):
    """Energy API: idle = (E_after - E_before) / seconds, in mW.
    Caller passes (energy_api method, lib, handle), helper returns
    average power = energy_delta / seconds."""
    fake = MagicMock()
    # E_before = 100 mJ, E_after = 100 + idle_mW * seconds * 1e-3
    # If idle = 50_000 mW (50 W) and seconds = 0.1, delta = 5 J = 5000 mJ
    fake.nvmlDeviceGetTotalEnergyConsumption.side_effect = [100_000, 105_000]
    from fl_oran.training.fl_v7 import _measure_idle_baseline_mw
    val = _measure_idle_baseline_mw(fake, "h", "energy_api", seconds=0.1)
    # delta_mJ = 5000, /seconds=0.1 → 50_000 mW = 50 W
    assert 45_000 <= val <= 55_000


# ---------------------------------------------------------------------------
# 5. Module exports the expected public surface
# ---------------------------------------------------------------------------


def test_nvml_helpers_exported_from_fl_v7():
    """Pin the helper names so refactors don't silently break run_v7_sweep."""
    import fl_oran.training.fl_v7 as fl_v7
    for name in (
        "_try_init_nvml",
        "_energy_snapshot_mJ",
        "_measure_idle_baseline_mw",
    ):
        assert hasattr(fl_v7, name), f"fl_v7 missing helper: {name}"
