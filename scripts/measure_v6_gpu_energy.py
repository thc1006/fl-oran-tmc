"""Tier A.2 — Real GPU energy measurement via NVML for every v6 cell.

Per the 2026 ML.ENERGY blog (https://ml.energy/blog/energy/measurement/measuring-gpu-energy-best-practices/),
the Energy API (`nvmlDeviceGetTotalEnergyConsumption`, available on Volta
and newer; RTX 4080 = Ada qualifies) is preferred over the polling
approach because it is hardware-counted total energy in millijoules,
not a Riemann sum of polled wattage. We use the Energy API when
available and fall back to a 10 Hz Power-API polling loop otherwise.

Per-cell protocol:

1. Load `best_state.pt` for the cell.
2. Build the architecture (LSTM / Mamba / Spiking) with the same
   constructor kwargs the cell was trained under (parsed from
   ``summary.json``).
3. Move to GPU. ``torch.backends.cudnn.deterministic = True``,
   ``benchmark = False`` for reproducible per-batch timing.
4. Warm-up: 50 inference batches discarded (cuDNN heuristic finalisation
   + memory layout settling).
5. Lock GPU clock to base SM clock if `pynvml` permits — otherwise note
   in output that clocks are dynamic.
6. Snapshot total-energy counter T0 + wallclock t0.
7. Run N=2000 inference batches of size 64 (= 128 000 inferences total).
   Synchronise after each batch to ensure GPU work completes before
   the next iteration.
8. Snapshot T1 + t1.
9. Energy / inference = (T1 - T0) / 128000 in pJ (after unit conversion
   from mJ to pJ via ×1e9).
10. Subtract idle wattage (sampled before warmup) × wallclock as the
    "model-attributable" energy, so we do not credit the GPU's idle
    base load to the model.

Output: writes `energy_measured.json` next to existing `summary.json`
in each cell directory:

```
{
  "n_inferences_measured": 128000,
  "wallclock_sec": 4.2,
  "total_energy_mJ": 850.3,
  "model_attributable_energy_mJ": 412.1,
  "energy_pJ_per_inference_measured": 3.22e+09,
  "energy_pJ_per_inference_theoretical_sparsity_aware": 4.79e+05,
  "ratio_measured_to_theoretical": 6720,
  "gpu_clock_locked": false,
  "method": "EnergyAPI"  // or "PollingPowerAPI"
}
```

The ratio is the "GPU realisation factor" — typical values 10⁴-10⁵ for
sparse-friendly archs because GPUs do dense matmul regardless of input
sparsity.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from fl_oran.data_v2.encoders import apply_continuous_scaler, fit_continuous_scaler
from fl_oran.data_v2.sequences import build_run_sequences
from fl_oran.data_v2.split import ood_split_by_tr
from fl_oran.logging_utils import get_logger
from fl_oran.models.forecaster_v2 import ForecasterV2
from fl_oran.models.mamba_forecaster import MambaForecaster
from fl_oran.models.spiking_forecaster import SpikingForecaster
from fl_oran.training.centralized_v3 import V3Config, _load_and_prepare

# Local helper (single source of truth shared with recompute_v6_energy.py).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _v6_cell_metadata import (  # noqa: E402
    atomic_write_text,
    build_kwargs_from_suffix,
    parse_cell_dir as _shared_parse_cell_dir,
)

log = get_logger(__name__)

ARCH_CTOR = {
    "lstm": ForecasterV2,
    "mamba": MambaForecaster,
    "mamba_expand2": MambaForecaster,
    "spiking": SpikingForecaster,
    "spiking_expand2": SpikingForecaster,
}


def _try_import_pynvml():
    try:
        import pynvml
        return pynvml
    except ImportError:
        return None


def _parse_cell_dir(name: str) -> tuple[str, int, str]:
    """Wrapper around :func:`_v6_cell_metadata.parse_cell_dir`.

    Kept as a private name so existing imports (and the orchestrator
    log style) continue to work. The implementation lives in
    ``scripts/_v6_cell_metadata.py`` and is shared with
    ``recompute_v6_energy.py``.
    """
    return _shared_parse_cell_dir(name)


def _build_kwargs_from_suffix(arch_base: str, suffix: str) -> dict:
    """Wrapper around :func:`_v6_cell_metadata.build_kwargs_from_suffix`."""
    return build_kwargs_from_suffix(arch_base, suffix)


def _measure_energy_api(handle, pynvml, model, x_cat_gpu, x_cont_gpu,
                        n_inferences: int, batch_size: int, idle_w: float):
    """Use NVML Energy API (Volta+) for hardware-counted total energy.

    Wallclock-timing caveat: ``torch.cuda.synchronize()`` is called after
    every batch so the energy counter delta is correctly attributed to
    finished work. This serialises GPU execution and yields a wallclock
    that is **slower** than production async inference (which overlaps
    kernel queueing with kernel execution). The energy/inference reported
    here is therefore an **upper bound** on the energy a real deployment
    would see for the same model. We accept this in exchange for
    measurement determinism.
    """
    n_batches = n_inferences // batch_size
    # Warmup (cuDNN heuristic + memory layout)
    with torch.no_grad():
        for _ in range(50):
            _ = model(x_cat_gpu, x_cont_gpu)
    torch.cuda.synchronize()

    e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)  # millijoules
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_batches):
            _ = model(x_cat_gpu, x_cont_gpu)
            torch.cuda.synchronize()
    t1 = time.perf_counter()
    e1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)

    total_mJ = float(e1 - e0)
    wallclock = float(t1 - t0)
    idle_mJ = idle_w * 1000.0 * wallclock  # idle_w (W) × t (s) × 1000 ms/s = mJ
    model_mJ = max(total_mJ - idle_mJ, 0.0)
    return {
        "method": "EnergyAPI",
        "wallclock_sec": wallclock,
        "total_energy_mJ": total_mJ,
        "idle_attributed_mJ": idle_mJ,
        "model_attributable_energy_mJ": model_mJ,
        "energy_pJ_per_inference_total": total_mJ * 1e9 / (n_batches * batch_size),
        "energy_pJ_per_inference_model_only": model_mJ * 1e9 / (n_batches * batch_size),
        "n_batches": n_batches,
        "n_inferences_measured": n_batches * batch_size,
    }


def _measure_poll_api(handle, pynvml, model, x_cat_gpu, x_cont_gpu,
                      n_inferences: int, batch_size: int, sample_hz: float):
    """Fallback: poll Power API at sample_hz, integrate via Riemann sum."""
    import threading

    n_batches = n_inferences // batch_size
    samples: list[float] = []
    stop = threading.Event()

    def poll_loop():
        period = 1.0 / sample_hz
        while not stop.is_set():
            try:
                w_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                samples.append(float(w_mw) / 1000.0)
            except pynvml.NVMLError:
                pass
            time.sleep(period)

    with torch.no_grad():
        for _ in range(50):
            _ = model(x_cat_gpu, x_cont_gpu)
    torch.cuda.synchronize()

    t = threading.Thread(target=poll_loop, daemon=True)
    t.start()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_batches):
            _ = model(x_cat_gpu, x_cont_gpu)
            torch.cuda.synchronize()
    t1 = time.perf_counter()
    stop.set()
    t.join(timeout=2.0)
    wallclock = float(t1 - t0)
    avg_w = float(np.mean(samples)) if samples else 0.0
    total_J = avg_w * wallclock
    return {
        "method": f"PollingPowerAPI@{sample_hz:.0f}Hz",
        "wallclock_sec": wallclock,
        "avg_wattage_W": avg_w,
        "n_power_samples": len(samples),
        "total_energy_mJ": total_J * 1000.0,
        "energy_pJ_per_inference_total": total_J * 1e12 / (n_batches * batch_size),
        "n_batches": n_batches,
        "n_inferences_measured": n_batches * batch_size,
    }


def _measure_idle_wattage(handle, pynvml, seconds: float = 1.0,
                           sample_hz: float = 10.0) -> tuple[float, int]:
    """Returns ``(mean_wattage_W, n_samples)``.

    The caller MUST inspect ``n_samples`` — when NVML's power query is
    unsupported on a particular GPU/driver combo, ``samples`` ends up
    empty and the wattage defaults to 0. Silently using 0 as the idle
    baseline would attribute ALL measured energy (including the ~15 W
    idle floor) to the model and over-count its energy by 30-50 %.
    """
    samples: list[float] = []
    period = 1.0 / sample_hz
    end = time.time() + seconds
    while time.time() < end:
        try:
            w_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            samples.append(float(w_mw) / 1000.0)
        except pynvml.NVMLError:
            pass
        time.sleep(period)
    return (float(np.mean(samples)) if samples else 0.0, len(samples))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", type=str, default="artifacts/v6_arch_sweep")
    parser.add_argument("--n-inferences", type=int, default=128_000,
                        help="total inferences per cell measurement (= n_batches × batch_size)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sample-hz", type=float, default=10.0,
                        help="polling rate for the Power API fallback (ML.ENERGY recommends 10 Hz)")
    parser.add_argument("--max-cells", type=int, default=None,
                        help="optional cap for testing")
    parser.add_argument("--cell-glob", type=str, default="*_s*",
                        help="glob to filter which cells to measure (default all)")
    parser.add_argument("--force", action="store_true",
                        help="re-measure cells that already have an "
                             "energy_measured.json (skipped by default — "
                             "the measurement loop is the slow part of "
                             "Tier A.2 so we don't redo it on rerun)")
    args = parser.parse_args()

    # Hard requirement: PyTorch must see CUDA. NVML can succeed on a
    # headless CPU-only host where torch.cuda.is_available()==False, and
    # we need .cuda() below — fail loudly here rather than after data
    # loading wastes 30s.
    if not torch.cuda.is_available():
        log.error("torch.cuda.is_available() is False — this script needs a "
                  "CUDA device. Aborting.")
        sys.exit(2)
    # Guard against an operator picking --n-inferences < --batch-size, which
    # gives n_batches=0 and a downstream ZeroDivisionError in pJ-per-inf
    # computation.
    if args.n_inferences // args.batch_size <= 0:
        log.error(
            "--n-inferences (%d) must be at least --batch-size (%d) to "
            "produce at least one full batch.",
            args.n_inferences, args.batch_size,
        )
        sys.exit(2)

    pynvml = _try_import_pynvml()
    if pynvml is None:
        log.error("pynvml is not installed; run `uv pip install pynvml`")
        sys.exit(2)
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
    except pynvml.NVMLError as exc:
        log.error("NVML init failed (no NVIDIA driver or no GPU?): %s", exc)
        sys.exit(2)
    log.info("NVML init: GPU 0 = %s", name)

    # Attempt to lock GPU clock for reproducible per-cell timing/energy.
    # Per ML.ENERGY blog 2026: "lock clocks to remove variability across
    # measurement runs". Requires admin permissions on most systems.
    clock_locked = False
    try:
        # Get max graphics clock for the GPU; lock min=max=this value.
        max_clock = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
        # Use 80% of max as a stable, non-thermally-throttled point.
        target = int(max_clock * 0.8)
        pynvml.nvmlDeviceSetGpuLockedClocks(handle, target, target)
        clock_locked = True
        log.info("GPU clock locked at %d MHz (80%% of max %d MHz)", target, max_clock)
    except (pynvml.NVMLError, AttributeError) as exc:
        log.warning("GPU clock lock not supported / permission denied (%s); "
                    "measurements may have higher variance from boost-clock dynamics.", exc)

    # Detect Energy API support (Volta+; raises NVMLError on older)
    use_energy_api = True
    try:
        _ = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    except pynvml.NVMLError as exc:
        log.warning("Energy API unavailable (%s); falling back to polling.", exc)
        use_energy_api = False

    # Sample idle wattage as a baseline. Validate the GPU is actually idle
    # at startup — if some other process is using it, our "idle" reading is
    # contaminated and the model_attributable_mJ subtraction undercounts
    # model energy.
    idle_w, idle_n_samples = _measure_idle_wattage(
        handle, pynvml, seconds=2.0, sample_hz=args.sample_hz,
    )
    log.info("Idle wattage baseline: %.1f W (%d samples)", idle_w, idle_n_samples)
    if idle_n_samples == 0:
        log.warning(
            "Idle wattage sampling returned 0 samples — NVML power query is "
            "likely unsupported on this GPU/driver. idle_w defaults to 0 W, "
            "which means ALL energy will be attributed to the model and "
            "over-count its true cost by ~30-50%% (the GPU's idle floor).",
        )
    # On RTX 4080 idle is typically ~15-20 W. Anything > 50 W means the GPU
    # is doing something else (driver telemetry, another process, a
    # not-fully-released CUDA context). Warn loudly so the operator notices.
    if idle_w > 50.0:
        log.warning(
            "Idle wattage %.1f W is anomalously high (>50 W threshold); "
            "another process may be using the GPU. The model-attributable "
            "energy subtraction will UNDERCOUNT real model energy by an "
            "amount proportional to (idle_w − true_idle) × wallclock.",
            idle_w,
        )

    # Load data once for inference inputs.
    cfg = V3Config(
        unified_parquet=Path("data/coloran_raw_unified.parquet"),
        sample_ratio=1.0, seq_len=5, threshold=0.10,
    )
    df, schema = _load_and_prepare(cfg)
    feat_cols = schema.categorical + schema.continuous
    split = ood_split_by_tr(df, cfg.train_tr, cfg.val_tr, cfg.test_tr)
    X_tr, _ = build_run_sequences(split.train, feat_cols, ["y_sla_next"], seq_len=cfg.seq_len)
    X_te, _ = build_run_sequences(split.test, feat_cols, ["y_sla_next"], seq_len=cfg.seq_len)
    scaler = fit_continuous_scaler({0: X_tr}, schema)
    cat, cont = apply_continuous_scaler(X_te, schema, scaler)
    # Cap to args.batch_size; we re-use the same batch for every inference iteration.
    cat = cat[: args.batch_size]
    cont = cont[: args.batch_size]
    x_cat_gpu = torch.from_numpy(cat).cuda()
    x_cont_gpu = torch.from_numpy(cont).cuda()

    sweep_dir = Path(args.sweep_dir)
    cell_dirs = sorted(d for d in sweep_dir.glob(args.cell_glob) if d.is_dir())
    if args.max_cells:
        cell_dirs = cell_dirs[: args.max_cells]
    log.info("Measuring %d cells under %s", len(cell_dirs), sweep_dir)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    n_measured = 0
    n_skipped_already = 0
    n_skipped_no_state = 0
    n_skipped_unknown = 0
    n_failed = 0

    for cell_dir in cell_dirs:
        try:
            arch_base, seed, suffix = _parse_cell_dir(cell_dir.name)
        except (ValueError, IndexError) as exc:
            log.warning("skip un-parseable cell %s (%s)", cell_dir.name, exc)
            n_skipped_unknown += 1
            continue
        if arch_base not in ARCH_CTOR:
            log.warning("skip unknown arch %s in %s", arch_base, cell_dir)
            n_skipped_unknown += 1
            continue

        out_path = cell_dir / "energy_measured.json"
        if out_path.exists() and not args.force:
            log.info("[%s s=%d %s] energy_measured.json already exists; "
                     "skipping (use --force to remeasure)",
                     arch_base, seed, suffix or "-")
            n_skipped_already += 1
            continue

        best_state_path = cell_dir / "best_state.pt"
        if not best_state_path.exists():
            log.warning("[%s s=%d %s] no best_state.pt — SKIPPING (refusing to "
                        "measure a randomly-initialised model)",
                        arch_base, seed, suffix or "-")
            n_skipped_no_state += 1
            continue

        try:
            kwargs = _build_kwargs_from_suffix(arch_base, suffix)
            ctor = ARCH_CTOR[arch_base]
            model = ctor(schema=schema, task="classification",
                         seq_len=cfg.seq_len, **kwargs)
            model.load_state_dict(
                torch.load(best_state_path, map_location="cpu", weights_only=True)
            )
            model = model.cuda().eval()

            if use_energy_api:
                stats = _measure_energy_api(
                    handle, pynvml, model, x_cat_gpu, x_cont_gpu,
                    args.n_inferences, args.batch_size, idle_w,
                )
            else:
                stats = _measure_poll_api(
                    handle, pynvml, model, x_cat_gpu, x_cont_gpu,
                    args.n_inferences, args.batch_size, args.sample_hz,
                )

            # Cross-reference theoretical energy from existing energy.json.
            theoretical = {}
            e_path = cell_dir / "energy.json"
            if e_path.exists():
                try:
                    e_dict = json.loads(e_path.read_text())
                    theoretical = {
                        "energy_pJ_per_inference_theoretical_sparsity_aware":
                            e_dict.get("total_energy_pJ_sparsity_aware",
                                       e_dict.get("total_energy_pJ", 0.0)),
                        "energy_pJ_per_inference_theoretical_gpu_dense":
                            e_dict.get("total_energy_pJ_gpu_dense",
                                       e_dict.get("total_energy_pJ", 0.0)),
                    }
                except json.JSONDecodeError:
                    pass
            out = {
                "arch_base": arch_base,
                "seed": seed,
                "suffix": suffix,
                "gpu_name": name if isinstance(name, str) else name.decode(
                    "utf-8", errors="replace"
                ),
                "idle_wattage_W": idle_w,
                "gpu_clock_locked": clock_locked,
                **stats,
                **theoretical,
            }
            if "energy_pJ_per_inference_theoretical_sparsity_aware" in out and \
                    out.get("energy_pJ_per_inference_total"):
                out["ratio_measured_to_theoretical_sparsity"] = (
                    out["energy_pJ_per_inference_total"] /
                    out["energy_pJ_per_inference_theoretical_sparsity_aware"]
                )
            if "energy_pJ_per_inference_theoretical_gpu_dense" in out and \
                    out.get("energy_pJ_per_inference_total"):
                out["ratio_measured_to_theoretical_gpu_dense"] = (
                    out["energy_pJ_per_inference_total"] /
                    out["energy_pJ_per_inference_theoretical_gpu_dense"]
                )

            atomic_write_text(out_path, json.dumps(out, indent=2))
            log.info(
                "[%s s=%d %s] measured=%.2e pJ/inf  theoretical_sparsity=%.2e  "
                "ratio=%.0fx",
                arch_base, seed, suffix or "-",
                out.get("energy_pJ_per_inference_total", 0.0),
                out.get("energy_pJ_per_inference_theoretical_sparsity_aware", 0.0),
                out.get("ratio_measured_to_theoretical_sparsity", 0.0),
            )
            n_measured += 1
        except Exception as exc:
            # Per-cell error isolation: one bad cell (state-dict mismatch,
            # OOM, etc.) must not abort the whole 150-cell sweep.
            log.error("[%s s=%d %s] measurement FAILED: %s\n%s",
                      arch_base, seed, suffix or "-", exc,
                      traceback.format_exc())
            n_failed += 1
            # Free GPU memory before continuing to the next cell.
            try:
                del model
            except UnboundLocalError:
                pass
            torch.cuda.empty_cache()
            continue

    log.info(
        "measurement summary: measured=%d skipped_already=%d "
        "skipped_no_state=%d skipped_unknown=%d failed=%d",
        n_measured, n_skipped_already, n_skipped_no_state,
        n_skipped_unknown, n_failed,
    )

    # Release any clock lock so subsequent users don't inherit the lock.
    # `try/finally` is intentionally outside the cell loop — even if every
    # cell crashed we still want the clock lock released.
    if clock_locked:
        try:
            pynvml.nvmlDeviceResetGpuLockedClocks(handle)
            log.info("GPU clock lock released")
        except (pynvml.NVMLError, AttributeError):
            pass
    pynvml.nvmlShutdown()

    if n_failed:
        # Signal failure to the orchestrator (set -e).
        sys.exit(1)


if __name__ == "__main__":
    main()
