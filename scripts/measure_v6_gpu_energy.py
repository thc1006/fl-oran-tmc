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
import time
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
    """Returns (arch_base, seed, suffix). Mirrors aggregator parser."""
    arch, _, rest = name.partition("_s")
    seed_part, _, suffix = rest.partition("_")
    return arch, int(seed_part), suffix


def _build_kwargs_from_suffix(arch_base: str, suffix: str) -> dict:
    """Reconstruct the kwargs the cell was trained with from its suffix."""
    kwargs: dict = {}
    if arch_base == "spiking":
        if "t5" in suffix:
            kwargs["t_inner"] = 5
        if "t5sum" in suffix:
            kwargs["t_inner"] = 5
            kwargs["decode_mode"] = "sum"
        # LIF threshold/beta from suffixes like "lif_t05_b09".
        if "lif_t" in suffix:
            try:
                t_str = suffix.split("lif_t")[1].split("_")[0]
                kwargs["lif_threshold"] = int(t_str) / 10.0
            except (IndexError, ValueError):
                pass
        if "_b" in suffix and "lif_t" in suffix:
            try:
                b_str = suffix.split("_b")[-1].split("_")[0]
                # b05 → 0.5, b09 → 0.9, b099 → 0.99
                if len(b_str) == 2:
                    kwargs["lif_beta"] = int(b_str) / 10.0
                elif len(b_str) == 3:
                    kwargs["lif_beta"] = int(b_str) / 100.0
            except (IndexError, ValueError):
                pass
    if arch_base == "mamba_expand2":
        # Hardcoded to match ARCH_REGISTRY["mamba_expand2"] in
        # experiments/run_v6_arch_sweep.py. If that registry entry
        # changes, this must change too.
        kwargs.update({"backbone_d_model": 48, "backbone_expand": 2, "n_blocks": 2})
    if arch_base == "spiking_expand2":
        # Mirrors ARCH_REGISTRY["spiking_expand2"]: d_model=56, expand=2, t_inner=1
        # gives ~43.6K params (-2.2% vs LSTM). Cell-trained kwargs are recovered
        # here exactly so the loaded best_state.pt has matching shapes.
        kwargs.update({"backbone_d_model": 56, "backbone_expand": 2, "t_inner": 1})
    return kwargs


def _measure_energy_api(handle, pynvml, model, x_cat_gpu, x_cont_gpu,
                        n_inferences: int, batch_size: int, idle_w: float):
    """Use NVML Energy API (Volta+) for hardware-counted total energy."""
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
                           sample_hz: float = 10.0) -> float:
    samples = []
    period = 1.0 / sample_hz
    end = time.time() + seconds
    while time.time() < end:
        try:
            w_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            samples.append(float(w_mw) / 1000.0)
        except pynvml.NVMLError:
            pass
        time.sleep(period)
    return float(np.mean(samples)) if samples else 0.0


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
    args = parser.parse_args()

    pynvml = _try_import_pynvml()
    if pynvml is None:
        log.error("pynvml is not installed; run `uv pip install pynvml`")
        return
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
    except pynvml.NVMLError as exc:
        log.error("NVML init failed (no NVIDIA driver or no GPU?): %s", exc)
        return
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

    # Sample idle wattage as a baseline.
    idle_w = _measure_idle_wattage(handle, pynvml, seconds=2.0, sample_hz=args.sample_hz)
    log.info("Idle wattage baseline: %.1f W", idle_w)

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

    for cell_dir in cell_dirs:
        try:
            arch_base, seed, suffix = _parse_cell_dir(cell_dir.name)
        except (ValueError, IndexError):
            log.warning("skip un-parseable cell name: %s", cell_dir.name)
            continue
        if arch_base not in ARCH_CTOR:
            log.warning("skip unknown arch %s in %s", arch_base, cell_dir)
            continue
        kwargs = _build_kwargs_from_suffix(arch_base, suffix)
        ctor = ARCH_CTOR[arch_base]
        model = ctor(schema=schema, task="classification", seq_len=cfg.seq_len, **kwargs)

        best_state_path = cell_dir / "best_state.pt"
        if not best_state_path.exists():
            log.warning("[%s s=%d %s] no best_state.pt — SKIPPING (refusing to "
                        "measure a randomly-initialised model)",
                        arch_base, seed, suffix or "-")
            continue
        model.load_state_dict(torch.load(best_state_path, map_location="cpu", weights_only=True))
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
            "gpu_name": name if isinstance(name, str) else name.decode("utf-8", errors="replace"),
            "idle_wattage_W": idle_w,
            "gpu_clock_locked": clock_locked,
            **stats,
            **theoretical,
        }
        if "energy_pJ_per_inference_theoretical_sparsity_aware" in out and out.get("energy_pJ_per_inference_total"):
            out["ratio_measured_to_theoretical_sparsity"] = (
                out["energy_pJ_per_inference_total"] /
                out["energy_pJ_per_inference_theoretical_sparsity_aware"]
            )
        if "energy_pJ_per_inference_theoretical_gpu_dense" in out and out.get("energy_pJ_per_inference_total"):
            out["ratio_measured_to_theoretical_gpu_dense"] = (
                out["energy_pJ_per_inference_total"] /
                out["energy_pJ_per_inference_theoretical_gpu_dense"]
            )

        (cell_dir / "energy_measured.json").write_text(json.dumps(out, indent=2))
        log.info("[%s s=%d %s] measured=%.2e pJ/inf  theoretical_sparsity=%.2e  ratio=%.0fx",
                 arch_base, seed, suffix or "-",
                 out.get("energy_pJ_per_inference_total", 0.0),
                 out.get("energy_pJ_per_inference_theoretical_sparsity_aware", 0.0),
                 out.get("ratio_measured_to_theoretical_sparsity", 0.0))

    # Release any clock lock so subsequent users don't inherit the lock.
    if clock_locked:
        try:
            pynvml.nvmlDeviceResetGpuLockedClocks(handle)
            log.info("GPU clock lock released")
        except (pynvml.NVMLError, AttributeError):
            pass
    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
