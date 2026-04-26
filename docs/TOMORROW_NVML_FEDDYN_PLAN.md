# Tomorrow (2026-04-28): NVML + FedDyn integration plan

**Goal**: get fl_v7 paper-ready before Phase 5 launch.

**Estimated effort**: 4-6 hours total (3 hr NVML + 1 hr FedDyn + 1 hr tests + 1 hr smoke).

---

## Part 1: NVML training-energy integration (Phase 1.5i, task #148)

### Reference implementation

`scripts/measure_v6_gpu_energy.py` already implements the Energy API pattern. We port to fl_v7 training, with these adaptations:

| Stage 1 (centralized) | Stage 2 (FL) | Adaptation |
|----------------------|---------------|------------|
| Per-inference energy | Per-round energy | Sample at round boundaries |
| One-shot model | Per-round client + aggregate work | Sum client local steps + aggregation |
| 2000-batch warmup | Each round = ~250 grads | Skip warmup; subtract idle baseline |
| `nvmlDeviceGetTotalEnergyConsumption` | Same | Same Energy API call, different boundaries |

### Design

```python
# Add to fl_v7.py module level (after imports):
from contextlib import contextmanager

def _try_init_nvml():
    """Lazy-init pynvml; return (handle, available_bool)."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        # Probe Energy API support (Volta+; RTX 4080 sm_89 supports it)
        try:
            _ = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
            return pynvml, handle, "energy_api"
        except pynvml.NVMLError:
            return pynvml, handle, "polling"
    except (ImportError, Exception) as e:
        log.warning("NVML unavailable: %s; energy will not be recorded", e)
        return None, None, None


def _energy_snapshot(pynvml, handle, method):
    """Return current cumulative energy in millijoules (Energy API)
    or instantaneous power in milliwatts (polling)."""
    if method == "energy_api":
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    return pynvml.nvmlDeviceGetPowerUsage(handle)


def _measure_idle_baseline(pynvml, handle, method, seconds=2.0):
    """Sample idle wattage for baseline subtraction."""
    if method == "energy_api":
        e0 = _energy_snapshot(pynvml, handle, method)
        time.sleep(seconds)
        e1 = _energy_snapshot(pynvml, handle, method)
        return (e1 - e0) / seconds  # mJ/s = mW
    # polling: sample N times, take mean
    samples = []
    t_end = time.time() + seconds
    while time.time() < t_end:
        samples.append(_energy_snapshot(pynvml, handle, method))
        time.sleep(0.05)
    return float(sum(samples) / max(len(samples), 1))  # mW
```

### Integration in `run_v7_sweep`

```python
# After setup_torch_perf:
pynvml, nvml_handle, nvml_method = _try_init_nvml()
energy_per_round_mJ: list[float] = []
idle_mw = None
if pynvml is not None:
    idle_mw = _measure_idle_baseline(pynvml, nvml_handle, nvml_method)

# Inside the for-r loop, around the round body:
e_round_start = (
    _energy_snapshot(pynvml, nvml_handle, nvml_method)
    if pynvml else None
)

# ... existing client_update + aggregate ...

if pynvml is not None and e_round_start is not None:
    e_round_end = _energy_snapshot(pynvml, nvml_handle, nvml_method)
    if nvml_method == "energy_api":
        # mJ direct from cumulative counter
        round_energy_mJ = float(e_round_end - e_round_start)
    else:
        # polling: approximate with mean wattage × round duration
        round_energy_mJ = float((e_round_start + e_round_end) / 2 * dt)
    # Subtract idle baseline (mW × s = mJ)
    idle_mJ = idle_mw * dt
    energy_per_round_mJ.append(round_energy_mJ - idle_mJ)

# After training loop:
if pynvml is not None:
    pynvml.nvmlShutdown()
    result["energy_measured"] = {
        "method": nvml_method,
        "idle_baseline_mW": idle_mw,
        "per_round_mJ": energy_per_round_mJ,
        "total_training_mJ": sum(energy_per_round_mJ),
        "mean_round_mJ": sum(energy_per_round_mJ) / max(len(energy_per_round_mJ), 1),
    }
```

### Edge cases

1. **Energy API not available** (very old GPU, no driver): fall back to polling → less accurate but still works
2. **pynvml not installed**: log warning, return early, no energy recorded — Phase 5 still produces AUC
3. **Multi-GPU systems**: use `device.index` instead of hard-coded 0
4. **Concurrent processes on same GPU**: idle_mw includes their load → subtraction biased. Note in §limitations.

### Tests to add

- `tests/test_v7_nvml_integration.py`:
  - `test_nvml_helper_handles_missing_pynvml()` — mock ImportError, ensure no crash
  - `test_nvml_energy_method_returns_mJ()` — mock pynvml, verify mJ unit
  - `test_run_v7_sweep_records_energy_when_nvml_available()` — mock pynvml, run mini-sweep, check `summary.json["energy_measured"]` exists

### Smoke test before Phase 5

```bash
# 1-cell smoke with NVML to verify reasonable numbers
python experiments/run_v7_phase_sweep.py --spec experiments/specs/phase2_min.yaml \
    --output-dir artifacts/v7_nvml_smoke --summary-tag nvml_smoke --limit 1
# Expect: total_training_mJ in 1000-10000 range (1-10 J), idle_baseline_mW ~25-50 W
```

---

## Part 2: FedDyn server step h_accum correction (Phase 1.5j, task #149)

### The fix

`src/fl_oran/federated/algorithms/feddyn.py:249-254` currently has:
```python
# TODO(M3): paper-faithful server step is
#   new_w[name] += h_accum[name] / (alpha * N)
# where N is the total number of clients ever seen. ...
# Today we return plain FedAvg weights and keep h_accum as a side accumulator
return new_w
```

### Implementation

Add `n_total_clients` to FedDyn `__init__` and apply in `server_aggregate`:

```python
def __init__(
    self,
    *,
    max_steps: int,
    batch_size: int,
    alpha: float,
    update_mode: str = "option_ii",
    grad_clip: float = 1.0,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype | None = None,
    n_total_clients: int = 1,  # NEW: paper-faithful server step
) -> None:
    ...
    self.n_total_clients = int(n_total_clients)
    if self.n_total_clients < 1:
        raise ValueError(f"n_total_clients must be >= 1, got {n_total_clients}")
    ...

def server_aggregate(...) -> dict[str, torch.Tensor]:
    ...
    # Accumulate delta_h_i ... (existing code)
    for name in self.h_accum:
        for d in deltas:
            if name in d:
                self.h_accum[name] = self.h_accum[name] + d[name]

    # NEW: apply paper-faithful h_accum correction.
    # new_w[name] += h_accum[name] / (alpha * N)
    # Skip if alpha=0 (degenerate to FedAvg).
    if self.alpha != 0.0:
        for name in new_w:
            if name in self.h_accum:
                correction = self.h_accum[name].to(new_w[name].device) / (
                    self.alpha * self.n_total_clients
                )
                new_w[name] = new_w[name] + correction
    return new_w
```

### Wire through fl_v7

In `run_v7_sweep`, when instantiating FedDyn:
```python
algo_kwargs.update(cfg.algo_kwargs)
# Inject orchestrator-controlled n_total_clients for FedDyn server step.
if cfg.algorithm == "feddyn" and "n_total_clients" not in algo_kwargs:
    algo_kwargs["n_total_clients"] = cfg.n_clients
algo_inst = algo_cls(**algo_kwargs)
```

### Update `_ALGO_REQUIRED_KWARGS`

`alpha` stays required. `n_total_clients` is auto-injected, so does NOT add to the required set. But the regression test `test_declared_required_kwargs_are_actually_required` will fail if I forget — the new param has a default (1), so signature check passes.

Actually wait, signature inspection in `_AUTO_FILLED_BY_FL_V7` only includes `{max_steps, batch_size, grad_clip, amp_enabled, amp_dtype}`. `n_total_clients` is not in this set. So if a user-spec doesn't provide `n_total_clients`, the existing test `test_all_truly_required_kwargs_are_declared` would be checking — but `n_total_clients` has a default (`= 1`) so it's not in `truly_required`. Test passes.

But: spec validation would let `n_total_clients` be either user-provided OR auto-injected. To prevent confusion, consider extending `_AUTO_FILLED_BY_FL_V7` to include it.

### Tests to add

- `tests/test_v7_feddyn_server_step.py`:
  - `test_feddyn_server_applies_h_accum_after_first_round()` — w_new should differ from FedAvg-baseline once h_accum non-zero
  - `test_feddyn_alpha_zero_degenerates_to_fedavg()` — alpha=0 should match FedAvg exactly
  - `test_feddyn_n_total_clients_auto_injected_by_fl_v7()` — fl_v7 should auto-set this

### Update SCAFFOLD partial-participation note

While we're at it, update `scaffold.py:244-251` comment to clarify:
```python
# Currently we approximate N=|S| (mean over participants), which over-counts
# c by N/|S| per round. For our 7-client × clients_per_round=5 setup the
# bias factor is 1.4×. Documented as known limitation in §method.
```

This is documentation-only, not a code fix.

---

## Part 3: Smoke + cross-GPU (Phase 3c, task #137)

After NVML + FedDyn fixes pass tests:

```bash
# 1. NVML smoke (1 cell, expect <90s + energy_measured.json populated)
python experiments/run_v7_phase_sweep.py --spec experiments/specs/phase2_min.yaml \
    --output-dir artifacts/v7_nvml_smoke --summary-tag nvml_smoke --limit 1

# 2. Cross-GPU smoke on 4080 (baseline)
python experiments/run_v7_phase_sweep.py --spec experiments/specs/phase3c_cross_gpu_smoke.yaml \
    --output-dir artifacts/v7_phase3c_4080 --summary-tag phase3c_4080

# 3. Cross-GPU on cloud T4 (Lambda Labs / vast.ai / Colab Pro)
# Same spec, expect AUC within ±0.3% of 4080, latency 2-3× slower

# 4. Cross-GPU on cloud A100 (similar)
# Same spec, expect AUC within ±0.3% of 4080, latency 2× faster
```

---

## Part 4: Phase 5 launch authorization

After NVML + FedDyn smoke green, request user authorization to launch Phase 5:
```bash
nohup python experiments/run_v7_phase_sweep.py \
    --spec experiments/specs/stage2_full.yaml \
    --output-dir artifacts/v7_stage2_full \
    --summary-tag stage2_full \
    --continue-on-cell-failure \
    > logs/stage2_full_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

Expected: 750 cells × ~70-90s × NVML overhead (~5%) ≈ 17-20 hours.

---

## Order of execution tomorrow

1. Read `measure_v6_gpu_energy.py` fully (~30 min)
2. TDD red: write `test_v7_nvml_integration.py` and `test_v7_feddyn_server_step.py` (~1 hr)
3. Implement NVML helpers in fl_v7 (~1.5 hr)
4. Implement FedDyn n_total_clients fix (~30 min)
5. Run all tests until green (~30 min)
6. NVML smoke (1 cell ~90s + sanity check on energy numbers)
7. Phase 3c cross-GPU smoke local (~3 min)
8. Update `PAPER_DRAFT.md` with NVML methodology paragraph in §5
9. Commit
10. Request Phase 5 launch authorization

Total estimate: 4-6 hours.
