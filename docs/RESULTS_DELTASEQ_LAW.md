# RESULTS — The Δ_seq law: the fragmentation AUC gap is task-conditional

Empirical record for the standalone methods paper *"Sequence Integrity Matters: Row-Level
Client Partitioning Artifacts in Federated RAN Time-Series Benchmarks."* All runs 2026-05-24,
RTX 4060 Ti (local, memory-caged) + the earlier V100 ColO-RAN sweep. Pre-registration:
`docs/PREREG-A2-deltaseq-law.md`. Raw per-cell JSONs under `artifacts/prea1/twinning/`
(gitignored, re-derivable via the scripts below); this file + the committed figure are the
recoverable source of truth.

## Claim

Row-level client partitioning fragments per-client sliding windows on **every** dataset
(mechanism, universal). But the **AUC impact is task-conditional**: it degrades AUC only on
**sequence-essential** tasks, and the fragmentation gap is a monotone function of a cheap,
partition-free diagnostic:

- **Δ_seq** = AUC(seq LSTM) − AUC(single-step LSTM)  — capacity-matched sequence value.
- **Δ_traj** = AUC(seq LSTM) − AUC(shuffle-within-window LSTM)  — isolates the *order/trajectory*
  value (partition-vulnerable) from the partition-invariant *run-rate*. Cleaner predictor.

## 1. Mechanism (universal) — fragmentation audit

| dataset | intact (run/entity) | random_split (row) | dirichlet (row) | groups, rows |
|---|---|---|---|---|
| ColO-RAN | 1.000 | ~0 | ~0 | (run_id,slice_id) |
| Twinning | 1.0000 | 0.0002 | 0.0031 | 6858 (run,UE), 30.97M |

Fragmentation-score = fraction of per-client windows with contiguous `step_idx`.
Scripts: `scripts/prea1/twinning_audit_stream.py` (+ ColO-RAN partition audit).

## 2. ColO-RAN Δ_seq law (money result) — `coloran_deltaseq_sweep.py`, 11 targets, 5 seeds, frac 0.3, 50 rounds

| target | Δ_seq | Δ_traj | gap (intact−row) | gap CI95 (5-seed paired boot) |
|---|---|---|---|---|
| buffer_med | +0.013 | +0.000 | +0.000 | [−0.0002, +0.0003] (incl 0) |
| mcs_med | +0.039 | +0.002 | +0.000 | [+0.0000, +0.0004] |
| cqi_med | +0.048 | +0.001 | −0.011 | [−0.0125, −0.0096] |
| brate_med | +0.111 | +0.010 | −0.008 | [−0.0103, −0.0043] |
| bler_trend5 | +0.159 | +0.060 | +0.083 | [+0.0815, +0.0843] |
| bler_k2 | +0.170 | +0.200 | +0.134 | [+0.1299, +0.1381] |
| bler_k3 | +0.209 | +0.219 | +0.121 | [+0.1167, +0.1251] |
| bler_th20 | +0.221 | +0.222 | +0.149 | [+0.1453, +0.1536] |
| bler_th10 | +0.227 | +0.225 | +0.152 | [+0.1487, +0.1563] |
| bler_th05 | +0.227 | +0.225 | +0.152 | [+0.1491, +0.1564] |
| bler_k5 | +0.233 | +0.232 | +0.158 | [+0.1537, +0.1636] |

- **gap vs Δ_seq**: Pearson **+0.943**, Spearman **+0.918**, OLS slope 0.825, slope CI95
  [0.706, 1.113] (excludes 0). **H1 CONFIRMED.**
- **gap vs Δ_traj**: Pearson **+0.974**, Spearman **+0.945** — the order-free shuffle baseline is a
  *cleaner* predictor and **fixes the `brate_med` deviation** (Δ_seq 0.111 but Δ_traj 0.010 → on
  the line: brate's multi-step value is run-rate denoising, not trajectory). `coloran_deltatraj_addendum.py`.
- Figure: `artifacts/prea1/twinning/deltaseq_law.pdf`.
- bler_th10 reproduces the established ColO-RAN gap (intact 0.91 / row 0.71 / gap +0.16).

## 3. Mechanism: run-rate (invariant) vs trajectory (vulnerable) — source-column lag-1 autocorr

| source | autocorr | nature | Δ_traj | gap |
|---|---|---|---|---|
| tx_brate_dl | +0.984 | persistent → run-rate | ~0 | ~0 |
| dl_buffer | +0.999 | persistent | ~0 | ~0 |
| dl_mcs | +0.902 | persistent | ~0 | ~0 |
| dl_cqi | +0.553 | moderate | ~0 | ~0 |
| **ul_bler** | **+0.022** | **white-noise → predicted via channel-state trajectory** | **0.22** | **0.15** |

Persistent targets are predictable order-free → fragmentation-robust. The white-noise BLER target
is predictable only via the multivariate channel-state **trajectory** → fragmentation-vulnerable.

## 4. Synthetic causal control — `synth_seqessential_control.py` (local)

Tunable sequence-essentiality knob λ (instantaneous vs 4-step difference), pos-rate held ~0.5,
all else fixed. gap monotone in Δ_seq, **Pearson +0.986**. **Falsifies** the conjectured
`gap ≤ Δ_seq` bound (super-linear at high λ where fragmented training is actively mis-led below
single-step). ⇒ Δ_seq/Δ_traj are monotone **diagnostics**, not bounds.

## 5. Architecture invariance + sanity — `coloran_arch_invariance.py` (local, 4 targets, 3 seeds, frac 0.15)

- **MLP sanity + gap decomposition:** a no-sequence mean-pool MLP shows gap `0.000` for the *point*
  target bler_th10 (per-seed −0.0002/−0.0002/+0.0004) but a real, seed-consistent `0.025` for the
  *aggregate* smoothed target bler_trend5 (per-seed 0.0252/0.0248/0.0242). This isolates a **second,
  smaller gap source — window-content** (fragmentation changes *which* steps populate a window, so a
  window-aggregate label shifts even for an order-invariant model), distinct from the **trajectory**
  component (sequence models only). For point targets the gap is ~pure trajectory (MLP ≈ 0, LSTM 0.16);
  for aggregate targets the trajectory component still dominates (MLP 0.025 ≪ LSTM 0.082). See
  `docs/SEQUENCE_INTEGRITY_THEORY.md`. Net: **the gap is sequence-specific for point targets and
  trajectory-dominated otherwise** (not "purely" sequence-specific — stated precisely).
- Partition client counts differ by mode (iid = 7 BS; run/row Dirichlet = 8) — immaterial to the
  global-test gap but noted for full disclosure.
- **LSTM + GRU** both learn the BLER trajectory (intact ~0.90) and show gap tracking Δ_seq
  (bler +0.16/+0.155, trend5 +0.08, brate/mcs ~0) → law is architecture-invariant.
- **Transformer: underfit** in this small-data FL regime (intact bler 0.69 ≈ MLP 0.66, never
  learned the trajectory) → honestly excluded as uninformative, not forced.

## 6. Twinning AUC-impact — negative control — `twinning_auc_impact.py` (V100, 1-seed smoke)

dl_cqi / dl_mcs next-step targets: gap −0.007 / −0.001 (no gap). A REAL null (single-step
diagnostic), consistent with low Δ_seq on Twinning's persistent targets. The mechanism replicates
(§1) but the AUC impact does not — exactly as the Δ_seq law predicts.

## Honest caveats

- The `gap ≤ Δ_seq` bound is **false** (synthetic). Diagnostics are monotone, not bounds.
- `brate_med` deviates under Δ_seq but is corrected by Δ_traj (run-rate vs trajectory).
- `bler_trend5` MLP shows a 0.025 gap (smoothing injects some run-rate); still ≪ LSTM's 0.082.
- Transformer underfit (small-data FL) — excluded, not evidence either way.
- Twinning AUC-impact is a 1-seed smoke (mechanism + diagnostic suffice; 5-seed CI is easy future hardening).

## Reproduce

```bash
# synthetic control (local, caged):
systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 .venv/bin/python \
  scripts/prea1/synth_seqessential_control.py
# ColO-RAN Delta_seq sweep (per-GPU --targets subsets; 4-GPU on V100 or local caged frac 0.3):
.venv/bin/python scripts/prea1/coloran_deltaseq_sweep.py --keep-run-frac 0.3 --seeds 0,1,2,3,4
# Delta_traj addendum + arch-invariance (local, caged):
.venv/bin/python scripts/prea1/coloran_deltatraj_addendum.py --keep-run-frac 0.3
.venv/bin/python scripts/prea1/coloran_arch_invariance.py --keep-run-frac 0.15
# aggregate + figure:
.venv/bin/python scripts/prea1/aggregate_deltaseq_law.py
```

Windowing optimization for repeated local runs: `scripts/prea1/window_cache.py`
(disk-cache + group-id intact indexing; bit-exact vs `build_run_sequences` —
`tests/test_window_cache.py`).
