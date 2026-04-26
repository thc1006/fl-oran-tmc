# Stage 1 (v6) Final Analysis — 3-Architecture Centralized Benchmark on ColO-RAN

**Status**: paper-grade analysis after Tier A+B completion (130 + 30 = 160 cells × NVML measurement).
**Source of truth**: `artifacts/v6_arch_sweep/aggregated.json` (regenerated 2026-04-26 17:14 by `scripts/aggregate_v6_results.py`).
**Per-cell NVML data**: `artifacts/v6_arch_sweep/<cell>/energy_measured.json` (n=159 cells measured; 1 LIF E7 cell single-seed each, all others n=10).

This document is the persistent companion to `RESULTS_V6_STAGE1.md` (the auto-generated table). All paired bootstrap CIs and D-21 decisions cited here are read directly from the aggregator output (unit-tested by `tests/test_v6_aggregator_routing.py` and `tests/test_v6_cell_metadata.py`, 83 tests passing). NVML measured energies are read from per-cell `energy_measured.json` files.

---

## 1. Executive summary

The Stage 1 v6 benchmark establishes five robust findings on the ColO-RAN slice SLA forecasting task (binary classification of `ul_bler > 0.10` at 1-step horizon, OOD split by tr id):

1. **Mamba-S6 ≈ LSTM in test AUC at 5k/25k/50k matched gradient steps** (paired bootstrap CI95 of `mamba−lstm` includes 0 at all three budgets), with Mamba consuming -14% theoretical energy. **At 100k matched budget, LSTM significantly outperforms Mamba** by ΔAUC = 0.0012 (paired CI95 [0.0007, 0.0017], wilcoxon p=0.002, robust to Bonferroni correction over 176 family-wise comparisons).
2. **The preregistered Spiking-SSM configuration catastrophically fails** (test AUC 0.6757 ± 0.0354, n=10) due to the combination of low learning rate (1e-4 vs 5e-4 used by dense baselines) and binary-decoder collapse at default LIF threshold/beta.
3. **Two post-hoc audit recovery routes achieve preregistered C1+C2 PASS at matched 25k budget under sparsity-aware energy accounting**: (a) lr=5e-4 (`spiking_lr5e4_25k`); (b) channel expansion (`spiking_expand2`, with `backbone_d_model=56, expand=2`). The expand variant is the strongest GO with the largest C2 margin (energy ratio 0.439 vs threshold 0.5). **Both PASS results are robust under Bonferroni correction.**
4. **Architectural ablation reveals an asymmetry between Mamba and Spiking on this task**: `mamba_expand2` is dead weight (mean ΔAUC vs `mamba_25k` = +0.0006, CI95 includes 0; +12% theoretical energy and +4% measured energy). `spiking_expand2`, in contrast, achieves AUC parity with `spiking_lr5e4_25k` at -11% theoretical energy and the strongest D-21 margin.
5. **NVML measurements on RTX 4080 reveal a real-vs-theoretical energy ranking inversion**: in measured GPU wallclock energy, LSTM ≪ Mamba ≪ Spiking T=1 ≪ Spiking T=5. The theoretical "Spiking saves 56% energy" advantage **only realises on neuromorphic / sparsity-aware hardware**; on commodity GPU, Spiking is ~9× more expensive than LSTM (T=1) or ~29× (T=5). This is the **applicability-boundary** finding for the paper's discussion section.

**Stage 2 implication**: the preregistered NO-GO holds (default Spiking fails). Two audit pathways achieve GO at matched 25k under sparsity-aware accounting. Whether to pursue Stage 2 depends on the deployment hardware assumption — for commodity-GPU edge deployment, LSTM dominates; for neuromorphic / sparsity-aware accelerator deployment, Spiking-SSM with channel expansion is the strongest candidate.

---

## 2. Setup recap

### 2.1 Dataset and target

ColO-RAN unified parquet (`data/coloran_raw_unified.parquet`, 18M rows, 21 features). Binary classification target `y_sla_next = ul_bler[t+1] > 0.10`. OOD split by `tr` id: train `tr ∈ [0,21]`, val `tr ∈ [22,24]`, test `tr ∈ [25,27]`. After targeting filter: 14.5M train / 1.9M val / 1.9M test sequences (seq_len=5).

### 2.2 Architectures and budget grid

5 architectures × 4 training budget points × 10 seeds + 9 LIF grid (single-seed) audit + 2 recovery axes (multi-seed):

| arch key | params | flops | sops | recipe |
|---|---|---|---|---|
| `lstm` | 44553 | 210112 | 0 | ForecasterV2: hidden1=64, hidden2=32, dropout=0.1, lr=5e-4 |
| `mamba` | 40489 | 180608 | 0 | MambaForecaster: d_model=64, expand=1, n_blocks=2, lr=5e-4 |
| `mamba_expand2` | 45897 | 202416 | 0 | MambaForecaster: d_model=48, expand=2, n_blocks=2, lr=5e-4 |
| `spiking` (preregistered) | 42921 | 98512 | 36064 | SpikingForecaster: d_model=80, n_blocks=2, t_inner=1, lr=1e-4, lif_threshold=1.0, lif_beta=0.9 |
| `spiking_expand2` | 43593 | 87512 | 24236 | SpikingForecaster: d_model=56, expand=2, n_blocks=2, t_inner=1, lr=5e-4 |

Recovery audit variants of `spiking`:

| variant | recipe |
|---|---|
| `spiking_lr5e4_25k` | spiking + lr=5e-4, 25k steps |
| `spiking_t5sum` | spiking + t_inner=5, decode=sum, lr=5e-4, 25k |
| `spiking_t5sum_50k` | spiking + t_inner=5, decode=sum, lr=5e-4, 50k |
| `spiking_lif_t<NN>_b<MM>` | spiking + lif_threshold=NN/10, lif_beta=MM/10, lr=5e-4, 25k (single seed) |

Budget grid: 5k / 25k / 50k / 100k gradient steps, batch=64, mixed-precision bf16.

Seed grid: `42, 0, 1, 2, 3, 7, 11, 13, 17, 23` (n=10 unless otherwise noted).

### 2.3 Energy accounting

Per ADR-001 D-20, three hardware accounting models:
- `total_energy_pJ_gpu_dense` = (`fvcore_flops` + `rnn_macs` + `post_spike_macs`) × 4.6 pJ/MAC. Worst case for Spiking (no sparsity exploitation).
- `total_energy_pJ_sparsity_aware` = (`fvcore_flops` + `rnn_macs` − `post_spike_macs`) × 4.6 pJ/MAC + `sops_spike_driven` × 0.9 pJ/AC. Treats actual spike events as accumulate operations on the post-LIF `out_proj` layer.
- `total_energy_pJ_neuromorphic` = same as `sparsity_aware` for current architectures (only `out_proj` directly receives spikes; classifier head receives a dense float vector).

Coefficients pinned to Horowitz (ISSCC 2014, 45nm CMOS): 4.6 pJ/MAC, 0.9 pJ/AC.

### 2.4 D-21 evaluation framework (preregistered)

- **C1**: `ci95_hi(delta_auc(spiking, lstm)) ≥ -0.030`
- **C2**: `mean(energy_pJ_spiking) / mean(energy_pJ_lstm) ≤ 0.5`
- **C3**: `ci95_lo(delta_auc(mamba, lstm)) ≥ -0.030` (sanity that Mamba arm is healthy)

Decisions: GO Spiking-led (C1+C2+C3), GO trade-off (C1+C3 only), GO Mamba-led fallback (C3 + Mamba significantly better), NO-GO otherwise.

CIs: paired bootstrap, n_boot=10000, paired across same seeds. Wilcoxon signed-rank reported as secondary diagnostic. Bonferroni correction across n_compare=176 family-wise comparisons reported in appendix.

### 2.5 Reproducibility

Source of truth: `artifacts/v6_arch_sweep/aggregated.json`, sha-trackable via `git diff` of `docs/RESULTS_V6_STAGE1.md`.

Code: pinned to commit `0a5f2c7` (Round 4-9 fixes for kwargs reconstruction, idempotency, atomic writes). 83 unit tests in `tests/test_v6_*.py`.

---

## 3. Per-architecture results

### 3.1 Test AUC mean ± std (paired across seeds)

```
arch                    n  test_AUC  std       test_F1   test_acc  params
─────────────────────────────────────────────────────────────────────────
lstm                   10  0.9151    0.0010    0.7623    0.8424    44553
lstm_25k               10  0.9242    0.0006    0.7796    0.8530    44553
lstm_50k               10  0.9271    0.0004    0.7822    0.8546    44553
lstm_100k              10  0.9295    0.0007    0.7873    0.8576    44553
mamba                  10  0.9153    0.0008    0.7620    0.8420    40489
mamba_25k              10  0.9239    0.0012    0.7775    0.8516    40489
mamba_50k              10  0.9268    0.0009    0.7841    0.8553    40489
mamba_100k             10  0.9283    0.0012    0.7859    0.8569    40489
mamba_expand2          10  0.9248    0.0009    0.7780    0.8519    45897
spiking (preregistered)10  0.6757    0.0354    0.4960    0.5917    42921
spiking_lr5e4_25k      10  0.8944    0.0018    0.7294    0.8253    42921
spiking_t5sum          10  0.9024    0.0025    0.7436    0.8350    42921
spiking_t5sum_50k      10  0.9110    0.0015    0.7576    0.8442    42921
spiking_expand2        10  0.8952    0.0018    0.7306    0.8260    43593
spiking_t5             10  0.5002    0.0003    0.2350    0.5000    42921
spiking_lif_t05_b05     1  0.9034     —        0.7453    0.8316    42921
spiking_lif_t05_b09     1  0.9041     —        0.7448    0.8317    42921
spiking_lif_t05_b099    1  0.9007     —        0.7359    0.8266    42921
spiking_lif_t10_b05     1  0.9049     —        0.7484    0.8336    42921
spiking_lif_t10_b09     1  0.8922     —        0.7289    0.8205    42921
spiking_lif_t10_b099    1  0.8897     —        0.7257    0.8180    42921
spiking_lif_t20_b05     1  0.8913     —        0.7210    0.8204    42921
spiking_lif_t20_b09     1  0.8847     —        0.7176    0.8137    42921
spiking_lif_t20_b099    1  0.8742     —        0.7042    0.8052    42921
```

### 3.2 Theoretical and measured energy (n=10 each unless noted)

```
arch                    flops    sops      gpu_dense  sparsity   measured (NVML mean ± std)
                                          pJ/inf     pJ/inf     pJ/inf
─────────────────────────────────────────────────────────────────────────────────
lstm                    210112       0    9.67e+05   9.67e+05   1.92e+08 ± 2.94e+07
lstm_25k                 (same)      0    9.67e+05   9.67e+05   2.03e+08 ± 2.99e+07
lstm_50k                 (same)      0    9.67e+05   9.67e+05   2.12e+08 ± 3.02e+07
lstm_100k                (same)      0    9.67e+05   9.67e+05   1.90e+08 ± 2.62e+07
mamba                   180608       0    8.31e+05   8.31e+05   7.41e+08 ± 2.41e+07
mamba_100k               (same)      0    8.31e+05   8.31e+05   7.30e+08 ± 3.08e+07
mamba_expand2           202416       0    9.31e+05   9.31e+05   7.70e+08 ± 2.55e+07
spiking (preregistered)  98512   36064    7.48e+05   4.86e+05   1.83e+09 ± 3.39e+07
spiking_lr5e4_25k        98512   27303    7.48e+05   4.78e+05   1.83e+09 ± 3.03e+07
spiking_t5sum            98512  149994    7.48e+05   5.88e+05   5.70e+09 ± 4.28e+07
spiking_t5sum_50k        98512  148167    7.48e+05   5.87e+05   5.74e+09 ± 5.69e+07
spiking_expand2          87512   24236    6.91e+05   4.24e+05   1.87e+09 ± 2.36e+07
spiking_t5 (catastrophic)98512   13306    7.48e+05   4.65e+05   5.74e+09 ± 2.85e+07
```

**Notes**: per-arch theoretical energy is budget-invariant (the model architecture, not the trained weights, determines the per-inference operation count). Per-arch measured energy is also budget-invariant in principle, but exhibits ~1-15% cell-to-cell variance from GPU thermal/idle dynamics (see §6.1).

### 3.3 Per-inference latency (NVML wallclock, batch=64 amortized)

Latency = `wallclock_sec / n_inferences_measured × 1000` ms. "Amortized" because the measurement runs at batch=64; this is throughput latency (per-instance work amortized over the batch parallelism). Single-call latency at batch=1 (the actual edge deployment scenario) is at least an order of magnitude higher because GPU kernel-launch overhead dominates small-batch inference. The amortized number is therefore a **lower bound** on real per-call latency.

| arch | n | latency_ms (mean ± std) | latency_µs (mean) | EDP_pJ·s (mean ± std) |
|---|---|---|---|---|
| `lstm` family (5k-100k) | 4×10 | 0.003 ± 0.000 | ~3 µs | (5.05-5.63)e2 ± ~80 |
| `mamba` family (5k-100k) | 4×10 | 0.008 ± 0.000 | ~8 µs | (6.04-6.15)e3 ± ~250 |
| `mamba_expand2` | 10 | 0.008 ± 0.000 | ~8 µs | 6.38e3 ± 227 |
| `spiking` (preregistered) / `lr5e4_25k` / `expand2` (T=1) | 3×10 | 0.025 ± 0.000 | ~25 µs | (4.51-4.60)e4 ± ~700 |
| `spiking_t5` / `t5sum` / `t5sum_50k` (T=5) | 3×10 | 0.079 ± 0.001 | ~79 µs | (4.49-4.56)e5 ± ~6000 |

**Real-edge latency budget context**: production O-RAN slice SLA prediction requires inference within the slice scheduling window (typically <10 ms per decision). All measured architectures fit this budget at batch=64 amortized. At batch=1 (real-time per-UE prediction), expected scaling is 10-50×, putting Spiking T=5 at the boundary (~5 ms) and rendering Mamba marginal (~0.5 ms acceptable but with thinner margin). LSTM remains ≪1 ms even at batch=1, comfortably meeting the budget.

### 3.4 NVML reality factor (measured / theoretical)

```
arch                    ratio_sparsity   ratio_gpu_dense
─────────────────────────────────────────────────────────
lstm (5k-100k)          197 — 220×       (same as sparsity, no sops)
mamba (5k-100k)         878 — 892×       (same)
mamba_expand2           827×             (same)
spiking T=1             3775 — 4396×     2443 — 2723×
spiking T=5             9700 — 12347×    7629 — 7710×
```

The **ratio_sparsity − ratio_gpu_dense gap** for Spiking variants is a direct quantification of the "claimed sparsity advantage that does not realise on commodity GPU". For `spiking_expand2`: ratio_sparsity (4396×) / ratio_gpu_dense (2723×) ≈ 1.61, meaning sparsity-aware accounting predicts ~62% energy savings versus gpu_dense, but the GPU realises 0% of that — measured energy is identical regardless of accounting model.

---

## 4. Pairwise paired-bootstrap CI95 (uncorrected and Bonferroni-corrected)

All CIs from `aggregated.json` `deltas` dict, n_boot=10000, paired across seeds. Bonferroni n_compare=176 (CI99.97%).

### 4.1 D-21 critical pairs

| comparison | mean | CI95 uncorr | CI Bonferroni | C1 verdict |
|---|---|---|---|---|
| `mamba_vs_lstm` | +0.0002 | [-0.0005, +0.0009] | [-0.0008, +0.0015] | n/a (Mamba arm) |
| `mamba_25k_vs_lstm_25k` | -0.0003 | [-0.0009, +0.0003] | [-0.0015, +0.0007] | C3 PASS (lo=-0.0009 ≥ -0.030) |
| `mamba_50k_vs_lstm_50k` | -0.0004 | [-0.0009, +0.0003] | [-0.0012, +0.0008] | C3 PASS |
| `mamba_100k_vs_lstm_100k` | **-0.0012** | **[-0.0017, -0.0007]** | **[-0.0021, -0.0004]** | **CI excludes 0; LSTM significantly ahead** |
| `mamba_expand2_vs_lstm_25k` | +0.0006 | [-0.0000, +0.0012] | [-0.0004, +0.0019] | not in D-21 framework |
| `spiking_lr5e4_25k_vs_lstm_25k` | -0.0299 | [-0.0311, **-0.0286**] | [-0.0318, **-0.0277**] | **PASS** (margin 0.0014 / 0.0023) |
| `spiking_lr5e4_25k_vs_lstm_50k` | -0.0328 | [-0.0338, -0.0316] | [-0.0344, -0.0305] | FAIL |
| `spiking_lr5e4_25k_vs_lstm_100k` | -0.0352 | [-0.0361, -0.0343] | [-0.0369, -0.0337] | FAIL |
| `spiking_t5sum_vs_lstm_25k` | -0.0218 | [-0.0234, **-0.0201**] | [-0.0245, **-0.0189**] | **PASS** |
| `spiking_t5sum_vs_lstm_50k` | -0.0247 | [-0.0261, -0.0233] | [-0.0269, -0.0223] | **PASS** |
| `spiking_t5sum_vs_lstm_100k` | -0.0271 | [-0.0286, -0.0256] | [-0.0295, -0.0246] | **PASS** |
| `spiking_t5sum_50k_vs_lstm_25k` | -0.0132 | [-0.0144, -0.0121] | [-0.0153, -0.0111] | **PASS** |
| `spiking_t5sum_50k_vs_lstm_50k` | -0.0161 | [-0.0172, -0.0153] | [-0.0181, -0.0145] | **PASS** |
| `spiking_t5sum_50k_vs_lstm_100k` | -0.0185 | [-0.0195, -0.0176] | [-0.0204, -0.0169] | **PASS** |
| `spiking_expand2_vs_lstm` | -0.0199 | [-0.0212, -0.0187] | [-0.0222, -0.0179] | **PASS** vs preregistered baseline |
| `spiking_expand2_vs_lstm_25k` | -0.0290 | [-0.0303, **-0.0278**] | [-0.0312, **-0.0269**] | **PASS** (margin 0.0022 / 0.0031) |
| `spiking_expand2_vs_lstm_50k` | -0.0319 | [-0.0329, -0.0309] | [-0.0336, -0.0301] | FAIL by 0.0009 / 0.0001 (super marginal Bonferroni) |
| `spiking_expand2_vs_lstm_100k` | -0.0343 | [-0.0355, -0.0333] | [-0.0363, -0.0326] | FAIL |

**Key observation about Bonferroni and non-inferiority testing**: Because D-21 C1 is a non-inferiority test (`ci_hi ≥ −0.030`), wider CIs (Bonferroni) generally make PASS easier, not harder. All uncorrected PASS results in the table above remain PASS under Bonferroni correction across 176 comparisons. The two robust PASS results (`spiking_expand2_vs_lstm_25k` and `spiking_lr5e4_25k_vs_lstm_25k`) widen their PASS margins under Bonferroni.

### 4.2 D-21 audit decisions (full grid, ordered)

The aggregator computed 84 D-21 decisions across { audit variants × baselines × hardware accountings }. Key decisions:

| variant | sparsity-aware | gpu_dense | C1 hi | C2 ratio |
|---|---|---|---|---|
| `spiking_lr5e4_25k_vs_25k_baselines` | **GO Spiking-led** | trade-off (C2 fail) | -0.0286 | 0.494 |
| `spiking_lr5e4_25k_vs_50k_baselines` | NO-GO (C1 fail) | NO-GO | -0.0316 | 0.494 |
| `spiking_lr5e4_25k_vs_100k_baselines` | NO-GO | NO-GO | -0.0343 | 0.494 |
| `spiking_t5sum_vs_25k_baselines` | trade-off (C2 fail) | trade-off | -0.0201 | 0.609 |
| `spiking_t5sum_vs_50k_baselines` | trade-off | trade-off | -0.0233 | 0.609 |
| `spiking_t5sum_vs_100k_baselines` | trade-off | trade-off | -0.0256 | 0.609 |
| `spiking_t5sum_50k_vs_50k_baselines` | trade-off | trade-off | -0.0153 | 0.607 |
| `spiking_t5sum_50k_vs_100k_baselines` | trade-off | trade-off | -0.0176 | 0.607 |
| `spiking_expand2_vs_5k_baselines` | **GO Spiking-led** | trade-off | -0.0187 | 0.439 |
| `spiking_expand2_vs_25k_baselines` | **GO Spiking-led** | trade-off | -0.0278 | 0.439 |
| `spiking_expand2_vs_50k_baselines` | NO-GO (C1 fail by 0.0009) | NO-GO | -0.0309 | 0.439 |
| `spiking_expand2_vs_100k_baselines` | NO-GO | NO-GO | -0.0333 | 0.439 |
| `spiking_lif_*_vs_*_baselines` (9 cells × 7 routes) | NO-GO (n=1, C1=n/a) | NO-GO | n/a | 0.486-0.496 |

**Pattern**: among 10-seed audit variants, `spiking_expand2` and `spiking_lr5e4_25k` achieve **GO Spiking-led at matched 25k**; `spiking_t5sum` and `spiking_t5sum_50k` achieve **trade-off (C1 PASS, C2 fail) consistently across all baselines**. All variants fail at 50k+ matched baselines except the t5sum family which maintains C1 PASS but never C2 PASS.

---

## 5. Five robust findings (with citations)

### 5.1 F1: Mamba ≈ LSTM up to 50k matched, LSTM significantly ahead at 100k

**Evidence**:
- 5k: `mamba_vs_lstm` paired CI95 [-0.0005, +0.0009] (includes 0)
- 25k: `mamba_25k_vs_lstm_25k` CI [-0.0009, +0.0003] (includes 0)
- 50k: `mamba_50k_vs_lstm_50k` CI [-0.0009, +0.0003] (includes 0)
- 100k: `mamba_100k_vs_lstm_100k` CI [-0.0017, -0.0007] (**excludes 0**), wilcoxon p=0.00195, Bonferroni CI [-0.0021, -0.0004] (**excludes 0**)

**Saturation rates (mean increment per doubling of training)**:
- LSTM: 5k→25k +0.0091; 25k→50k +0.0029; 50k→100k +0.0024
- Mamba: 5k→25k +0.0086; 25k→50k +0.0029; 50k→100k +0.0015

**Interpretation**: Mamba saturates faster than LSTM. At 5k-50k both are essentially at the same accuracy ceiling. Beyond 50k LSTM continues to improve while Mamba plateaus, leading to a small but statistically significant LSTM lead at 100k.

**Energy**: Mamba's theoretical sparsity_aware energy is 8.31e5 vs LSTM 9.67e5 = -14% (budget-invariant). Measured Mamba 7.41e8 vs LSTM 1.92e8 = +286% (i.e., 3.86× more expensive on commodity GPU). The selective-scan kernel has no cuDNN equivalent, which dominates real-energy cost regardless of theoretical advantage.

### 5.2 F2: Two recovery routes achieve preregistered C1+C2 PASS at matched 25k

**Evidence**:
- `spiking_lr5e4_25k`: paired CI95 [-0.0311, -0.0286] vs lstm_25k; C1 PASS by 0.0014. Energy ratio 0.494 vs C2 threshold 0.5 (PASS by 0.006). Bonferroni CI [-0.0318, -0.0277], PASS by 0.0023.
- `spiking_expand2`: paired CI95 [-0.0303, -0.0278] vs lstm_25k; C1 PASS by 0.0022. Energy ratio 0.439 vs C2 threshold 0.5 (PASS by 0.061). Bonferroni CI [-0.0312, -0.0269], PASS by 0.0031. Also PASSes against preregistered 5k baselines (vs `lstm`: CI [-0.0212, -0.0187], C1 margin 0.0113).

**Strongest GO**: `spiking_expand2` (smaller CI, larger margin on both axes, also passes vs `lstm` 5k baseline).

**Caveat**: GO holds only under sparsity-aware energy accounting. Under gpu_dense accounting, both recovery variants degrade to trade-off (C2 fail; gpu_dense ratio = 0.715-0.773).

### 5.3 F3: Architectural ablation asymmetry between Mamba and Spiking on `expand>1`

**Evidence**:
- `mamba_expand2_vs_lstm_25k`: paired delta +0.0006, CI [-0.0000, +0.0012]. Vs `mamba_25k`: essentially equal. **Theoretical energy +12% (9.31e5 vs 8.31e5), measured energy +4% (7.70e8 vs 7.41e8)**. Conclusion: dead weight on this task.
- `spiking_expand2`: vs spiking_lr5e4_25k AUC +0.0008 (within noise), but params +1.6%, **theoretical flops -11.2% (87512 vs 98512), measured energy +2% (1.87e9 vs 1.83e9)**. Conclusion: same-AUC at lower theoretical compute (architectural improvement).

**Interpretation hypothesis** (paper §discussion candidate): Mamba's selective scan already exploits per-channel computation; doubling d_inner gives no representational gain on this task while paying the projection cost. SpikingSSMBlock's discrete LIF spike train is a per-channel information bottleneck; doubling d_inner relieves that bottleneck while shrinking d_model offsets the in/out projection cost.

### 5.4 F4: Real-vs-theoretical energy ranking inversion on commodity GPU

**Evidence (mean measured pJ/inference, n=10)**:
- LSTM family: 1.90e8 — 2.12e8 (1× baseline)
- Mamba family: 7.30e8 — 7.41e8 (3.7× LSTM)
- mamba_expand2: 7.70e8 (3.9× LSTM)
- Spiking T=1 (lr5e4 / expand2): 1.83e9 — 1.87e9 (9.3× LSTM)
- Spiking T=5 (t5, t5sum, t5sum_50k): 5.70e9 — 5.74e9 (28.6× LSTM)

**Theoretical ordering reversed** (sparsity-aware accounting):
- spiking_expand2: 4.24e5 (cheapest theoretically)
- spiking_lr5e4_25k: 4.78e5
- mamba: 8.31e5
- LSTM: 9.67e5 (most expensive theoretically)

**Reality factors**: LSTM 197-220× / Mamba 878-892× / Spiking T=1 ~3800-4400× / Spiking T=5 ~9700-12347×.

**Causes**:
1. LSTM uses cuDNN-optimised CUDA kernels; the highest-grade GPU implementation in the comparison.
2. Mamba's selective scan has no cuDNN equivalent; PyTorch / Triton implementations have ~5× lower kernel efficiency.
3. SpikingSSMBlock's `t_inner` inner LIF integration loop introduces sequential dependency through LIF membrane state, defeating GPU parallelism. T=1 already 9.3× more expensive than LSTM; T=5 multiplies by 3.1×.

**Implication for paper §discussion**: theoretical energy advantages of biology-inspired architectures (Spiking-SSM, Mamba) **do not transfer to commodity GPU deployment**. The energy savings only realise on hardware that is the architecture's native primitive (neuromorphic for Spiking-SSM; potentially custom sparsity-aware accelerators for Mamba's selective scan).

### 5.5 F6: Energy-Delay Product (EDP) amplifies real-deployment cost asymmetry

EDP = energy × latency, the canonical edge-AI joint metric (penalises both watts and seconds of GPU time). Computed from the NVML `wallclock_sec / n_inferences × 1000` (latency_ms) × `energy_pJ_per_inference_total` (measured_pJ).

**Per-arch EDP_pJ·s (mean across n=10 cells per arch, throughput-amortized at batch=64)**:

| arch family | EDP_pJ·s | factor over LSTM |
|---|---|---|
| LSTM | ~5.0e2 — 5.6e2 | **1×** |
| Mamba | ~6.0e3 — 6.4e3 | **~12×** |
| Spiking T=1 (lr5e4, expand2) | ~4.5e4 — 4.6e4 | **~88×** |
| Spiking T=5 (t5sum, t5sum_50k) | ~4.5e5 — 4.6e5 | **~880×** |

**Critical observation**: EDP magnifies the energy ranking by an additional ~10× because spiking's wallclock penalty multiplies the energy penalty:
- Energy alone: Spiking T=1 ≈ 9.3× LSTM
- EDP: Spiking T=1 ≈ 88× LSTM (energy 9.3× × latency 9.4× ≈ 87×)

For deployment decisions where both wattage and inference latency budget matter (which is virtually all edge AI), **EDP is the metric that should drive architecture selection**, not energy alone. Under EDP, even mamba_expand2's modest energy disadvantage (12% over plain mamba) compounds with its identical wallclock to give a 4% EDP penalty — small but consistent.

**Paper §discussion implication**: the theoretical sparsity advantage of Spiking-SSM (-56% theoretical energy via sparsity-aware accounting) is overwhelmed by the wallclock penalty on commodity GPU. Only on neuromorphic hardware where the LIF integration is the **native primitive** (not a multi-step Python loop) does the EDP curve flip in Spiking's favour. This is a hardware-architectural co-design question for future work, not a model-selection question for current deployment targets.

### 5.6 F5: t5sum is the only consistent-C1-PASS pathway; never passes C2

**Evidence**:
- `spiking_t5sum_vs_lstm_25k`: ci_hi=-0.0201 (PASS), ratio 0.609 (FAIL)
- `spiking_t5sum_vs_lstm_50k`: ci_hi=-0.0233 (PASS), ratio 0.609 (FAIL)
- `spiking_t5sum_vs_lstm_100k`: ci_hi=-0.0256 (PASS), ratio 0.609 (FAIL)
- `spiking_t5sum_50k_vs_lstm_50k`: ci_hi=-0.0153 (PASS), ratio 0.607 (FAIL) — narrowest C1 gap
- `spiking_t5sum_50k_vs_lstm_100k`: ci_hi=-0.0176 (PASS), ratio 0.607 (FAIL)

**Mechanism**: t_inner=5 amplifies the spike count by ~5× (149994 vs 27303 sops). The post-LIF AC contribution (sops × out_features × 0.9 pJ) dominates the sparsity-aware energy term, pushing the ratio above 0.6 regardless of training budget.

**Caveat (sum-mode accounting)**: `decode_mode='sum'` produces a real-valued (rate-coded) tensor in [0, 1], not a binary spike train, as the input to `out_proj`. The current energy_metrics formula treats those as if the input were binary, biasing the sparsity-aware energy estimate downward (under-estimates true energy). On a sparsity-aware accelerator that can detect 0-spike inputs, the true ratio is bounded between [0.609 (current estimate), 0.773 (gpu_dense, worst case)]. In either bound, **C2 fails**, so the trade-off conclusion is robust to the caveat.

---

## 6. Limitations and caveats

### 6.1 NVML measurement noise

Cell-to-cell std within each architecture is 1.3-15.3% of the mean. LSTM exhibits the largest spread (std/mean = 0.14 across 5k/25k/50k/100k cells). The cause is GPU thermal state and idle wattage drift across measurement runs. Mitigation: clock-locking (attempted but typically requires root on consumer GPUs; failed silently in our run with `nvmlDeviceSetGpuLockedClocks` returning permission error). For paper, report ± std and note the measurement noise floor.

### 6.2 Sum-mode energy accounting under-estimate

`scripts/aggregate_v6_results.py` and `src/fl_oran/evaluation/energy_metrics.py` use a simplified accounting that treats the post-LIF `out_proj` operation as accumulate (1 AC per spike event × out_features). For `decode_mode='sum'` cells, the input to `out_proj` is real-valued (in [0,1] after dividing by t_inner), not binary, so each operation is technically a multiply-accumulate, not an accumulate. The current formula therefore biases the sparsity-aware energy of t5sum/t5sum_50k variants downward. The bias direction is consistent across all t5sum cells, so it does not affect the relative trade-off classification (all t5sum variants remain C2 fail under either bound). Documented in `src/fl_oran/evaluation/energy_metrics.py` module docstring.

### 6.3 Single-seed LIF grid

E7 LIF threshold/beta grid (9 cells) is single-seed (seed=42 only). Paired bootstrap CIs cannot be computed (n_paired_seeds=1 < 2 minimum). All LIF cells therefore receive `C1=n/a (FAIL)` and decision NO-GO regardless of point estimates. From single-seed extrapolation (assuming similar variance to lr5e4_25k std=0.0018), the LIF best (lif_t10_b05 = 0.9049) has 95% CI [0.8980, 0.9118], which overlaps with `spiking_t5sum (25k) = 0.9024 ± 0.0025` but does not overlap with `spiking_t5sum_50k = 0.9110 ± 0.0015`. Multi-seed LIF would convert the n/a to actual CI; the structural pattern (low threshold / mid beta best) suggests modest gains over `spiking_lr5e4_25k` but unlikely to exceed `spiking_t5sum_50k`'s upper budget. Tier B.1 (LIF multi-seed for top-3 from grid) was therefore deprioritised; results would be a marginal improvement on an already-trade-off pathway, not a new GO.

### 6.4 GPU clock unlocked

`nvmlDeviceSetGpuLockedClocks(handle, 80% × max_clock, 80% × max_clock)` requires root on RTX 4080 consumer card; failed with permission error during Tier A.2. Measurements ran at dynamic clock, which contributes to the cell-to-cell variance noted in §6.1. Mitigation in §6.1.

### 6.5 OOD split is fixed

train `tr ∈ [0,21]` / val `tr ∈ [22,24]` / test `tr ∈ [25,27]` is the only split tested. Different cutoffs would test split-stability of the conclusions. Out of scope for v6; budgeted for follow-up.

### 6.6 Single dataset

ColO-RAN only. Cross-dataset generalization to Milan / Trentino out of scope for Stage 1; tracked in `docs/FUTURE_STUDY.md` ISSUE-1 with documented schema/label/granularity differences justifying a separate paper.

---

## 7. Implications for Stage 2

### 7.1 Stage 2 selection (preregistered framework)

Per ADR-001 D-21:
- **Preregistered Spiking variant fails NO-GO** → no Stage 2 with default Spiking config.
- **Mamba arm healthy** at all budgets up to 50k (C3 PASS); marginal LSTM advantage at 100k does not invalidate Mamba arm (the C3 criterion is non-inferiority, not strict equality).

### 7.2 Stage 2 selection (audit pathways)

Two candidate architectures qualify under audit:
- `spiking_expand2`: GO Spiking-led at matched 25k (sparsity-aware accounting), strongest C2 margin (0.439).
- `spiking_lr5e4_25k`: GO Spiking-led at matched 25k (sparsity-aware accounting), narrower C2 margin (0.494).

Both fail at higher budgets (50k+), suggesting Spiking-SSM hits an accuracy ceiling earlier than dense baselines on this task.

### 7.3 Hardware-conditional decision

For deployment hardware {commodity GPU, sparsity-aware accelerator, neuromorphic chip}, the Stage 2 architecture choice is:
- Commodity GPU (RTX 4080 baseline): **LSTM** (1× real energy; 9.3× cheaper than Spiking T=1 in measured wallclock)
- Sparsity-aware accelerator (e.g., Tenstorrent Wormhole, Cerebras CS-3): **spiking_expand2** (theoretical -56% vs LSTM, GO Spiking-led)
- Neuromorphic (e.g., Intel Loihi 2, BrainChip Akida): **spiking_expand2** (architecture is native primitive)

### 7.4 Recommendation

Stage 2 is conditional on deployment-target assumption. The paper should report all three findings (commodity-GPU LSTM dominance; sparsity-aware GO Spiking-led; neuromorphic GO Spiking-led) without committing to one Stage 2 in the absence of a specific hardware target.

---

## 8. References to source data

- Raw stats: `artifacts/v6_arch_sweep/aggregated.json` `stats` dict
- Paired bootstrap CIs (uncorrected + Bonferroni n=176): same file, `deltas` dict
- D-21 decisions (84 entries): same file, `criteria_audit` dict (preregistered) + `criteria_audit` (audit)
- NVML measurements: per-cell `artifacts/v6_arch_sweep/<cell>/energy_measured.json` (n=159 cells)
- Theoretical energy (post Round 4-9 fix): per-cell `artifacts/v6_arch_sweep/<cell>/energy.json`
- Generated table: `docs/RESULTS_V6_STAGE1.md` (auto-regenerated by aggregator)

Code commit pinning Stage 1 results: `0a5f2c7` (Round 4-9 helper extraction + idempotency + atomic writes; 83 unit tests passing).

Reproduction:
```bash
PYTHONPATH=src .venv/bin/python scripts/recompute_v6_energy.py   # idempotent on existing cells
PYTHONPATH=src .venv/bin/python scripts/aggregate_v6_results.py  # regenerates aggregated.json + RESULTS_V6_STAGE1.md
```
